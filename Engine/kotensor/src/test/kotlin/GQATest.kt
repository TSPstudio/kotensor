package io.kotensor.core

import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class GQATest {

    private fun seed(model: Transformer, seed: Int) {
        val random = Random(seed)
        for (p in model.parameters()) {
            for (i in p.data.indices) p.data[i] = (random.nextFloat() * 2f - 1f) * 0.3f
        }
    }

    private fun replicatedMultiHeadTwin(gqa: Transformer): Transformer {
        val mha = Transformer(gqa.vocabularySize, gqa.dimension, gqa.numLayers, gqa.numHeads, gqa.hiddenDimension)
        val headDim = gqa.headDim
        val group = gqa.numHeads / gqa.numKeyValueHeads
        val targets = mha.namedParameters()
        mha.loadParameters(gqa.namedParameters().filter { (name, tensor) -> targets.getValue(name).data.size == tensor.data.size })
        for (l in gqa.blocks.indices) {
            val source = gqa.blocks[l].attention
            val target = mha.blocks[l].attention
            for ((from, to) in listOf(source.wK to target.wK, source.wV to target.wV)) {
                val fromLinear = from as Linear
                val toLinear = to as Linear
                for (row in 0 until gqa.dimension) {
                    for (h in 0 until gqa.numHeads) {
                        val kvHead = h / group
                        for (d in 0 until headDim) {
                            toLinear.weight.data[row * gqa.dimension + h * headDim + d] =
                                fromLinear.weight.data[row * source.kvDimension + kvHead * headDim + d]
                        }
                    }
                }
                for (h in 0 until gqa.numHeads) {
                    val kvHead = h / group
                    for (d in 0 until headDim) {
                        toLinear.bias.data[h * headDim + d] = fromLinear.bias.data[kvHead * headDim + d]
                    }
                }
            }
        }
        return mha
    }

    @Test
    fun testGroupedQueryAttentionEqualsReplicatedMultiHead() {
        val gqa = Transformer(vocabularySize = 20, dimension = 24, numLayers = 2, numHeads = 6, hiddenDimension = 40, numKeyValueHeads = 2)
        seed(gqa, 3)
        assertEquals(8, gqa.blocks[0].attention.kvDimension)
        assertEquals(3, gqa.blocks[0].attention.groupSize)
        val mha = replicatedMultiHeadTwin(gqa)

        val tokens = intArrayOf(4, 9, 1, 17, 0, 12)
        val expected = mha.forward(tokens)
        val actual = gqa.forward(tokens)
        for (i in expected.data.indices) {
            assertEquals(expected.data[i], actual.data[i], 1e-5f, "logit $i")
        }

        val batch = Tensor(FloatArray(12) { i -> tokens[i % 6].toFloat() }, intArrayOf(2, 6))
        val expectedBatch = mha.forward(batch)
        val actualBatch = gqa.forward(batch)
        for (i in expectedBatch.data.indices) {
            assertEquals(expectedBatch.data[i], actualBatch.data[i], 1e-5f)
        }
    }

    @Test
    fun testGroupedQueryGradientsSumOverTheGroup() {
        val gqa = Transformer(vocabularySize = 12, dimension = 16, numLayers = 1, numHeads = 4, hiddenDimension = 32, numKeyValueHeads = 2)
        seed(gqa, 11)
        val mha = replicatedMultiHeadTwin(gqa)
        val inputs = intArrayOf(1, 5, 3, 8, 2)
        val targets = intArrayOf(5, 3, 8, 2, 0)

        crossEntropyLoss(gqa.forward(inputs), targets).backward()
        crossEntropyLoss(mha.forward(inputs), targets).backward()

        val headDim = gqa.headDim
        val gqaK = gqa.blocks[0].attention.wK as Linear
        val mhaK = mha.blocks[0].attention.wK as Linear
        for (row in 0 until gqa.dimension) {
            for (kvHead in 0 until gqa.numKeyValueHeads) {
                for (d in 0 until headDim) {
                    var summed = 0.0f
                    for (h in 0 until gqa.numHeads) {
                        if (h / 2 == kvHead) summed += mhaK.weight.grad[row * 16 + h * headDim + d]
                    }
                    assertEquals(summed, gqaK.weight.grad[row * 8 + kvHead * headDim + d], 1e-5f, "dK[$row, $kvHead, $d]")
                }
            }
        }
        val gqaQ = gqa.blocks[0].attention.wQ as Linear
        val mhaQ = mha.blocks[0].attention.wQ as Linear
        for (i in gqaQ.weight.grad.indices) {
            assertEquals(mhaQ.weight.grad[i], gqaQ.weight.grad[i], 1e-5f)
        }
    }

    @Test
    fun testCachedAndDecodePathsMatchWithGroupedHeads() {
        val model = Transformer(vocabularySize = 30, dimension = 32, numLayers = 2, numHeads = 8, hiddenDimension = 48, numKeyValueHeads = 2)
        seed(model, 5)
        val sequence = intArrayOf(7, 3, 29, 0, 14, 14, 2)
        val full = model.forward(sequence)

        val cache = model.newCache(sequence.size)
        assertEquals(2, cache.numHeads)
        assertEquals(4, cache.headDim)
        val prefill = model.forward(sequence.sliceArray(0 until 2), cache)
        for (t in 0 until 2) {
            for (v in 0 until 30) assertEquals(full.data[t * 30 + v], prefill.data[t * 30 + v], 1e-4f)
        }
        for (t in 2 until 4) {
            val step = model.forward(intArrayOf(sequence[t]), cache)
            for (v in 0 until 30) assertEquals(full.data[t * 30 + v], step.data[v], 1e-4f)
        }

        val buffers = model.newInferenceBuffers(sequence.size)
        assertEquals(8, buffers.kvDimension)
        for (t in 4 until sequence.size) {
            val step = model.decode(sequence[t], cache, buffers)
            for (v in 0 until 30) assertEquals(full.data[t * 30 + v], step.data[v], 1e-4f)
        }

        val greedy = model.generate(intArrayOf(7, 3), maxTokens = 5)
        assertEquals(7, greedy.size)
    }

    @Test
    fun testTrainingWithGroupedHeadsConverges() {
        val model = Transformer(vocabularySize = 8, dimension = 16, numLayers = 1, numHeads = 4, hiddenDimension = 32, numKeyValueHeads = 1)
        val optimizer = AdamW(model.parameters(), learningRate = 0.05f)
        val inputs = intArrayOf(0, 1, 2, 3, 4)
        val targets = intArrayOf(1, 2, 3, 4, 5)
        var first = 0.0f
        var last = 0.0f
        for (epoch in 0 until 30) {
            val loss = crossEntropyLoss(model.forward(inputs), targets)
            if (epoch == 0) first = loss.data[0]
            last = loss.data[0]
            loss.backward()
            optimizer.step()
            optimizer.zeroGrad()
        }
        assertTrue(last < first)
        assertTrue(model.namedParameters().containsKey("block.0.attention.wK.weight"))
        assertEquals(16 * 4, (model.blocks[0].attention.wK as Linear).weight.data.size)
    }

    @Test
    fun testInvalidHeadGroupingIsRejected() {
        assertFailsWith<IllegalArgumentException> { Attention(dimension = 16, numHeads = 4, numKeyValueHeads = 3) }
        assertFailsWith<IllegalArgumentException> { Attention(dimension = 16, numHeads = 4, numKeyValueHeads = 8) }

        val model = Transformer(vocabularySize = 8, dimension = 16, numLayers = 1, numHeads = 4, numKeyValueHeads = 2)
        val wrongCache = KVCache(numLayers = 1, numHeads = 4, maxSequenceLength = 4, headDim = 4)
        assertFailsWith<IllegalArgumentException> { model.forward(intArrayOf(1), wrongCache) }
    }

    @Test
    fun testRoPEThetaAndLlama3Scaling() {
        val plain = RoPE(8)
        val wide = RoPE(8, base = 500000.0f)
        assertEquals(1.0f, plain.invFreq[0])
        assertTrue(wide.invFreq[3] < plain.invFreq[3])

        val scaling = RoPEScaling(factor = 8.0f, lowFrequencyFactor = 1.0f, highFrequencyFactor = 4.0f, originalMaxPositionEmbeddings = 8192)
        val scaled = RoPE(64, base = 500000.0f, scaling = scaling)
        val unscaled = RoPE(64, base = 500000.0f)
        assertEquals(unscaled.invFreq[0], scaled.invFreq[0])
        assertEquals(unscaled.invFreq[31] / 8.0f, scaled.invFreq[31], 1e-9f)
        var smoothed = 0
        for (i in 0 until 32) {
            val ratio = scaled.invFreq[i] / unscaled.invFreq[i]
            assertTrue(ratio in 0.125f - 1e-6f..1.0f + 1e-6f)
            if (ratio > 0.125f + 1e-4f && ratio < 1.0f - 1e-4f) smoothed++
        }
        assertTrue(smoothed > 0)
    }
}

