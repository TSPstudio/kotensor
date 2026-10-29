package io.kotensor.core

import java.lang.management.ManagementFactory
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class InferenceTest {

    private fun seededModel(numLayers: Int = 2, seed: Int = 21): Transformer {
        val model = Transformer(vocabularySize = 40, dimension = 32, numLayers = numLayers, numHeads = 4, hiddenDimension = 96)
        val random = Random(seed)
        for (p in model.parameters()) {
            for (i in p.data.indices) p.data[i] = (random.nextFloat() * 2f - 1f) * 0.3f
        }
        return model
    }

    @Test
    fun testDecodeMatchesGraphForward() {
        val model = seededModel()
        val sequence = intArrayOf(5, 17, 3, 39, 0, 22, 8)

        val graphCache = model.newCache(sequence.size)
        val bufferCache = model.newCache(sequence.size)
        val buffers = model.newInferenceBuffers(sequence.size)
        val full = model.forward(sequence)

        for (t in sequence.indices) {
            val expected = model.forward(intArrayOf(sequence[t]), graphCache)
            val actual = model.decode(sequence[t], bufferCache, buffers)
            assertTrue(actual === buffers.logits)
            assertEquals(t + 1, bufferCache.length)
            for (v in 0 until model.vocabularySize) {
                assertEquals(expected.data[v], actual.data[v], 1e-5f, "step $t vocab $v (graph vs decode)")
                assertEquals(full.data[t * model.vocabularySize + v], actual.data[v], 1e-4f, "step $t vocab $v (full vs decode)")
            }
        }
    }

    @Test
    fun testDecodeMatchesGraphForwardWithLoRA() {
        val model = seededModel(numLayers = 1)
        val adapter = LoRAAdapter(model, rank = 3, alpha = 2.0f)
        val random = Random(4)
        for (p in adapter.parameters()) {
            for (i in p.data.indices) p.data[i] = (random.nextFloat() * 2f - 1f) * 0.3f
        }

        val sequence = intArrayOf(1, 2, 3, 4)
        val graphCache = model.newCache(sequence.size)
        val bufferCache = model.newCache(sequence.size)
        val buffers = model.newInferenceBuffers(sequence.size)
        for (token in sequence) {
            val expected = model.forward(intArrayOf(token), graphCache)
            val actual = model.decode(token, bufferCache, buffers)
            for (v in 0 until model.vocabularySize) {
                assertEquals(expected.data[v], actual.data[v], 1e-5f)
            }
        }
    }

    @Test
    fun testInPlaceModulesMatchGraphModules() {
        val random = Random(8)
        val input = Tensor(FloatArray(32) { random.nextFloat() - 0.5f }, intArrayOf(1, 32))

        val norm = RMSNorm(32)
        for (i in norm.weight.data.indices) norm.weight.data[i] = 0.5f + random.nextFloat()
        val normOut = Tensor(FloatArray(32), intArrayOf(1, 32))
        norm.forwardInto(input, normOut)
        val normRef = norm(input)
        for (i in 0 until 32) assertEquals(normRef.data[i], normOut.data[i], 1e-6f)

        val ffn = SwiGLU(32, 48)
        val ffnOut = Tensor(FloatArray(32), intArrayOf(1, 32))
        ffn.forwardInto(input, ffnOut, Tensor(FloatArray(48), intArrayOf(1, 48)), Tensor(FloatArray(48), intArrayOf(1, 48)))
        val ffnRef = ffn(input)
        for (i in 0 until 32) assertEquals(ffnRef.data[i], ffnOut.data[i], 1e-5f)

        val rope = RoPE(8)
        val rotated = Tensor(input.data.copyOf(), intArrayOf(1, 32))
        rope.applyInPlace(rotated, position = 6)
        val ropeRef = rope.apply(input, startPosition = 6)
        for (i in 0 until 32) assertEquals(ropeRef.data[i], rotated.data[i], 1e-6f)

        val embedding = Embedding(10, 32)
        val row = Tensor(FloatArray(32), intArrayOf(1, 32))
        embedding.lookupInto(7, row)
        val embeddingRef = embedding.forward(intArrayOf(7))
        for (i in 0 until 32) assertEquals(embeddingRef.data[i], row.data[i])
    }

    @Test
    fun testDecodeStepAllocatesNothing() {
        val threadBean = ManagementFactory.getThreadMXBean() as? com.sun.management.ThreadMXBean ?: return
        if (!threadBean.isThreadAllocatedMemorySupported) return
        threadBean.isThreadAllocatedMemoryEnabled = true

        val model = seededModel()
        val cache = model.newCache(4096)
        val buffers = model.newInferenceBuffers(4096)
        var token = 3
        repeat(2000) {
            if (cache.remaining == 0) cache.reset()
            val logits = model.decode(token, cache, buffers)
            token = (token + 7) % model.vocabularySize + (if (logits.data[0] > 1e9f) 1 else 0)
        }

        cache.reset()
        val before = threadBean.currentThreadAllocatedBytes
        repeat(200) {
            val logits = model.decode(token, cache, buffers)
            token = (token + 7) % model.vocabularySize + (if (logits.data[0] > 1e9f) 1 else 0)
        }
        val allocated = threadBean.currentThreadAllocatedBytes - before
        assertEquals(0L, allocated, "bytes allocated by 200 decode steps")
    }

    @Test
    fun testDecodeChecksCacheAndBuffers() {
        val model = seededModel()
        val cache = model.newCache(2)
        val buffers = model.newInferenceBuffers(2)
        model.decode(1, cache, buffers)
        model.decode(2, cache, buffers)
        assertFailsWith<IllegalArgumentException> { model.decode(3, cache, buffers) }

        val otherBuffers = Transformer(vocabularySize = 40, dimension = 16, numLayers = 1, numHeads = 2).newInferenceBuffers(8)
        assertFailsWith<IllegalArgumentException> { model.decode(1, model.newCache(8), otherBuffers) }

        val shortBuffers = model.newInferenceBuffers(1)
        val longCache = model.newCache(8)
        model.decode(1, longCache, shortBuffers)
        assertFailsWith<IllegalArgumentException> { model.decode(1, longCache, shortBuffers) }

        assertEquals((6L * 32 + 2L * 32 + 2L * 96 + 40 + 2) * 4L, buffers.sizeInBytes)
    }
}

