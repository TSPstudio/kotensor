package io.kotensor.core

import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContentEquals

class ParallelAttentionTest {

    private fun seededModel(): Transformer {
        val model = Transformer(vocabularySize = 60, dimension = 48, numLayers = 2, numHeads = 6, hiddenDimension = 96, numKeyValueHeads = 2)
        val random = Random(19)
        for (p in model.parameters()) {
            for (i in p.data.indices) p.data[i] = (random.nextFloat() * 2f - 1f) * 0.3f
        }
        return model
    }

    private fun <T> withThreshold(threshold: Long, block: () -> T): T {
        val saved = MatMul.parallelThreshold
        MatMul.parallelThreshold = threshold
        try {
            return block()
        } finally {
            MatMul.parallelThreshold = saved
        }
    }

    private fun run(model: Transformer, prompt: IntArray): List<FloatArray> {
        val cache = model.newCache(prompt.size + 4)
        val buffers = model.newInferenceBuffers(prompt.size + 4)
        val result = ArrayList<FloatArray>()
        result.add(model.prefill(prompt, cache, buffers).data.copyOf())
        for (token in intArrayOf(5, 17, 33, 2)) result.add(model.decode(token, cache, buffers).data.copyOf())
        val graphCache = model.newCache(prompt.size)
        result.add(model.forward(prompt, graphCache).data.copyOf())
        return result
    }

    @Test
    fun testParallelAttentionEqualsSerialAttention() {
        val random = Random(3)
        val prompt = IntArray(90) { random.nextInt(60) }
        for (model in listOf(seededModel(), seededModel().quantize(Quantization.INT4))) {
            val serial = withThreshold(Long.MAX_VALUE) { run(model, prompt) }
            val parallel = withThreshold(0L) { run(model, prompt) }
            for (i in serial.indices) assertContentEquals(serial[i], parallel[i], "output $i")
        }
    }

    @Test
    fun testParallelGenerationEqualsSerialGeneration() {
        val random = Random(4)
        val prompt = IntArray(70) { random.nextInt(60) }
        val model = seededModel().quantize(Quantization.INT8)
        val serial = withThreshold(Long.MAX_VALUE) { model.generate(prompt, maxTokens = 12) }
        val parallel = withThreshold(0L) { model.generate(prompt, maxTokens = 12) }
        assertContentEquals(serial, parallel)
    }
}
