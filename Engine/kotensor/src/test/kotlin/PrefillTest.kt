package io.kotensor.core

import java.lang.management.ManagementFactory
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertSame
import kotlin.test.assertTrue

class PrefillTest {

    private fun seededModel(vocabularySize: Int = 40, dimension: Int = 32, numLayers: Int = 2, numHeads: Int = 4, numKeyValueHeads: Int = 2, hiddenDimension: Int = 96, seed: Int = 23): Transformer {
        val model = Transformer(vocabularySize, dimension, numLayers, numHeads, hiddenDimension, numKeyValueHeads)
        val random = Random(seed)
        for (p in model.parameters()) {
            for (i in p.data.indices) p.data[i] = (random.nextFloat() * 2f - 1f) * 0.3f
        }
        return model
    }

    private fun prompt(length: Int, vocabularySize: Int, seed: Int = 5): IntArray {
        val random = Random(seed)
        return IntArray(length) { random.nextInt(vocabularySize) }
    }

    private fun maxAbsDiff(a: FloatArray, aOffset: Int, b: FloatArray, bOffset: Int, length: Int): Float {
        var max = 0.0f
        for (i in 0 until length) max = maxOf(max, abs(a[aOffset + i] - b[bOffset + i]))
        return max
    }

    private fun assertPrefillMatchesGraph(model: Transformer, tokens: IntArray, tolerance: Float) {
        val graphCache = model.newCache(tokens.size + 1)
        val graph = model.forward(tokens, graphCache)

        val cache = model.newCache(tokens.size + 1)
        val buffers = model.newInferenceBuffers(tokens.size + 1)
        val logits = model.prefill(tokens, cache, buffers)

        assertSame(buffers.logits, logits)
        assertEquals(tokens.size, cache.length)
        val lastRow = (tokens.size - 1) * model.vocabularySize
        val diff = maxAbsDiff(graph.data, lastRow, logits.data, 0, model.vocabularySize)
        assertTrue(diff <= tolerance, "prompt of ${tokens.size}: logits differ by $diff")
        assertTrue(maxAbsDiff(graphCache.keys, 0, cache.keys, 0, cache.keys.size) <= tolerance, "prompt of ${tokens.size}: cached keys differ")
        assertTrue(maxAbsDiff(graphCache.values, 0, cache.values, 0, cache.values.size) <= tolerance, "prompt of ${tokens.size}: cached values differ")
    }

    @Test
    fun testPrefillMatchesGraphForwardAcrossChunkBoundaries() {
        val model = seededModel()
        for (length in intArrayOf(1, 2, 31, 32, 33, 64, 70)) {
            assertPrefillMatchesGraph(model, prompt(length, model.vocabularySize, seed = length), 1e-4f)
        }
    }

    @Test
    fun testPrefillMatchesGraphForwardOnQuantizedModels() {
        for (quantization in listOf(Quantization.INT8, Quantization.INT4)) {
            val model = seededModel().quantize(quantization)
            for (length in intArrayOf(1, 33, 40)) {
                assertPrefillMatchesGraph(model, prompt(length, model.vocabularySize, seed = length), 1e-4f)
            }
        }
    }

    @Test
    fun testPrefillMatchesGraphForwardWithLoRA() {
        val model = seededModel(numLayers = 1)
        val adapter = LoRAAdapter(model, rank = 3, alpha = 2.0f)
        val random = Random(9)
        for (p in adapter.parameters()) {
            for (i in p.data.indices) p.data[i] = (random.nextFloat() * 2f - 1f) * 0.3f
        }
        assertPrefillMatchesGraph(model, prompt(37, model.vocabularySize), 1e-4f)
    }

    @Test
    fun testPrefillContinuesAnExistingCache() {
        val model = seededModel()
        val tokens = prompt(50, model.vocabularySize)
        val graphCache = model.newCache(tokens.size)
        val graph = model.forward(tokens, graphCache)

        val cache = model.newCache(tokens.size)
        val buffers = model.newInferenceBuffers(tokens.size)
        model.prefill(tokens.copyOfRange(0, 20), cache, buffers)
        val logits = model.prefill(tokens.copyOfRange(20, tokens.size), cache, buffers)

        assertEquals(tokens.size, cache.length)
        val lastRow = (tokens.size - 1) * model.vocabularySize
        assertTrue(maxAbsDiff(graph.data, lastRow, logits.data, 0, model.vocabularySize) <= 1e-4f)
    }

    @Test
    fun testDecodeAfterPrefillMatchesGraphDecoding() {
        val model = seededModel().quantize(Quantization.INT4)
        val tokens = prompt(35, model.vocabularySize)
        val continuation = intArrayOf(3, 17, 8, 29)

        val graphCache = model.newCache(tokens.size + continuation.size)
        model.forward(tokens, graphCache)
        val cache = model.newCache(tokens.size + continuation.size)
        val buffers = model.newInferenceBuffers(tokens.size + continuation.size)
        model.prefill(tokens, cache, buffers)

        for (token in continuation) {
            val expected = model.forward(intArrayOf(token), graphCache)
            val actual = model.decode(token, cache, buffers)
            assertTrue(maxAbsDiff(expected.data, 0, actual.data, 0, model.vocabularySize) <= 1e-4f)
        }
    }

    @Test
    fun testGenerateMatchesGraphGreedyDecoding() {
        for (model in listOf(seededModel(), seededModel().quantize(Quantization.INT4))) {
            val tokens = prompt(45, model.vocabularySize)
            val maxTokens = 12
            val cache = model.newCache(tokens.size + maxTokens)
            var logits = model.forward(tokens, cache)
            val expected = ArrayList<Int>()
            for (step in 0 until maxTokens) {
                val offset = (logits.data.size / model.vocabularySize - 1) * model.vocabularySize
                var best = 0
                for (v in 0 until model.vocabularySize) {
                    if (logits.data[offset + v] > logits.data[offset + best]) best = v
                }
                expected.add(best)
                if (step < maxTokens - 1) logits = model.forward(intArrayOf(best), cache)
            }
            val generated = model.generate(tokens, maxTokens)
            assertContentEquals(tokens, generated.copyOfRange(0, tokens.size))
            assertContentEquals(expected.toIntArray(), generated.copyOfRange(tokens.size, generated.size))
        }
    }

    @Test
    fun testPrefillRejectsMismatchedCacheAndBuffers() {
        val model = seededModel()
        val tokens = prompt(10, model.vocabularySize)
        assertFailsWith<IllegalArgumentException> { model.prefill(tokens, model.newCache(9), model.newInferenceBuffers(10)) }
        assertFailsWith<IllegalArgumentException> { model.prefill(tokens, model.newCache(10), model.newInferenceBuffers(9)) }
        assertFailsWith<IllegalArgumentException> { model.prefill(IntArray(0), model.newCache(10), model.newInferenceBuffers(10)) }
        val other = seededModel(hiddenDimension = 64)
        assertFailsWith<IllegalArgumentException> { model.prefill(tokens, model.newCache(10), other.newInferenceBuffers(10)) }
    }

    @Test
    fun testPrefillMemoryDoesNotGrowWithLayersHiddenSizeOrVocabulary() {
        val threadBean = ManagementFactory.getThreadMXBean() as? com.sun.management.ThreadMXBean ?: return
        if (!threadBean.isThreadAllocatedMemorySupported) return
        threadBean.isThreadAllocatedMemoryEnabled = true

        val model = seededModel(vocabularySize = 500, dimension = 32, numLayers = 4, hiddenDimension = 256).quantize(Quantization.INT4)
        val short = prompt(64, model.vocabularySize)
        val long = prompt(512, model.vocabularySize)

        fun prefillBytes(tokens: IntArray): Long {
            val cache = model.newCache(tokens.size)
            val buffers = model.newInferenceBuffers(tokens.size)
            val before = threadBean.currentThreadAllocatedBytes
            model.prefill(tokens, cache, buffers)
            return threadBean.currentThreadAllocatedBytes - before
        }

        fun graphBytes(tokens: IntArray): Long {
            val cache = model.newCache(tokens.size)
            val before = threadBean.currentThreadAllocatedBytes
            model.forward(tokens, cache)
            return threadBean.currentThreadAllocatedBytes - before
        }

        repeat(3) { prefillBytes(short) }
        val shortBytes = prefillBytes(short)
        val longBytes = prefillBytes(long)
        val perToken = (longBytes - shortBytes) / (long.size - short.size)
        assertTrue(perToken < 1024, "prefill allocates $perToken bytes per prompt token")
        assertTrue(longBytes * 10 < graphBytes(long), "prefill allocated $longBytes bytes for ${long.size} tokens")
    }

    @Test
    fun testNoGradDropsTheGraphAndRestoresRecording() {
        val a = Tensor(floatArrayOf(1f, 2f), intArrayOf(1, 2))
        val b = Tensor(floatArrayOf(3f, 4f), intArrayOf(1, 2))
        val recorded = a + b
        assertEquals(2, recorded.parents.size)
        assertTrue(recorded.requiresGrad)

        val (inside, nested) = noGrad {
            val leaf = Tensor(FloatArray(2), intArrayOf(1, 2))
            assertTrue(leaf.requiresGrad)
            val sum = a + b
            val product = noGrad { sum * a }
            sum to product
        }
        assertTrue(inside.parents.isEmpty())
        assertFalse(inside.requiresGrad)
        assertTrue(nested.parents.isEmpty())
        assertContentEquals(floatArrayOf(4f, 6f), inside.data)

        val after = a * b
        assertEquals(2, after.parents.size)
        after.backward()
        assertContentEquals(floatArrayOf(3f, 4f), a.grad)
    }

    @Test
    fun testNoGradRestoresRecordingAfterAnException() {
        val a = Tensor(floatArrayOf(1f), intArrayOf(1, 1))
        assertFailsWith<IllegalStateException> { noGrad<Unit> { throw IllegalStateException("boom") } }
        assertTrue((a * a).parents.isNotEmpty())
    }

    @Test
    fun testFrozenOperandsKeepNoParents() {
        val a = Tensor(floatArrayOf(1f, 2f), intArrayOf(1, 2)).also { it.requiresGrad = false }
        val b = Tensor(floatArrayOf(3f, 4f), intArrayOf(1, 2)).also { it.requiresGrad = false }
        val sum = a + b
        assertFalse(sum.requiresGrad)
        assertTrue(sum.parents.isEmpty())
    }

    @Test
    fun testCachedForwardKeepsNoGraph() {
        val model = seededModel()
        val logits = model.forward(prompt(9, model.vocabularySize), model.newCache(9))
        assertTrue(logits.parents.isEmpty())
        assertFalse(logits.requiresGrad)
        assertTrue(model.forward(prompt(9, model.vocabularySize)).parents.isNotEmpty())
    }

    @Test
    fun testLoRAForwardIntoHandlesSeveralRows() {
        val base = Linear(6, 5)
        val lora = LoRALinear(base, rank = 2, alpha = 1.5f)
        val random = Random(3)
        for (p in lora.parameters()) {
            for (i in p.data.indices) p.data[i] = random.nextFloat() - 0.5f
        }
        val input = Tensor(FloatArray(18) { random.nextFloat() - 0.5f }, intArrayOf(3, 6))
        val expected = lora.forward(input)
        val out = Tensor(FloatArray(15), intArrayOf(3, 5))
        lora.forwardInto(input, out)
        assertTrue(maxAbsDiff(expected.data, 0, out.data, 0, 15) <= 1e-6f)
    }

    @Test
    fun testRotatingRowsInPlaceMatchesApply() {
        val rope = RoPE(8, 10000.0f)
        val random = Random(2)
        val rows = Tensor(FloatArray(5 * 16) { random.nextFloat() - 0.5f }, intArrayOf(5, 16))
        val expected = rope.apply(rows, 7)
        rope.applyRowsInPlace(rows, 7)
        assertTrue(maxAbsDiff(expected.data, 0, rows.data, 0, rows.data.size) <= 1e-6f)
    }
}
