package io.kotensor.core

import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class PromptPrefixTest {

    private fun seededModel(seed: Int = 41): Transformer {
        val model = Transformer(vocabularySize = 50, dimension = 32, numLayers = 2, numHeads = 4, hiddenDimension = 96, numKeyValueHeads = 2)
        val random = Random(seed)
        for (p in model.parameters()) {
            for (i in p.data.indices) p.data[i] = (random.nextFloat() * 2f - 1f) * 0.3f
        }
        return model
    }

    private fun tokens(length: Int, seed: Int): IntArray {
        val random = Random(seed)
        return IntArray(length) { random.nextInt(50) }
    }

    @Test
    fun testGenerationWithAPrefixMatchesGenerationWithout() {
        for (model in listOf(seededModel(), seededModel().quantize(Quantization.INT4))) {
            val system = tokens(40, 1)
            val prefix = model.cachePrefix(system)
            assertEquals(40, prefix.length)
            for (prompt in listOf(system + tokens(9, 2), system.copyOfRange(0, 25) + tokens(30, 3), tokens(12, 4), system.copyOf())) {
                val expected = model.generate(prompt, maxTokens = 10)
                assertContentEquals(expected, model.generate(prompt, maxTokens = 10, stopTokens = emptySet(), prefix = prefix))
                assertContentEquals(expected.copyOfRange(prompt.size, expected.size), model.generateStream(prompt, 10, emptySet(), prefix).toList().toIntArray())
            }
        }
    }

    @Test
    fun testPrefixIsReusableAndUnchanged() {
        val model = seededModel()
        val system = tokens(33, 5)
        val prefix = model.cachePrefix(system)
        val first = model.generate(system + tokens(5, 6), 6, emptySet(), prefix)
        model.generate(system + tokens(20, 7), 6, emptySet(), prefix)
        assertContentEquals(first, model.generate(system + tokens(5, 6), 6, emptySet(), prefix))
    }

    @Test
    fun testMatchLengthLeavesTheLastPromptTokenToPrefill() {
        val model = seededModel()
        val prefix = model.cachePrefix(intArrayOf(1, 2, 3, 4))
        assertEquals(4, prefix.matchLength(intArrayOf(1, 2, 3, 4, 5)))
        assertEquals(3, prefix.matchLength(intArrayOf(1, 2, 3, 4)))
        assertEquals(2, prefix.matchLength(intArrayOf(1, 2, 9, 4, 5)))
        assertEquals(0, prefix.matchLength(intArrayOf(7, 2, 3)))
        assertEquals(0, prefix.matchLength(intArrayOf(1)))
    }

    @Test
    fun testCachedPrefixEqualsAPrefilledCache() {
        val model = seededModel()
        val system = tokens(37, 8)
        val prefix = model.cachePrefix(system)
        val cache = model.newCache(37)
        model.prefill(system, cache, model.newInferenceBuffers(37))
        val copy = model.newCache(60)
        copy.copyFrom(cache, 37)
        assertEquals(37, copy.length)
        for (layer in 0 until model.numLayers) {
            for (head in 0 until model.numKeyValueHeads) {
                for (position in 0 until 37) {
                    for (d in 0 until model.headDim) {
                        assertEquals(cache.keys[cache.offset(layer, head, position) + d], copy.keys[copy.offset(layer, head, position) + d])
                        assertEquals(cache.values[cache.offset(layer, head, position) + d], copy.values[copy.offset(layer, head, position) + d])
                    }
                }
            }
        }
        assertTrue(prefix.sizeInBytes > 0)
    }

    @Test
    fun testCopyFromRejectsMismatchedCaches() {
        val model = seededModel()
        val source = model.newCache(10)
        model.prefill(tokens(10, 9), source, model.newInferenceBuffers(10))
        assertFailsWith<IllegalArgumentException> { model.newCache(20).copyFrom(source, 11) }
        assertFailsWith<IllegalArgumentException> { model.newCache(5).copyFrom(source, 6) }
        assertFailsWith<IllegalArgumentException> { KVCache(3, 2, 20, 8).copyFrom(source, 4) }
        val partial = model.newCache(20)
        partial.copyFrom(source, 4)
        assertEquals(4, partial.length)
    }

    @Test
    fun testPrefixFromAnotherModelIsRejected() {
        val prefix = seededModel().cachePrefix(tokens(8, 10))
        assertFailsWith<IllegalArgumentException> { seededModel().generate(tokens(12, 11), 3, emptySet(), prefix) }
    }

    @Test
    fun testPrefixMatchesALargePromptAcrossChunks() {
        val model = seededModel().quantize(Quantization.INT8)
        val system = tokens(70, 12)
        val prefix = model.cachePrefix(system)
        val prompt = system + tokens(45, 13)
        val cache = model.newCache(prompt.size)
        val reference = model.prefill(prompt, cache, model.newInferenceBuffers(prompt.size)).data.copyOf()
        val withPrefix = model.newCache(prompt.size)
        withPrefix.copyFrom(model.newCache(70).also { model.prefill(system, it, model.newInferenceBuffers(70)) }, 70)
        val logits = model.prefill(prompt.copyOfRange(70, prompt.size), withPrefix, model.newInferenceBuffers(prompt.size)).data
        var max = 0.0f
        for (i in reference.indices) max = maxOf(max, abs(reference[i] - logits[i]))
        assertTrue(max <= 1e-4f, "logits differ by $max")
    }
}
