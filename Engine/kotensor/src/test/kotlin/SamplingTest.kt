package io.kotensor.core

import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class SamplingTest {

    private fun seededModel(seed: Int = 21): Transformer {
        val model = Transformer(vocabularySize = 40, dimension = 32, numLayers = 2, numHeads = 4, hiddenDimension = 96)
        val random = Random(seed)
        for (p in model.parameters()) {
            for (i in p.data.indices) p.data[i] = (random.nextFloat() * 2f - 1f) * 0.3f
        }
        return model
    }

    @Test
    fun testGreedySamplingMatchesDefaultGeneration() {
        val model = seededModel()
        val prompt = intArrayOf(3, 14, 15, 9)
        val expected = model.generate(prompt, 12)
        assertContentEquals(expected, model.generate(prompt, 12, emptySet(), Sampling.GREEDY))
        assertContentEquals(expected, model.generate(prompt, 12, emptySet(), Sampling(temperature = 1.0f, topK = 1, repetitionPenalty = 1.0f)))
        val prefix = model.cachePrefix(intArrayOf(3, 14))
        assertContentEquals(expected, model.generate(prompt, 12, emptySet(), prefix, Sampling.GREEDY))
    }

    @Test
    fun testSeededSamplingIsReproducible() {
        val model = seededModel()
        val prompt = intArrayOf(1, 2, 3)
        val sampling = Sampling(temperature = 1.2f, topK = 0, topP = 1.0f, repetitionPenalty = 1.0f, seed = 99L)
        val first = model.generate(prompt, 16, emptySet(), sampling)
        val second = model.generate(prompt, 16, emptySet(), sampling)
        assertContentEquals(first, second)
        assertEquals(prompt.size + 16, first.size)
        for (token in first) assertTrue(token in 0 until model.vocabularySize)
    }

    @Test
    fun testTopKKeepsOnlyTheBestCandidates() {
        val logits = floatArrayOf(0.0f, 5.0f, 1.0f, 4.0f, 3.0f, 4.5f, -2.0f)
        val sampler = Sampling(temperature = 1.0f, topK = 3, topP = 1.0f, repetitionPenalty = 1.0f, seed = 5L).newSampler(logits.size)
        val counts = IntArray(logits.size)
        repeat(3000) { counts[sampler.next(logits.copyOf())]++ }
        assertEquals(0, counts[0] + counts[2] + counts[4] + counts[6])
        assertTrue(counts[1] > counts[5] && counts[5] > counts[3] && counts[3] > 0, counts.contentToString())
    }

    @Test
    fun testTopPDropsTheTail() {
        val logits = floatArrayOf(10.0f, 6.0f, 5.0f, 0.0f)
        val sampler = Sampling(temperature = 1.0f, topK = 0, topP = 0.9f, repetitionPenalty = 1.0f, seed = 1L).newSampler(logits.size)
        repeat(500) { assertEquals(0, sampler.next(logits.copyOf())) }
        val wide = Sampling(temperature = 1.0f, topK = 0, topP = 1.0f, repetitionPenalty = 1.0f, seed = 1L).newSampler(logits.size)
        val seen = HashSet<Int>()
        repeat(4000) { seen.add(wide.next(logits.copyOf())) }
        assertTrue(seen.size >= 2)
    }

    @Test
    fun testRepetitionPenaltyLowersRecentTokens() {
        val greedy = Sampling(temperature = 0.0f, repetitionPenalty = 1.2f, repetitionWindow = 4)
        val sampler = greedy.newSampler(3)
        assertEquals(0, sampler.next(floatArrayOf(2.0f, 1.9f, -1.0f)))
        assertEquals(1, sampler.next(floatArrayOf(2.0f, 1.9f, -1.0f), intArrayOf(0, 0, 0)))
        val logits = floatArrayOf(-0.5f, -0.55f, -3.0f)
        assertEquals(1, sampler.next(logits, intArrayOf(5, 0)))
        assertEquals(-0.6f, logits[0], 1e-6f)
        assertEquals(0, sampler.next(floatArrayOf(2.0f, 1.9f, -1.0f), intArrayOf(0, 1, 1, 1, 1), 5))
    }

    @Test
    fun testInvalidSettingsAreRejected() {
        assertFailsWith<IllegalArgumentException> { Sampling(temperature = -0.1f) }
        assertFailsWith<IllegalArgumentException> { Sampling(topK = -1) }
        assertFailsWith<IllegalArgumentException> { Sampling(topP = 0.0f) }
        assertFailsWith<IllegalArgumentException> { Sampling(topP = 1.5f) }
        assertFailsWith<IllegalArgumentException> { Sampling(repetitionPenalty = 0.0f) }
        assertFailsWith<IllegalArgumentException> { Sampling(repetitionWindow = -1) }
        assertFailsWith<IllegalArgumentException> { Sampling().newSampler(3).next(FloatArray(2)) }
    }

    @Test
    fun testStreamingSessionAcceptsSampling() {
        val model = seededModel()
        val tokenizer = object : Tokenizer {
            override val vocabularySize: Int = 40
            override fun encode(text: String): IntArray = IntArray(text.length) { i -> text[i].code % 40 }
            override fun decode(tokens: IntArray): String = tokens.joinToString("") { ('a' + it % 26).toString() }
        }
        val session = StreamingSession(model, tokenizer)
        val greedy = session.stream("abc", 6) { }
        val sampled = session.stream("abc", 6, emptySet(), Sampling.GREEDY) { }
        assertEquals(greedy.generatedText, sampled.generatedText)
        val seeded = session.stream("abc", 6, emptySet(), Sampling(seed = 3L)) { }
        assertEquals(6, seeded.tokensGenerated)
    }
}
