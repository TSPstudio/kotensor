package io.kotensor.core

import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class KVCacheTest {

    private fun seededModel(vocabularySize: Int = 12, numLayers: Int = 2, seed: Int = 7): Transformer {
        val model = Transformer(vocabularySize, dimension = 16, numLayers = numLayers, numHeads = 2)
        val random = Random(seed)
        for (p in model.parameters()) {
            for (i in p.data.indices) {
                p.data[i] = (random.nextFloat() * 2f - 1f) * 0.3f
            }
        }
        return model
    }

    private fun uncachedGreedy(model: Transformer, prompt: IntArray, maxTokens: Int): IntArray {
        var tokens = prompt.copyOf()
        repeat(maxTokens) {
            val logits = model.forward(tokens)
            val offset = (tokens.size - 1) * model.vocabularySize
            var best = 0
            for (v in 1 until model.vocabularySize) {
                if (logits.data[offset + v] > logits.data[offset + best]) best = v
            }
            tokens = tokens + best
        }
        return tokens
    }

    @Test
    fun testCacheLayoutIsLayersHeadsPositionsHeadDim() {
        val cache = KVCache(numLayers = 3, numHeads = 2, maxSequenceLength = 5, headDim = 4)
        assertEquals(3 * 2 * 5 * 4, cache.keys.size)
        assertEquals(cache.keys.size, cache.values.size)
        assertEquals(4, cache.offset(0, 0, 1))
        assertEquals(5 * 4, cache.offset(0, 1, 0))
        assertEquals(2 * 5 * 4, cache.offset(1, 0, 0))
        assertEquals(cache.keys.size - 4, cache.offset(2, 1, 4))
        assertEquals(2L * cache.keys.size * 4L, cache.sizeInBytes)

        assertEquals(0, cache.length)
        assertEquals(5, cache.remaining)
        cache.advance(3)
        assertEquals(3, cache.length)
        assertFailsWith<IllegalArgumentException> { cache.advance(3) }
        cache.reset()
        assertEquals(0, cache.length)
    }

    @Test
    fun testCachedDecodeMatchesFullSequenceForward() {
        val model = seededModel()
        val sequence = intArrayOf(3, 1, 4, 1, 5, 9, 2, 6)
        val full = model.forward(sequence)

        val cache = model.newCache(sequence.size)
        val prefill = model.forward(sequence.sliceArray(0 until 3), cache)
        assertEquals(3, cache.length)
        assertTrue(prefill.shape.contentEquals(intArrayOf(3, model.vocabularySize)))

        val rows = mutableListOf<FloatArray>()
        for (t in 0 until 3) {
            rows.add(prefill.data.sliceArray(t * model.vocabularySize until (t + 1) * model.vocabularySize))
        }
        for (t in 3 until sequence.size) {
            val step = model.forward(intArrayOf(sequence[t]), cache)
            assertTrue(step.shape.contentEquals(intArrayOf(1, model.vocabularySize)))
            rows.add(step.data)
            assertEquals(t + 1, cache.length)
        }

        for (t in sequence.indices) {
            for (v in 0 until model.vocabularySize) {
                assertEquals(full.data[t * model.vocabularySize + v], rows[t][v], 1e-4f, "position $t, vocab $v")
            }
        }
    }

    @Test
    fun testCachedDecodeMatchesFullForwardWithLoRA() {
        val model = seededModel(numLayers = 1)
        val adapter = LoRAAdapter(model, rank = 2)
        val random = Random(11)
        for (p in adapter.parameters()) {
            for (i in p.data.indices) p.data[i] = (random.nextFloat() * 2f - 1f) * 0.3f
        }

        val sequence = intArrayOf(0, 7, 7, 2, 10)
        val full = model.forward(sequence)
        val cache = model.newCache(sequence.size)
        var last: Tensor? = null
        for (token in sequence) {
            last = model.forward(intArrayOf(token), cache)
        }
        val lastOffset = (sequence.size - 1) * model.vocabularySize
        for (v in 0 until model.vocabularySize) {
            assertEquals(full.data[lastOffset + v], last!!.data[v], 1e-4f)
        }
    }

    @Test
    fun testGenerateMatchesUncachedGreedyDecoding() {
        val model = seededModel()
        val prompt = intArrayOf(2, 5, 8)
        val expected = uncachedGreedy(model, prompt, maxTokens = 6)

        assertContentEquals(expected, model.generate(prompt, maxTokens = 6))
        assertContentEquals(expected.sliceArray(prompt.size until expected.size), model.generateStream(prompt, maxTokens = 6).toList().toIntArray())
    }

    @Test
    fun testGenerateStopsAtStopToken() {
        val model = seededModel()
        val prompt = intArrayOf(2, 5, 8)
        val greedy = uncachedGreedy(model, prompt, maxTokens = 4)
        val secondNewToken = greedy[prompt.size + 1]

        val stopped = model.generate(prompt, maxTokens = 4, stopTokens = setOf(secondNewToken))
        assertContentEquals(greedy.sliceArray(0 until prompt.size + 1), stopped)
        assertContentEquals(prompt, model.generate(prompt, maxTokens = 4, stopTokens = setOf(greedy[prompt.size])))
        assertContentEquals(prompt, model.generate(prompt, maxTokens = 0))
    }

    @Test
    fun testCacheRejectsMoreTokensThanItsCapacity() {
        val model = seededModel()
        val cache = model.newCache(3)
        model.forward(intArrayOf(1, 2, 3), cache)
        assertFailsWith<IllegalArgumentException> { model.forward(intArrayOf(4), cache) }
        assertEquals(3, cache.length)

        val wrongShape = KVCache(numLayers = 1, numHeads = 2, maxSequenceLength = 4, headDim = 8)
        assertFailsWith<IllegalArgumentException> { model.forward(intArrayOf(1), wrongShape) }
    }

    @Test
    fun testResetReusesTheCacheForANewSequence() {
        val model = seededModel()
        val cache = model.newCache(8)
        val first = model.forward(intArrayOf(1, 2, 3), cache).data.copyOf()
        model.forward(intArrayOf(4), cache)
        cache.reset()
        val again = model.forward(intArrayOf(1, 2, 3), cache).data
        assertContentEquals(first, again)
    }

    @Test
    fun testStreamingTextUsesCacheAndIncrementalDecoding() {
        val text = "hello world"
        val tokenizer = SimpleTokenizer(text)
        val model = Transformer(tokenizer.vocabularySize, dimension = 8, numLayers = 1, numHeads = 2)

        val pieces = mutableListOf<String>()
        model.generateStream("h", tokenizer, maxTokens = 5) { piece -> pieces.add(piece) }
        assertEquals(5, pieces.size)
        assertEquals(pieces.joinToString(""), model.generate("h", tokenizer, maxTokens = 5).drop(1))

        val session = StreamingSession(model, tokenizer)
        val state = session.stream("h", maxTokens = 5) { }
        assertEquals(5, state.tokensGenerated)
        assertEquals(pieces.joinToString(""), state.generatedText)
    }
}

