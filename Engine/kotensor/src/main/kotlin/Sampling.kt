package io.kotensor.core

import kotlin.math.exp
import kotlin.random.Random

class Sampling @JvmOverloads constructor(
    val temperature: Float = 0.7f,
    val topK: Int = 40,
    val topP: Float = 0.9f,
    val repetitionPenalty: Float = 1.1f,
    val repetitionWindow: Int = 64,
    val seed: Long? = null
) {
    init {
        require(temperature >= 0.0f && !temperature.isNaN()) { "temperature must be at least 0, got $temperature" }
        require(topK >= 0) { "topK must be at least 0, got $topK" }
        require(topP > 0.0f && topP <= 1.0f) { "topP must be in (0, 1], got $topP" }
        require(repetitionPenalty > 0.0f && !repetitionPenalty.isNaN()) { "repetitionPenalty must be positive, got $repetitionPenalty" }
        require(repetitionWindow >= 0) { "repetitionWindow must be at least 0, got $repetitionWindow" }
    }

    val isGreedy: Boolean
        get() = temperature == 0.0f || topK == 1

    fun newSampler(vocabularySize: Int): Sampler = Sampler(this, vocabularySize, seed?.let { Random(it) } ?: Random.Default)

    override fun toString(): String =
        "Sampling(temperature=$temperature, topK=$topK, topP=$topP, repetitionPenalty=$repetitionPenalty, repetitionWindow=$repetitionWindow, seed=$seed)"

    companion object {
        @JvmField
        val GREEDY = Sampling(temperature = 0.0f, topK = 0, topP = 1.0f, repetitionPenalty = 1.0f, repetitionWindow = 0)
    }
}

class Sampler internal constructor(
    val sampling: Sampling,
    val vocabularySize: Int,
    private val random: Random
) {
    private val penalized = BooleanArray(vocabularySize)
    private var candidates = IntArray(0)
    private var weights = FloatArray(0)

    init {
        require(vocabularySize > 0) { "vocabularySize must be positive" }
    }

    fun next(logits: FloatArray, history: IntArray = IntArray(0), historyLength: Int = history.size): Int {
        require(logits.size >= vocabularySize) { "logits has ${logits.size} entries, vocabulary has $vocabularySize" }
        require(historyLength in 0..history.size) { "historyLength $historyLength out of range for ${history.size} tokens" }
        applyRepetitionPenalty(logits, history, historyLength)
        if (sampling.isGreedy) return argmax(logits)
        val count = selectCandidates(logits)
        val inverseTemperature = 1.0f / sampling.temperature
        val maxLogit = logits[candidates[0]]
        var total = 0.0f
        for (i in 0 until count) {
            val w = exp((logits[candidates[i]] - maxLogit) * inverseTemperature)
            weights[i] = w
            total += w
        }
        var kept = count
        if (sampling.topP < 1.0f) {
            val limit = sampling.topP * total
            var cumulative = 0.0f
            for (i in 0 until count) {
                cumulative += weights[i]
                if (cumulative >= limit) {
                    kept = i + 1
                    break
                }
            }
            total = 0.0f
            for (i in 0 until kept) total += weights[i]
        }
        var target = random.nextFloat() * total
        for (i in 0 until kept) {
            target -= weights[i]
            if (target < 0.0f) return candidates[i]
        }
        return candidates[kept - 1]
    }

    private fun applyRepetitionPenalty(logits: FloatArray, history: IntArray, historyLength: Int) {
        val penalty = sampling.repetitionPenalty
        if (penalty == 1.0f || sampling.repetitionWindow == 0 || historyLength == 0) return
        val start = maxOf(0, historyLength - sampling.repetitionWindow)
        for (i in start until historyLength) {
            val token = history[i]
            if (token !in 0 until vocabularySize || penalized[token]) continue
            penalized[token] = true
            val logit = logits[token]
            logits[token] = if (logit > 0.0f) logit / penalty else logit * penalty
        }
        for (i in start until historyLength) {
            val token = history[i]
            if (token in 0 until vocabularySize) penalized[token] = false
        }
    }

    private fun argmax(logits: FloatArray): Int {
        var bestIndex = 0
        var maxLogit = -1e9f
        for (v in 0 until vocabularySize) {
            val logit = logits[v]
            if (logit > maxLogit) {
                maxLogit = logit
                bestIndex = v
            }
        }
        return bestIndex
    }

    private fun selectCandidates(logits: FloatArray): Int {
        val k = if (sampling.topK == 0 || sampling.topK > vocabularySize) vocabularySize else sampling.topK
        if (candidates.size < k) {
            candidates = IntArray(k)
            weights = FloatArray(k)
        }
        val heap = candidates
        var size = 0
        for (v in 0 until vocabularySize) {
            val logit = logits[v]
            if (size < k) {
                heap[size] = v
                var child = size
                size++
                while (child > 0) {
                    val parent = (child - 1) / 2
                    if (!less(logits, heap[child], heap[parent])) break
                    val swap = heap[child]
                    heap[child] = heap[parent]
                    heap[parent] = swap
                    child = parent
                }
            } else if (logit > logits[heap[0]] || (logit == logits[heap[0]] && v < heap[0])) {
                heap[0] = v
                siftDown(logits, heap, 0, size)
            }
        }
        var end = size
        while (end > 1) {
            end--
            val swap = heap[0]
            heap[0] = heap[end]
            heap[end] = swap
            siftDown(logits, heap, 0, end)
        }
        return size
    }

    private fun less(logits: FloatArray, a: Int, b: Int): Boolean {
        val la = logits[a]
        val lb = logits[b]
        return la < lb || (la == lb && a > b)
    }

    private fun siftDown(logits: FloatArray, heap: IntArray, start: Int, size: Int) {
        var parent = start
        while (true) {
            val left = 2 * parent + 1
            if (left >= size) return
            val right = left + 1
            val smallest = if (right < size && less(logits, heap[right], heap[left])) right else left
            if (!less(logits, heap[smallest], heap[parent])) return
            val swap = heap[parent]
            heap[parent] = heap[smallest]
            heap[smallest] = swap
            parent = smallest
        }
    }
}
