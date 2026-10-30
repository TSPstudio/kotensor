package io.kotensor.core

class DataLoader @JvmOverloads constructor(
    val tokens: IntArray,
    val batchSize: Int,
    val sequenceLength: Int,
    val labels: IntArray = tokens
) {
    val totalBatches = (tokens.size - 1) / (batchSize * sequenceLength)
    private var currentIndex = 0

    init {
        require(labels.size == tokens.size) { "labels has ${labels.size} entries, tokens has ${tokens.size}" }
    }

    fun hasNext(): Boolean {
        return currentIndex + (batchSize * sequenceLength) < tokens.size
    }

    fun reset() {
        currentIndex = 0
    }

    fun nextBatch(): Pair<Tensor, Tensor> {
        if (!hasNext()) {
            reset()
        }

        val totalElements = batchSize * sequenceLength
        val inputData = FloatArray(totalElements)
        val targetData = FloatArray(totalElements)

        for (b in 0 until batchSize) {
            for (s in 0 until sequenceLength) {
                val bufferIdx = b * sequenceLength + s
                val streamIdx = currentIndex + bufferIdx
                inputData[bufferIdx] = tokens[streamIdx].toFloat()
                targetData[bufferIdx] = labels[streamIdx + 1].toFloat()
            }
        }

        currentIndex += totalElements

        val x = Tensor(inputData, intArrayOf(batchSize, sequenceLength))
        val y = Tensor(targetData, intArrayOf(batchSize, sequenceLength))
        return Pair(x, y)
    }

    companion object {
        fun promptMasked(examples: List<Pair<IntArray, IntArray>>, batchSize: Int, sequenceLength: Int): DataLoader {
            var total = 0
            for ((prompt, response) in examples) total += prompt.size + response.size
            val tokens = IntArray(total)
            val labels = IntArray(total)
            var position = 0
            for ((prompt, response) in examples) {
                for (token in prompt) {
                    tokens[position] = token
                    labels[position] = IGNORE_INDEX
                    position++
                }
                for (token in response) {
                    tokens[position] = token
                    labels[position] = token
                    position++
                }
            }
            return DataLoader(tokens, batchSize, sequenceLength, labels)
        }
    }
}
