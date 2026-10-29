package io.kotensor.core

class DataLoader(
    val tokens: IntArray,
    val batchSize: Int,
    val sequenceLength: Int
) {
    val totalBatches = (tokens.size - 1) / (batchSize * sequenceLength)
    private var currentIndex = 0

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
                targetData[bufferIdx] = tokens[streamIdx + 1].toFloat()
            }
        }

        currentIndex += totalElements

        val x = Tensor(inputData, intArrayOf(batchSize, sequenceLength))
        val y = Tensor(targetData, intArrayOf(batchSize, sequenceLength))
        return Pair(x, y)
    }
}

