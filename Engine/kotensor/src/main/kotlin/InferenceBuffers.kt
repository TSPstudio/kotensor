package io.kotensor.core

class InferenceBuffers(
    val dimension: Int,
    val hiddenDimension: Int,
    val vocabularySize: Int,
    val maxSequenceLength: Int,
    val kvDimension: Int = dimension
) {

    val x = Tensor(FloatArray(dimension), intArrayOf(1, dimension))
    val normed = Tensor(FloatArray(dimension), intArrayOf(1, dimension))
    val q = Tensor(FloatArray(dimension), intArrayOf(1, dimension))
    val k = Tensor(FloatArray(kvDimension), intArrayOf(1, kvDimension))
    val v = Tensor(FloatArray(kvDimension), intArrayOf(1, kvDimension))
    val context = Tensor(FloatArray(dimension), intArrayOf(1, dimension))
    val attentionOut = Tensor(FloatArray(dimension), intArrayOf(1, dimension))
    val gate = Tensor(FloatArray(hiddenDimension), intArrayOf(1, hiddenDimension))
    val up = Tensor(FloatArray(hiddenDimension), intArrayOf(1, hiddenDimension))
    val ffnOut = Tensor(FloatArray(dimension), intArrayOf(1, dimension))

    val logits = Tensor(FloatArray(vocabularySize), intArrayOf(1, vocabularySize))

    val scores = FloatArray(maxSequenceLength)

    init {
        require(dimension > 0 && hiddenDimension > 0 && vocabularySize > 0 && maxSequenceLength > 0 && kvDimension > 0) {
            "InferenceBuffers dimensions must be positive"
        }
    }

    val sizeInBytes: Long
        get() = (6L * dimension + 2L * kvDimension + 2L * hiddenDimension + vocabularySize + maxSequenceLength) * 4L
}

