package io.kotensor.core

class KVCache(
    val numLayers: Int,
    val numHeads: Int,
    val maxSequenceLength: Int,
    val headDim: Int
) {
    val keys = FloatArray(numLayers * numHeads * maxSequenceLength * headDim)
    val values = FloatArray(numLayers * numHeads * maxSequenceLength * headDim)

    var length: Int = 0
        private set

    init {
        require(numLayers > 0) { "numLayers must be positive" }
        require(numHeads > 0) { "numHeads must be positive" }
        require(maxSequenceLength > 0) { "maxSequenceLength must be positive" }
        require(headDim > 0) { "headDim must be positive" }
    }

    val remaining: Int
        get() = maxSequenceLength - length

    fun offset(layer: Int, head: Int, position: Int): Int {
        return ((layer * numHeads + head) * maxSequenceLength + position) * headDim
    }

    fun advance(count: Int) {
        require(count >= 0) { "count must not be negative" }
        require(length + count <= maxSequenceLength) {
            "KV cache full: $length + $count positions exceed maxSequenceLength $maxSequenceLength"
        }
        length += count
    }

    fun reset() {
        length = 0
    }

    val sizeInBytes: Long
        get() = (keys.size.toLong() + values.size.toLong()) * 4L
}

