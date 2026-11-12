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

    fun copyFrom(source: KVCache, length: Int) {
        require(source.numLayers == numLayers && source.numHeads == numHeads && source.headDim == headDim) {
            "Cannot copy a cache of ${source.numLayers} layers × ${source.numHeads} heads × ${source.headDim} dims into $numLayers × $numHeads × $headDim"
        }
        require(length in 0..source.length) { "Cannot copy $length positions from a cache holding ${source.length}" }
        require(length <= maxSequenceLength) { "Cannot copy $length positions into a cache of $maxSequenceLength" }
        val count = length * headDim
        for (layer in 0 until numLayers) {
            for (head in 0 until numHeads) {
                System.arraycopy(source.keys, source.offset(layer, head, 0), keys, offset(layer, head, 0), count)
                System.arraycopy(source.values, source.offset(layer, head, 0), values, offset(layer, head, 0), count)
            }
        }
        this.length = length
    }

    val sizeInBytes: Long
        get() = (keys.size.toLong() + values.size.toLong()) * 4L
}

class PromptPrefix internal constructor(
    internal val owner: Transformer,
    tokens: IntArray,
    internal val cache: KVCache
) {
    private val tokens = tokens.copyOf()

    val length: Int
        get() = tokens.size

    fun matchLength(prompt: IntArray): Int {
        val limit = minOf(tokens.size, prompt.size - 1)
        var matched = 0
        while (matched < limit && tokens[matched] == prompt[matched]) matched++
        return matched
    }

    val sizeInBytes: Long
        get() = cache.sizeInBytes
}
