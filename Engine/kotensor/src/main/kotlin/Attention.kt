package io.kotensor.core

import kotlin.math.exp
import kotlin.math.sqrt

class Attention(
    val dimension: Int,
    val numHeads: Int,
    val numKeyValueHeads: Int = numHeads,
    val ropeTheta: Float = 10000.0f,
    ropeScaling: RoPEScaling? = null,
    quantization: Quantization? = null
) : Module() {
    val headDim = dimension / numHeads
    val kvDimension = numKeyValueHeads * headDim
    val groupSize = numHeads / numKeyValueHeads
    val scale = 1.0f / sqrt(headDim.toFloat())

    var wQ: Module = projection(dimension, dimension, quantization)
    var wK: Module = projection(dimension, kvDimension, quantization)
    var wV: Module = projection(dimension, kvDimension, quantization)
    var wOut: Module = projection(dimension, dimension, quantization)
    val rope = RoPE(headDim, ropeTheta, ropeScaling)

    init {
        require(dimension % numHeads == 0) { "dimension must be divisible by numHeads" }
        require(numKeyValueHeads in 1..numHeads && numHeads % numKeyValueHeads == 0) {
            "numHeads $numHeads must be a multiple of numKeyValueHeads $numKeyValueHeads"
        }
    }

    override fun forward(input: Tensor): Tensor {
        return forward(input, input.data.size / dimension)
    }

    fun forward(input: Tensor, sequenceLength: Int): Tensor {
        val totalElements = input.data.size
        require(totalElements % dimension == 0) { "Input elements $totalElements not divisible by dimension $dimension" }
        val tokens = totalElements / dimension
        require(sequenceLength > 0 && tokens % sequenceLength == 0) {
            "Token count $tokens must be a positive multiple of sequenceLength $sequenceLength"
        }

        val qProj = wQ(input)
        val kProj = wK(input)
        val vProj = wV(input)

        val qRoPE = rope.apply(qProj, 0, sequenceLength)
        val kRoPE = rope.apply(kProj, 0, sequenceLength)

        val attnScores = FloatArray(numHeads * tokens * tokens)
        val attnWeights = FloatArray(numHeads * tokens * tokens)
        val contextData = FloatArray(tokens * dimension)

        for (h in 0 until numHeads) {
            val headOffset = h * headDim
            val kvOffset = (h / groupSize) * headDim
            val matrixOffset = h * tokens * tokens

            for (i in 0 until tokens) {
                val rowOffset = matrixOffset + i * tokens
                val seqStart = (i / sequenceLength) * sequenceLength
                var maxScore = -1e9f

                for (j in seqStart..i) {
                    var dot = 0.0f
                    for (d in 0 until headDim) {
                        val qVal = qRoPE.data[i * dimension + headOffset + d]
                        val kVal = kRoPE.data[j * kvDimension + kvOffset + d]
                        dot += qVal * kVal
                    }
                    val score = dot * scale
                    attnScores[rowOffset + j] = score
                    if (score > maxScore) {
                        maxScore = score
                    }
                }

                var sumExp = 0.0f
                for (j in seqStart..i) {
                    val expVal = exp(attnScores[rowOffset + j] - maxScore)
                    attnWeights[rowOffset + j] = expVal
                    sumExp += expVal
                }

                for (j in seqStart..i) {
                    attnWeights[rowOffset + j] /= sumExp
                }

                for (d in 0 until headDim) {
                    var outVal = 0.0f
                    for (j in seqStart..i) {
                        val w = attnWeights[rowOffset + j]
                        val vVal = vProj.data[j * kvDimension + kvOffset + d]
                        outVal += w * vVal
                    }
                    contextData[i * dimension + headOffset + d] = outVal
                }
            }
        }

        lateinit var context: Tensor
        context = Tensor(contextData, intArrayOf(tokens, dimension), listOf(qRoPE, kRoPE, vProj)) {
            val needQ = qRoPE.requiresGrad
            val needK = kRoPE.requiresGrad
            val needScores = needQ || needK
            val contextGrad = context.grad
            val vGrad = if (vProj.requiresGrad) vProj.grad else null
            val qGrad = if (needQ) qRoPE.grad else null
            val kGrad = if (needK) kRoPE.grad else null
            val dAttnWeights = FloatArray(if (needScores) numHeads * tokens * tokens else 0)

            for (h in 0 until numHeads) {
                val headOffset = h * headDim
                val kvOffset = (h / groupSize) * headDim
                val matrixOffset = h * tokens * tokens

                for (i in 0 until tokens) {
                    val rowOffset = matrixOffset + i * tokens
                    val seqStart = (i / sequenceLength) * sequenceLength

                    if (vGrad != null) {
                        for (j in seqStart..i) {
                            val w = attnWeights[rowOffset + j]
                            for (d in 0 until headDim) {
                                vGrad[j * kvDimension + kvOffset + d] += contextGrad[i * dimension + headOffset + d] * w
                            }
                        }
                    }

                    if (!needScores) continue

                    for (j in seqStart..i) {
                        var dWeight = 0.0f
                        for (d in 0 until headDim) {
                            dWeight += contextGrad[i * dimension + headOffset + d] * vProj.data[j * kvDimension + kvOffset + d]
                        }
                        dAttnWeights[rowOffset + j] = dWeight
                    }

                    var sumDWeightWeight = 0.0f
                    for (j in seqStart..i) {
                        sumDWeightWeight += dAttnWeights[rowOffset + j] * attnWeights[rowOffset + j]
                    }

                    for (j in seqStart..i) {
                        val dScore = attnWeights[rowOffset + j] * (dAttnWeights[rowOffset + j] - sumDWeightWeight)
                        val scaledDScore = dScore * scale
                        if (qGrad != null) {
                            for (d in 0 until headDim) {
                                qGrad[i * dimension + headOffset + d] += scaledDScore * kRoPE.data[j * kvDimension + kvOffset + d]
                            }
                        }
                        if (kGrad != null) {
                            for (d in 0 until headDim) {
                                kGrad[j * kvDimension + kvOffset + d] += scaledDScore * qRoPE.data[i * dimension + headOffset + d]
                            }
                        }
                    }
                }
            }
        }

        return wOut(context)
    }

    fun forward(input: Tensor, cache: KVCache, layer: Int): Tensor {
        val totalElements = input.data.size
        require(totalElements % dimension == 0) { "Input elements $totalElements not divisible by dimension $dimension" }
        val tokens = totalElements / dimension
        require(tokens > 0) { "Input cannot be empty" }
        require(layer in 0 until cache.numLayers) { "Layer $layer out of cache bounds ${cache.numLayers}" }
        require(cache.numHeads == numKeyValueHeads && cache.headDim == headDim) {
            "Cache built for ${cache.numHeads} heads × ${cache.headDim} dims, attention has $numKeyValueHeads KV heads × $headDim"
        }
        val startPosition = cache.length
        require(startPosition + tokens <= cache.maxSequenceLength) {
            "KV cache full: $startPosition + $tokens positions exceed maxSequenceLength ${cache.maxSequenceLength}"
        }

        val qProj = wQ(input)
        val kProj = wK(input)
        val vProj = wV(input)

        val qRoPE = rope.apply(qProj, startPosition)
        val kRoPE = rope.apply(kProj, startPosition)

        for (h in 0 until numKeyValueHeads) {
            val headOffset = h * headDim
            for (t in 0 until tokens) {
                val tokenOffset = t * kvDimension + headOffset
                val cacheOffset = cache.offset(layer, h, startPosition + t)
                for (d in 0 until headDim) {
                    cache.keys[cacheOffset + d] = kRoPE.data[tokenOffset + d]
                    cache.values[cacheOffset + d] = vProj.data[tokenOffset + d]
                }
            }
        }

        val contextData = FloatArray(tokens * dimension)
        attendCached(qRoPE.data, tokens, startPosition, cache, layer, contextData, FloatArray(startPosition + tokens))

        val context = Tensor(contextData, intArrayOf(tokens, dimension))
        return wOut(context)
    }

    fun decodeInto(input: Tensor, cache: KVCache, layer: Int, out: Tensor, buffers: InferenceBuffers): Tensor {
        require(input.data.size == dimension) { "decodeInto takes one token of $dimension elements, got ${input.data.size}" }
        require(layer in 0 until cache.numLayers) { "Layer $layer out of cache bounds ${cache.numLayers}" }
        require(cache.numHeads == numKeyValueHeads && cache.headDim == headDim) {
            "Cache built for ${cache.numHeads} heads × ${cache.headDim} dims, attention has $numKeyValueHeads KV heads × $headDim"
        }
        require(buffers.k.data.size == kvDimension && buffers.v.data.size == kvDimension) {
            "InferenceBuffers built for ${buffers.k.data.size} key/value elements, attention has $kvDimension"
        }
        val position = cache.length
        require(position < cache.maxSequenceLength) { "KV cache full: maxSequenceLength ${cache.maxSequenceLength}" }
        require(buffers.scores.size > position) { "InferenceBuffers sized for ${buffers.scores.size} positions, cache is at $position" }

        val q = wQ.forwardInto(input, buffers.q)
        val k = wK.forwardInto(input, buffers.k)
        val v = wV.forwardInto(input, buffers.v)
        rope.applyInPlace(q, position)
        rope.applyInPlace(k, position)

        for (h in 0 until numKeyValueHeads) {
            val headOffset = h * headDim
            val cacheOffset = cache.offset(layer, h, position)
            for (d in 0 until headDim) {
                cache.keys[cacheOffset + d] = k.data[headOffset + d]
                cache.values[cacheOffset + d] = v.data[headOffset + d]
            }
        }

        val context = buffers.context
        attendCached(q.data, 1, position, cache, layer, context.data, buffers.scores)

        return wOut.forwardInto(context, out)
    }

    internal fun prefillInto(input: Tensor, cache: KVCache, layer: Int, buffers: PrefillBuffers, scores: FloatArray): Tensor {
        val tokens = buffers.rows
        require(input.data.size == tokens * dimension) { "prefillInto takes $tokens tokens of $dimension elements, got ${input.data.size}" }
        require(layer in 0 until cache.numLayers) { "Layer $layer out of cache bounds ${cache.numLayers}" }
        require(cache.numHeads == numKeyValueHeads && cache.headDim == headDim) {
            "Cache built for ${cache.numHeads} heads × ${cache.headDim} dims, attention has $numKeyValueHeads KV heads × $headDim"
        }
        val startPosition = cache.length
        require(startPosition + tokens <= cache.maxSequenceLength) {
            "KV cache full: $startPosition + $tokens positions exceed maxSequenceLength ${cache.maxSequenceLength}"
        }
        require(scores.size >= startPosition + tokens) { "Scores sized for ${scores.size} positions, prompt ends at ${startPosition + tokens}" }

        val q = wQ.forwardInto(input, buffers.q)
        val k = wK.forwardInto(input, buffers.k)
        val v = wV.forwardInto(input, buffers.v)
        rope.applyRowsInPlace(q, startPosition)
        rope.applyRowsInPlace(k, startPosition)

        for (h in 0 until numKeyValueHeads) {
            val headOffset = h * headDim
            for (t in 0 until tokens) {
                val tokenOffset = t * kvDimension + headOffset
                val cacheOffset = cache.offset(layer, h, startPosition + t)
                for (d in 0 until headDim) {
                    cache.keys[cacheOffset + d] = k.data[tokenOffset + d]
                    cache.values[cacheOffset + d] = v.data[tokenOffset + d]
                }
            }
        }

        attendCached(q.data, tokens, startPosition, cache, layer, buffers.context.data, scores)

        return wOut.forwardInto(buffers.context, buffers.attentionOut)
    }

    private fun attendCached(q: FloatArray, tokens: Int, startPosition: Int, cache: KVCache, layer: Int, context: FloatArray, scores: FloatArray) {
        val units = numHeads * tokens
        val work = units.toLong() * (startPosition + tokens) * headDim * 2L
        if (MatMul.runsInParallel(units, work)) {
            MatMul.parallelFor(units, work) { unit ->
                val token = unit % tokens
                val position = startPosition + token
                attendHead(q, unit / tokens, token, position, cache, layer, context, threadScores(position + 1))
            }
        } else {
            for (h in 0 until numHeads) {
                for (t in 0 until tokens) {
                    attendHead(q, h, t, startPosition + t, cache, layer, context, scores)
                }
            }
        }
    }

    private fun attendHead(q: FloatArray, head: Int, token: Int, position: Int, cache: KVCache, layer: Int, context: FloatArray, scores: FloatArray) {
        val qOffset = token * dimension + head * headDim
        val base = cache.offset(layer, head / groupSize, 0)
        val keys = cache.keys
        val values = cache.values
        var maxScore = -1e9f

        for (j in 0..position) {
            val score = dot(q, qOffset, keys, base + j * headDim) * scale
            scores[j] = score
            if (score > maxScore) {
                maxScore = score
            }
        }

        var sumExp = 0.0f
        for (j in 0..position) {
            val expVal = exp(scores[j] - maxScore)
            scores[j] = expVal
            sumExp += expVal
        }
        val invSumExp = 1.0f / sumExp

        for (d in 0 until headDim) {
            context[qOffset + d] = 0.0f
        }
        for (j in 0..position) {
            val w = scores[j] * invSumExp
            val vOffset = base + j * headDim
            for (d in 0 until headDim) {
                context[qOffset + d] += w * values[vOffset + d]
            }
        }
    }

    private fun dot(a: FloatArray, aOffset: Int, b: FloatArray, bOffset: Int): Float {
        var s0 = 0.0f
        var s1 = 0.0f
        var s2 = 0.0f
        var s3 = 0.0f
        var d = 0
        while (d + 4 <= headDim) {
            s0 += a[aOffset + d] * b[bOffset + d]
            s1 += a[aOffset + d + 1] * b[bOffset + d + 1]
            s2 += a[aOffset + d + 2] * b[bOffset + d + 2]
            s3 += a[aOffset + d + 3] * b[bOffset + d + 3]
            d += 4
        }
        var sum = (s0 + s1) + (s2 + s3)
        while (d < headDim) {
            sum += a[aOffset + d] * b[bOffset + d]
            d++
        }
        return sum
    }

    override fun parameters(): List<Tensor> {
        return wQ.parameters() + wK.parameters() + wV.parameters() + wOut.parameters()
    }
}

private val attentionScratch = object : ThreadLocal<FloatArray>() {
    override fun initialValue(): FloatArray = FloatArray(0)
}

private fun threadScores(size: Int): FloatArray {
    val current = attentionScratch.get()
    if (current.size >= size) return current
    return FloatArray(maxOf(size, current.size * 2)).also { attentionScratch.set(it) }
}
