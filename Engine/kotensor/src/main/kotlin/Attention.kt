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
            val dAttnWeights = FloatArray(numHeads * tokens * tokens)

            for (h in 0 until numHeads) {
                val headOffset = h * headDim
                val kvOffset = (h / groupSize) * headDim
                val matrixOffset = h * tokens * tokens

                for (i in 0 until tokens) {
                    val rowOffset = matrixOffset + i * tokens
                    val seqStart = (i / sequenceLength) * sequenceLength

                    for (j in seqStart..i) {
                        var dWeight = 0.0f
                        for (d in 0 until headDim) {
                            val dOut = context.grad[i * dimension + headOffset + d]
                            val vVal = vProj.data[j * kvDimension + kvOffset + d]
                            dWeight += dOut * vVal
                            vProj.grad[j * kvDimension + kvOffset + d] += dOut * attnWeights[rowOffset + j]
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
                        for (d in 0 until headDim) {
                            val qVal = qRoPE.data[i * dimension + headOffset + d]
                            val kVal = kRoPE.data[j * kvDimension + kvOffset + d]
                            qRoPE.grad[i * dimension + headOffset + d] += scaledDScore * kVal
                            kRoPE.grad[j * kvDimension + kvOffset + d] += scaledDScore * qVal
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
        val attnWeights = FloatArray(startPosition + tokens)

        for (h in 0 until numHeads) {
            val headOffset = h * headDim
            val kvHead = h / groupSize

            for (t in 0 until tokens) {
                val position = startPosition + t
                val qOffset = t * dimension + headOffset
                var maxScore = -1e9f

                for (j in 0..position) {
                    val kOffset = cache.offset(layer, kvHead, j)
                    var dot = 0.0f
                    for (d in 0 until headDim) {
                        dot += qRoPE.data[qOffset + d] * cache.keys[kOffset + d]
                    }
                    val score = dot * scale
                    attnWeights[j] = score
                    if (score > maxScore) {
                        maxScore = score
                    }
                }

                var sumExp = 0.0f
                for (j in 0..position) {
                    val expVal = exp(attnWeights[j] - maxScore)
                    attnWeights[j] = expVal
                    sumExp += expVal
                }
                val invSumExp = 1.0f / sumExp

                for (j in 0..position) {
                    val w = attnWeights[j] * invSumExp
                    val vOffset = cache.offset(layer, kvHead, j)
                    for (d in 0 until headDim) {
                        contextData[qOffset + d] += w * cache.values[vOffset + d]
                    }
                }
            }
        }

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
        val scores = buffers.scores
        for (h in 0 until numHeads) {
            val headOffset = h * headDim
            val kvHead = h / groupSize
            var maxScore = -1e9f

            for (j in 0..position) {
                val kOffset = cache.offset(layer, kvHead, j)
                var dot = 0.0f
                for (d in 0 until headDim) {
                    dot += q.data[headOffset + d] * cache.keys[kOffset + d]
                }
                val score = dot * scale
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
                context.data[headOffset + d] = 0.0f
            }
            for (j in 0..position) {
                val w = scores[j] * invSumExp
                val vOffset = cache.offset(layer, kvHead, j)
                for (d in 0 until headDim) {
                    context.data[headOffset + d] += w * cache.values[vOffset + d]
                }
            }
        }

        return wOut.forwardInto(context, out)
    }

    override fun parameters(): List<Tensor> {
        return wQ.parameters() + wK.parameters() + wV.parameters() + wOut.parameters()
    }
}

