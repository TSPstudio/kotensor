package io.kotensor.core

import kotlin.math.exp
import kotlin.math.sqrt

class Attention(
    val dimension: Int,
    val numHeads: Int
) : Module() {
    val headDim = dimension / numHeads
    val scale = 1.0f / sqrt(headDim.toFloat())

    var wQ: Module = Linear(dimension, dimension)
    var wK: Module = Linear(dimension, dimension)
    var wV: Module = Linear(dimension, dimension)
    var wOut: Module = Linear(dimension, dimension)
    val rope = RoPE(headDim)

    init {
        require(dimension % numHeads == 0) { "dimension must be divisible by numHeads" }
    }

    override fun forward(input: Tensor): Tensor {
        val totalElements = input.data.size
        require(totalElements % dimension == 0)
        val tokens = totalElements / dimension

        val qProj = wQ(input)
        val kProj = wK(input)
        val vProj = wV(input)

        val qRoPE = rope.apply(qProj)
        val kRoPE = rope.apply(kProj)

        val attnScores = FloatArray(numHeads * tokens * tokens)
        val attnWeights = FloatArray(numHeads * tokens * tokens)
        val contextData = FloatArray(tokens * dimension)

        for (h in 0 until numHeads) {
            val headOffset = h * headDim
            val matrixOffset = h * tokens * tokens

            for (i in 0 until tokens) {
                val rowOffset = matrixOffset + i * tokens
                var maxScore = -1e9f

                for (j in 0..i) {
                    var dot = 0.0f
                    for (d in 0 until headDim) {
                        val qVal = qRoPE.data[i * dimension + headOffset + d]
                        val kVal = kRoPE.data[j * dimension + headOffset + d]
                        dot += qVal * kVal
                    }
                    val score = dot * scale
                    attnScores[rowOffset + j] = score
                    if (score > maxScore) {
                        maxScore = score
                    }
                }

                var sumExp = 0.0f
                for (j in 0..i) {
                    val expVal = exp(attnScores[rowOffset + j] - maxScore)
                    attnWeights[rowOffset + j] = expVal
                    sumExp += expVal
                }

                for (j in 0..i) {
                    attnWeights[rowOffset + j] /= sumExp
                }

                for (d in 0 until headDim) {
                    var outVal = 0.0f
                    for (j in 0..i) {
                        val w = attnWeights[rowOffset + j]
                        val vVal = vProj.data[j * dimension + headOffset + d]
                        outVal += w * vVal
                    }
                    contextData[i * dimension + headOffset + d] = outVal
                }
            }
        }

        lateinit var context: Tensor
        context = Tensor(contextData, intArrayOf(tokens, dimension), listOf(qRoPE, kRoPE, vProj)) {
            val dAttnWeights = FloatArray(numHeads * tokens * tokens)
            val dScores = FloatArray(numHeads * tokens * tokens)

            for (h in 0 until numHeads) {
                val headOffset = h * headDim
                val matrixOffset = h * tokens * tokens

                for (i in 0 until tokens) {
                    val rowOffset = matrixOffset + i * tokens

                    for (j in 0..i) {
                        var dWeight = 0.0f
                        for (d in 0 until headDim) {
                            val dOut = context.grad[i * dimension + headOffset + d]
                            val vVal = vProj.data[j * dimension + headOffset + d]
                            dWeight += dOut * vVal
                            vProj.grad[j * dimension + headOffset + d] += dOut * attnWeights[rowOffset + j]
                        }
                        dAttnWeights[rowOffset + j] = dWeight
                    }

                    var sumDWeightWeight = 0.0f
                    for (j in 0..i) {
                        sumDWeightWeight += dAttnWeights[rowOffset + j] * attnWeights[rowOffset + j]
                    }

                    for (j in 0..i) {
                        val dScore = attnWeights[rowOffset + j] * (dAttnWeights[rowOffset + j] - sumDWeightWeight)
                        dScores[rowOffset + j] = dScore

                        val scaledDScore = dScore * scale
                        for (d in 0 until headDim) {
                            val qVal = qRoPE.data[i * dimension + headOffset + d]
                            val kVal = kRoPE.data[j * dimension + headOffset + d]
                            qRoPE.grad[i * dimension + headOffset + d] += scaledDScore * kVal
                            kRoPE.grad[j * dimension + headOffset + d] += scaledDScore * qVal
                        }
                    }
                }
            }
        }

        return wOut(context)
    }

    override fun parameters(): List<Tensor> {
        return wQ.parameters() + wK.parameters() + wV.parameters() + wOut.parameters()
    }
}
