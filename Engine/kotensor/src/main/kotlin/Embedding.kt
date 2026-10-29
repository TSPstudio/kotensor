package io.kotensor.core

import kotlin.random.Random

class Embedding(
    val vocabularySize: Int,
    val embeddingDim: Int
): Module() {
    val weight: Tensor

    init {
        val scale = 0.02f
        val weightData = FloatArray(vocabularySize * embeddingDim) {
            (Random.nextFloat() * 2f - 1f) * scale
        }
        weight = Tensor(weightData, intArrayOf(vocabularySize, embeddingDim))
    }

    fun forward(indices: IntArray): Tensor {
        val count = indices.size
        val outShape = intArrayOf(count, embeddingDim)
        val outData = FloatArray(count * embeddingDim)

        for (t in indices.indices) {
            val token = indices[t]
            require(token in 0 until vocabularySize) { "$token out of vocab bounds $vocabularySize" }
            for (d in 0 until embeddingDim) {
                outData[t * embeddingDim + d] = weight.data[token * embeddingDim + d]
            }
        }

        lateinit var out: Tensor
        out = Tensor(outData, outShape, listOf(weight)) {
            for (t in indices.indices) {
                val token = indices[t]
                for (d in 0 until embeddingDim) {
                    weight.grad[token * embeddingDim + d] += out.grad[t * embeddingDim + d]
                }
            }
        }
        return out
    }

    override fun forward(input: Tensor): Tensor {
        val indices = IntArray(input.data.size) { i -> input.data[i].toInt() }
        return forward(indices)
    }

    override fun parameters(): List<Tensor> {
        return listOf(weight)
    }
}
