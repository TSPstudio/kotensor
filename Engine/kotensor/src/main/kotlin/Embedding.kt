package io.kotensor.core

import kotlin.random.Random

abstract class TokenEmbedding(
    val vocabularySize: Int,
    val embeddingDim: Int
) : Module() {
    abstract fun forward(indices: IntArray): Tensor
    abstract fun lookupInto(token: Int, out: Tensor): Tensor

    override fun forward(input: Tensor): Tensor {
        val indices = IntArray(input.data.size) { i -> input.data[i].toInt() }
        return forward(indices)
    }
}

class Embedding(
    vocabularySize: Int,
    embeddingDim: Int
): TokenEmbedding(vocabularySize, embeddingDim) {
    val weight: Tensor

    init {
        val scale = 0.02f
        val weightData = FloatArray(vocabularySize * embeddingDim) {
            (Random.nextFloat() * 2f - 1f) * scale
        }
        weight = Tensor(weightData, intArrayOf(vocabularySize, embeddingDim))
    }

    override fun forward(indices: IntArray): Tensor {
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
            if (weight.requiresGrad) {
                val weightGrad = weight.grad
                val outGrad = out.grad
                for (t in indices.indices) {
                    val token = indices[t]
                    for (d in 0 until embeddingDim) {
                        weightGrad[token * embeddingDim + d] += outGrad[t * embeddingDim + d]
                    }
                }
            }
        }
        return out
    }

    override fun lookupInto(token: Int, out: Tensor): Tensor {
        require(token in 0 until vocabularySize) { "$token out of vocab bounds $vocabularySize" }
        require(out.data.size == embeddingDim) { "out has ${out.data.size} elements, embedding has $embeddingDim" }
        System.arraycopy(weight.data, token * embeddingDim, out.data, 0, embeddingDim)
        return out
    }

    override fun parameters(): List<Tensor> {
        return listOf(weight)
    }
}

class QuantizedEmbedding(
    vocabularySize: Int,
    embeddingDim: Int,
    val quantization: Quantization
) : TokenEmbedding(vocabularySize, embeddingDim) {
    var weight: QuantizedTensor = QuantizedTensor.empty(vocabularySize, embeddingDim, quantization)
        set(value) {
            require(value.rows == vocabularySize && value.columns == embeddingDim) {
                "Quantized embedding is ${value.rows} × ${value.columns}, table is $vocabularySize × $embeddingDim"
            }
            field = value
        }

    override fun forward(indices: IntArray): Tensor {
        val outData = FloatArray(indices.size * embeddingDim)
        for (t in indices.indices) {
            val token = indices[t]
            require(token in 0 until vocabularySize) { "$token out of vocab bounds $vocabularySize" }
            weight.dequantizeRow(token, outData, t * embeddingDim)
        }
        return Tensor(outData, intArrayOf(indices.size, embeddingDim)).also { it.requiresGrad = false }
    }

    override fun lookupInto(token: Int, out: Tensor): Tensor {
        require(token in 0 until vocabularySize) { "$token out of vocab bounds $vocabularySize" }
        require(out.data.size == embeddingDim) { "out has ${out.data.size} elements, embedding has $embeddingDim" }
        weight.dequantizeRow(token, out.data)
        return out
    }

    val sizeInBytes: Long
        get() = weight.sizeInBytes

    companion object {
        fun fromEmbedding(embedding: Embedding, quantization: Quantization): QuantizedEmbedding {
            val quantized = QuantizedEmbedding(embedding.vocabularySize, embedding.embeddingDim, quantization)
            quantized.weight = QuantizedTensor.quantize(embedding.weight, quantization)
            return quantized
        }
    }
}

