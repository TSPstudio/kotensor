package io.kotensor.core

import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.pow

class RoPE(
    val dimension: Int,
    val base: Float = 10000.0f
) : Module() {
    init {
        require(dimension % 2 == 0) { "RoPE dimension must be even" }
    }

    fun apply(input: Tensor, startPosition: Int = 0): Tensor {
        val totalElements = input.data.size
        require(totalElements % dimension == 0) { "Input elements not divisible by dimension" }
        val tokens = totalElements / dimension
        val halfDim = dimension / 2

        val outData = FloatArray(totalElements)
        val cosValues = FloatArray(tokens * halfDim)
        val sinValues = FloatArray(tokens * halfDim)

        for (t in 0 until tokens) {
            val position = startPosition + t
            val tokenOffset = t * dimension
            val cacheOffset = t * halfDim

            for (i in 0 until halfDim) {
                val freq = 1.0f / base.pow((2 * i).toFloat() / dimension)
                val theta = position * freq
                val c = cos(theta)
                val s = sin(theta)

                cosValues[cacheOffset + i] = c
                sinValues[cacheOffset + i] = s

                val idx1 = tokenOffset + i
                val idx2 = tokenOffset + i + halfDim
                val x1 = input.data[idx1]
                val x2 = input.data[idx2]

                outData[idx1] = x1 * c - x2 * s
                outData[idx2] = x1 * s + x2 * c
            }
        }

        lateinit var out: Tensor
        out = Tensor(outData, input.shape.copyOf(), listOf(input)) {
            for (t in 0 until tokens) {
                val tokenOffset = t * dimension
                val cacheOffset = t * halfDim

                for (i in 0 until halfDim) {
                    val c = cosValues[cacheOffset + i]
                    val s = sinValues[cacheOffset + i]

                    val idx1 = tokenOffset + i
                    val idx2 = tokenOffset + i + halfDim
                    val dy1 = out.grad[idx1]
                    val dy2 = out.grad[idx2]

                    input.grad[idx1] += dy1 * c + dy2 * s
                    input.grad[idx2] += -dy1 * s + dy2 * c
                }
            }
        }
        return out
    }

    override fun forward(input: Tensor): Tensor {
        return apply(input, 0)
    }

    override fun parameters(): List<Tensor> {
        return emptyList()
    }
}
