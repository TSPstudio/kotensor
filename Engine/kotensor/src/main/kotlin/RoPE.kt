package io.kotensor.core

import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.pow

class RoPEScaling(
    val factor: Float,
    val lowFrequencyFactor: Float = 1.0f,
    val highFrequencyFactor: Float = 4.0f,
    val originalMaxPositionEmbeddings: Int = 8192
) {
    init {
        require(factor > 0f && originalMaxPositionEmbeddings > 0) { "Invalid RoPE scaling" }
        require(highFrequencyFactor > lowFrequencyFactor) { "highFrequencyFactor must exceed lowFrequencyFactor" }
    }

    fun scale(invFreq: Float): Float {
        val lowFrequencyWavelength = originalMaxPositionEmbeddings / lowFrequencyFactor
        val highFrequencyWavelength = originalMaxPositionEmbeddings / highFrequencyFactor
        val wavelength = (2.0 * Math.PI / invFreq).toFloat()
        return when {
            wavelength < highFrequencyWavelength -> invFreq
            wavelength > lowFrequencyWavelength -> invFreq / factor
            else -> {
                val smooth = (originalMaxPositionEmbeddings / wavelength - lowFrequencyFactor) / (highFrequencyFactor - lowFrequencyFactor)
                (1.0f - smooth) * invFreq / factor + smooth * invFreq
            }
        }
    }
}

class RoPE(
    val dimension: Int,
    val base: Float = 10000.0f,
    val scaling: RoPEScaling? = null
) : Module() {
    init {
        require(dimension % 2 == 0) { "RoPE dimension must be even" }
    }

    private val halfDim = dimension / 2
    val invFreq = FloatArray(halfDim) { i ->
        val raw = 1.0f / base.pow((2 * i).toFloat() / dimension)
        scaling?.scale(raw) ?: raw
    }

    fun apply(input: Tensor, startPosition: Int = 0, sequenceLength: Int? = null): Tensor {
        val totalElements = input.data.size
        require(totalElements % dimension == 0) { "Input elements not divisible by dimension" }
        val rowWidth = if (input.shape.size == 2 && input.shape[1] % dimension == 0) input.shape[1] else dimension
        val headsPerToken = rowWidth / dimension
        val tokens = totalElements / rowWidth
        val seqLen = sequenceLength ?: tokens
        require(seqLen > 0 && tokens % seqLen == 0) { "Token count $tokens must be a positive multiple of sequenceLength $seqLen" }

        val outData = FloatArray(totalElements)
        val cosValues = FloatArray(tokens * halfDim)
        val sinValues = FloatArray(tokens * halfDim)

        for (t in 0 until tokens) {
            val position = startPosition + (t % seqLen)
            val cacheOffset = t * halfDim

            for (i in 0 until halfDim) {
                val theta = position * invFreq[i]
                cosValues[cacheOffset + i] = cos(theta)
                sinValues[cacheOffset + i] = sin(theta)
            }

            for (h in 0 until headsPerToken) {
                val headOffset = t * rowWidth + h * dimension
                for (i in 0 until halfDim) {
                    val c = cosValues[cacheOffset + i]
                    val s = sinValues[cacheOffset + i]

                    val idx1 = headOffset + i
                    val idx2 = headOffset + i + halfDim
                    val x1 = input.data[idx1]
                    val x2 = input.data[idx2]

                    outData[idx1] = x1 * c - x2 * s
                    outData[idx2] = x1 * s + x2 * c
                }
            }
        }

        lateinit var out: Tensor
        out = Tensor(outData, input.shape.copyOf(), listOf(input)) {
            for (t in 0 until tokens) {
                val cacheOffset = t * halfDim

                for (h in 0 until headsPerToken) {
                    val headOffset = t * rowWidth + h * dimension
                    for (i in 0 until halfDim) {
                        val c = cosValues[cacheOffset + i]
                        val s = sinValues[cacheOffset + i]

                        val idx1 = headOffset + i
                        val idx2 = headOffset + i + halfDim
                        val dy1 = out.grad[idx1]
                        val dy2 = out.grad[idx2]

                        input.grad[idx1] += dy1 * c + dy2 * s
                        input.grad[idx2] += -dy1 * s + dy2 * c
                    }
                }
            }
        }
        return out
    }

    fun applyInPlace(token: Tensor, position: Int): Tensor {
        val width = token.data.size
        require(width % dimension == 0) { "Token elements $width not divisible by dimension $dimension" }
        val headsPerToken = width / dimension
        val data = token.data

        for (i in 0 until halfDim) {
            val theta = position * invFreq[i]
            val c = cos(theta)
            val s = sin(theta)
            for (h in 0 until headsPerToken) {
                val idx1 = h * dimension + i
                val idx2 = idx1 + halfDim
                val x1 = data[idx1]
                val x2 = data[idx2]
                data[idx1] = x1 * c - x2 * s
                data[idx2] = x1 * s + x2 * c
            }
        }
        return token
    }

    override fun forward(input: Tensor): Tensor {
        return apply(input, 0)
    }

    override fun parameters(): List<Tensor> {
        return emptyList()
    }
}

