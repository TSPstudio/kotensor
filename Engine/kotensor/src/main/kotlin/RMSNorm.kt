package io.kotensor.core

import kotlin.math.sqrt

class RMSNorm(
    val dimension: Int,
    val eps: Float = 1e-6f
) : Module() {
    val weight = Tensor(FloatArray(dimension) { 1.0f }, intArrayOf(1, dimension))

    override fun forward(input: Tensor): Tensor {
        val totalElements = input.data.size
        require(totalElements % dimension == 0) { "Input elements $totalElements not divisible by dimension $dimension" }
        val rows = totalElements / dimension

        val outData = FloatArray(totalElements)
        val invRms = FloatArray(rows)

        for (r in 0 until rows) {
            val rowOffset = r * dimension
            var sumSq = 0.0f
            for (d in 0 until dimension) {
                val x = input.data[rowOffset + d]
                sumSq += x * x
            }
            val meanSq = sumSq / dimension
            val rRms = 1.0f / sqrt(meanSq + eps)
            invRms[r] = rRms

            for (d in 0 until dimension) {
                outData[rowOffset + d] = input.data[rowOffset + d] * rRms * weight.data[d]
            }
        }

        lateinit var out: Tensor
        out = Tensor(outData, input.shape.copyOf(), listOf(input, weight)) {
            val outGrad = out.grad
            val weightGrad = if (weight.requiresGrad) weight.grad else null
            val inputGrad = if (input.requiresGrad) input.grad else null
            for (r in 0 until rows) {
                val rowOffset = r * dimension
                val rRms = invRms[r]

                if (weightGrad != null) {
                    for (d in 0 until dimension) {
                        weightGrad[d] += outGrad[rowOffset + d] * (input.data[rowOffset + d] * rRms)
                    }
                }

                if (inputGrad == null) continue

                var sumDyGammaX = 0.0f
                for (d in 0 until dimension) {
                    sumDyGammaX += outGrad[rowOffset + d] * weight.data[d] * input.data[rowOffset + d]
                }

                val rRmsCubeOverDim = (rRms * rRms * rRms) / dimension
                for (d in 0 until dimension) {
                    val dy = outGrad[rowOffset + d]
                    val gamma = weight.data[d]
                    val x = input.data[rowOffset + d]
                    val term1 = dy * gamma * rRms
                    val term2 = x * rRmsCubeOverDim * sumDyGammaX
                    inputGrad[rowOffset + d] += term1 - term2
                }
            }
        }
        return out
    }

    override fun forwardInto(input: Tensor, out: Tensor): Tensor {
        val totalElements = input.data.size
        require(totalElements % dimension == 0) { "Input elements $totalElements not divisible by dimension $dimension" }
        require(out.data.size == totalElements) { "out has ${out.data.size} elements, input has $totalElements" }
        val rows = totalElements / dimension

        for (r in 0 until rows) {
            val rowOffset = r * dimension
            var sumSq = 0.0f
            for (d in 0 until dimension) {
                val x = input.data[rowOffset + d]
                sumSq += x * x
            }
            val rRms = 1.0f / sqrt(sumSq / dimension + eps)
            for (d in 0 until dimension) {
                out.data[rowOffset + d] = input.data[rowOffset + d] * rRms * weight.data[d]
            }
        }
        return out
    }

    override fun parameters(): List<Tensor> {
        return listOf(weight)
    }
}

