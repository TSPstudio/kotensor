package io.kotensor.core

import kotlin.math.sqrt
import kotlin.random.Random

abstract class Projection(
    val inSize: Int,
    val outSize: Int
) : Module() {
    abstract val bias: Tensor
}

class Linear(
    inSize: Int,
    outSize: Int
): Projection(inSize, outSize) {
    val weight: Tensor
    override val bias: Tensor

    init{
    val scale = sqrt(2.0f/(inSize + outSize))
        val weightData = FloatArray(inSize * outSize){
            (Random.nextFloat() * 2f - 1f) * scale
        }
        weight = Tensor(weightData, intArrayOf(inSize, outSize))
        val biasData = FloatArray(outSize)
        bias = Tensor(biasData, intArrayOf(1, outSize))
    }

    override fun forward(input:Tensor): Tensor{
        return(input matmul weight) + bias
    }

    override fun forwardInto(input: Tensor, out: Tensor): Tensor {
        input.matmulInto(weight, out)
        return out.addInPlace(bias)
    }

    override fun parameters(): List<Tensor> {
        return listOf(weight, bias)
    }

}

