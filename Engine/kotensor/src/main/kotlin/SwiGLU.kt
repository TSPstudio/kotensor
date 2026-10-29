package io.kotensor.core

class SwiGLU(
    val dimension: Int,
    val hiddenDimension: Int
) : Module() {
    val wGate = Linear(dimension, hiddenDimension)
    val wUp = Linear(dimension, hiddenDimension)
    val wDown = Linear(hiddenDimension, dimension)

    override fun forward(input: Tensor): Tensor {
        val gate = wGate(input).silu()
        val up = wUp(input)
        val fused = gate * up
        return wDown(fused)
    }

    override fun parameters(): List<Tensor> {
        return wGate.parameters() + wUp.parameters() + wDown.parameters()
    }
}
