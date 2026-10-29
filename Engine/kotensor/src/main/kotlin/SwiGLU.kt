package io.kotensor.core

class SwiGLU(
    val dimension: Int,
    val hiddenDimension: Int,
    quantization: Quantization? = null
) : Module() {
    var wGate: Module = projection(dimension, hiddenDimension, quantization)
    var wUp: Module = projection(dimension, hiddenDimension, quantization)
    var wDown: Module = projection(hiddenDimension, dimension, quantization)

    override fun forward(input: Tensor): Tensor {
        val gate = wGate(input).silu()
        val up = wUp(input)
        val fused = gate * up
        return wDown(fused)
    }

    fun forwardInto(input: Tensor, out: Tensor, gate: Tensor, up: Tensor): Tensor {
        wGate.forwardInto(input, gate)
        gate.siluInPlace()
        wUp.forwardInto(input, up)
        gate.mulInPlace(up)
        return wDown.forwardInto(gate, out)
    }

    override fun parameters(): List<Tensor> {
        return wGate.parameters() + wUp.parameters() + wDown.parameters()
    }
}

