package io.kotensor.core

class TransformerBlock(
    val dimension: Int,
    val numHeads: Int,
    val hiddenDimension: Int = 4 * dimension
) : Module() {
    val norm1 = RMSNorm(dimension)
    val attention = Attention(dimension, numHeads)
    val norm2 = RMSNorm(dimension)
    val ffn = SwiGLU(dimension, hiddenDimension)

    /** Treats the whole input (tokens × dimension) as a single sequence. */
    override fun forward(input: Tensor): Tensor {
        return forward(input, input.data.size / dimension)
    }

    /** Processes (batch × sequenceLength) tokens as independent sequences of [sequenceLength]. */
    fun forward(input: Tensor, sequenceLength: Int): Tensor {
        val normed1 = norm1(input)
        val attnOut = attention.forward(normed1, sequenceLength)
        val res1 = input + attnOut
        val normed2 = norm2(res1)
        val ffnOut = ffn(normed2)
        return res1 + ffnOut
    }

    override fun parameters(): List<Tensor> {
        return norm1.parameters() + attention.parameters() + norm2.parameters() + ffn.parameters()
    }
}
