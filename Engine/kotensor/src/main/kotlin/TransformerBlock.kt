package io.kotensor.core

class TransformerBlock(
    val dimension: Int,
    val numHeads: Int,
    val hiddenDimension: Int = 4 * dimension,
    val numKeyValueHeads: Int = numHeads,
    val normEps: Float = 1e-6f,
    val ropeTheta: Float = 10000.0f,
    ropeScaling: RoPEScaling? = null,
    quantization: Quantization? = null
) : Module() {
    val norm1 = RMSNorm(dimension, normEps)
    val attention = Attention(dimension, numHeads, numKeyValueHeads, ropeTheta, ropeScaling, quantization)
    val norm2 = RMSNorm(dimension, normEps)
    val ffn = SwiGLU(dimension, hiddenDimension, quantization)

    override fun forward(input: Tensor): Tensor {
        return forward(input, input.data.size / dimension)
    }

    fun forward(input: Tensor, sequenceLength: Int): Tensor {
        val normed1 = norm1(input)
        val attnOut = attention.forward(normed1, sequenceLength)
        val res1 = input + attnOut
        val normed2 = norm2(res1)
        val ffnOut = ffn(normed2)
        return res1 + ffnOut
    }

    fun forward(input: Tensor, cache: KVCache, layer: Int): Tensor {
        val normed1 = norm1(input)
        val attnOut = attention.forward(normed1, cache, layer)
        val res1 = input + attnOut
        val normed2 = norm2(res1)
        val ffnOut = ffn(normed2)
        return res1 + ffnOut
    }

    fun decodeInto(x: Tensor, cache: KVCache, layer: Int, buffers: InferenceBuffers): Tensor {
        norm1.forwardInto(x, buffers.normed)
        attention.decodeInto(buffers.normed, cache, layer, buffers.attentionOut, buffers)
        x.addInPlace(buffers.attentionOut)
        norm2.forwardInto(x, buffers.normed)
        ffn.forwardInto(buffers.normed, buffers.ffnOut, buffers.gate, buffers.up)
        return x.addInPlace(buffers.ffnOut)
    }

    override fun parameters(): List<Tensor> {
        return norm1.parameters() + attention.parameters() + norm2.parameters() + ffn.parameters()
    }
}

