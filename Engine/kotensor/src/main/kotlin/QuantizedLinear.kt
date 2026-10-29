package io.kotensor.core

class QuantizedLinear(
    inSize: Int,
    outSize: Int,
    val quantization: Quantization
) : Projection(inSize, outSize) {
    var weight: QuantizedTensor = QuantizedTensor.empty(inSize, outSize, quantization)
        set(value) {
            require(value.rows == inSize && value.columns == outSize) {
                "Quantized weight is ${value.rows} × ${value.columns}, layer is $inSize × $outSize"
            }
            field = value
        }

    override val bias = Tensor(FloatArray(outSize), intArrayOf(1, outSize))

    override fun forward(input: Tensor): Tensor {
        return input.matmulDequant(weight) + bias
    }

    override fun forwardInto(input: Tensor, out: Tensor): Tensor {
        input.matmulDequantInto(weight, out)
        return out.addInPlace(bias)
    }

    override fun parameters(): List<Tensor> {
        return listOf(bias)
    }

    val sizeInBytes: Long
        get() = weight.sizeInBytes + bias.data.size.toLong() * 4L

    companion object {
        fun fromLinear(linear: Linear, quantization: Quantization): QuantizedLinear {
            val quantized = QuantizedLinear(linear.inSize, linear.outSize, quantization)
            quantized.weight = QuantizedTensor.quantize(linear.weight, quantization)
            System.arraycopy(linear.bias.data, 0, quantized.bias.data, 0, linear.outSize)
            return quantized
        }
    }
}

fun projection(inSize: Int, outSize: Int, quantization: Quantization?): Projection {
    return if (quantization == null) Linear(inSize, outSize) else QuantizedLinear(inSize, outSize, quantization)
}

