package io.kotensor.core

import java.io.File
import kotlin.math.sqrt
import kotlin.random.Random

class LoRALinear(
    val baseLinear: Linear,
    val rank: Int = 4,
    val alpha: Float = 1.0f
) : Module() {
    val loraA: Tensor
    val loraB: Tensor
    val scaling = alpha / rank.toFloat()

    init {
        val inFeatures = baseLinear.inSize
        val outFeatures = baseLinear.outSize

        val scaleA = 1.0f / sqrt(inFeatures.toFloat())
        val dataA = FloatArray(inFeatures * rank) {
            (Random.nextFloat() * 2f - 1f) * scaleA
        }
        loraA = Tensor(dataA, intArrayOf(inFeatures, rank))

        val dataB = FloatArray(rank * outFeatures)
        loraB = Tensor(dataB, intArrayOf(rank, outFeatures))
    }

    override fun forward(input: Tensor): Tensor {
        val baseOutput = baseLinear(input)
        val loraPath = (input matmul loraA) matmul loraB
        val scaledLora = loraPath * scaling
        return baseOutput + scaledLora
    }

    override fun parameters(): List<Tensor> {
        return listOf(loraA, loraB)
    }

    fun deltaWeights(): Tensor {
        val delta = loraA matmul loraB
        return delta * scaling
    }
}

class LoRAAdapter(
    val model: Transformer,
    val rank: Int = 4,
    val alpha: Float = 1.0f
) {
    val layers = mutableMapOf<String, LoRALinear>()

    init {
        for (l in model.blocks.indices) {
            val block = model.blocks[l]
            val qLinear = block.attention.wQ as? Linear
            if (qLinear != null) {
                val qLora = LoRALinear(qLinear, rank, alpha)
                block.attention.wQ = qLora
                layers["block.$l.attention.wQ"] = qLora
            }
            val vLinear = block.attention.wV as? Linear
            if (vLinear != null) {
                val vLora = LoRALinear(vLinear, rank, alpha)
                block.attention.wV = vLora
                layers["block.$l.attention.wV"] = vLora
            }
        }
    }

    fun parameters(): List<Tensor> {
        val params = mutableListOf<Tensor>()
        for (layer in layers.values) {
            params.addAll(layer.parameters())
        }
        return params
    }

    fun merge() {
        for (layer in layers.values) {
            val delta = layer.deltaWeights()
            for (i in layer.baseLinear.weight.data.indices) {
                layer.baseLinear.weight.data[i] += delta.data[i]
            }
        }
    }

    fun saveDelta(file: File) {
        val deltaTensors = mutableMapOf<String, Tensor>()
        for ((name, layer) in layers) {
            deltaTensors["$name.loraA"] = layer.loraA
            deltaTensors["$name.loraB"] = layer.loraB
        }
        SafeTensors.save(file, deltaTensors)
    }

    fun loadDelta(file: File) {
        val deltaTensors = SafeTensors.load(file)
        for ((name, layer) in layers) {
            val a = deltaTensors["$name.loraA"]
            val b = deltaTensors["$name.loraB"]
            if (a != null) {
                for (i in layer.loraA.data.indices) {
                    layer.loraA.data[i] = a.data[i]
                }
            }
            if (b != null) {
                for (i in layer.loraB.data.indices) {
                    layer.loraB.data[i] = b.data[i]
                }
            }
        }
    }
}
