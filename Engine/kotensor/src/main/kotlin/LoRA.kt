package io.kotensor.core

import java.io.File
import kotlin.math.sqrt
import kotlin.random.Random

class LoRALinear(
    val baseLinear: Projection,
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

    private val lowRank = Tensor(FloatArray(rank), intArrayOf(1, rank))

    override fun forwardInto(input: Tensor, out: Tensor): Tensor {
        require(input.shape.size == 2 && input.shape[0] == 1) { "forwardInto takes a single token row, got ${input.shape.contentToString()}" }
        baseLinear.forwardInto(input, out)
        input.matmulInto(loraA, lowRank)
        return lowRank.matmulInto(loraB, out, accumulate = true, scale = scaling)
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
            val qLinear = block.attention.wQ as? Projection
            if (qLinear != null) {
                val qLora = LoRALinear(qLinear, rank, alpha)
                block.attention.wQ = qLora
                layers["block.$l.attention.wQ"] = qLora
            }
            val vLinear = block.attention.wV as? Projection
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
        val merged = layers.values.toSet()
        for (layer in merged) {
            val base = layer.baseLinear as? Linear
                ?: throw IllegalStateException("Cannot merge an adapter into quantized weights; export it with saveDelta instead")
            val delta = layer.deltaWeights()
            for (i in base.weight.data.indices) {
                base.weight.data[i] += delta.data[i]
            }
        }
        for (block in model.blocks) {
            val q = block.attention.wQ
            if (q is LoRALinear && q in merged) {
                block.attention.wQ = q.baseLinear
            }
            val v = block.attention.wV
            if (v is LoRALinear && v in merged) {
                block.attention.wV = v.baseLinear
            }
        }
        layers.clear()
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

