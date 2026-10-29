package io.kotensor.benchmark

import io.kotensor.core.Quantization
import io.kotensor.core.Transformer
import kotlin.random.Random

object Models {
    const val SMALL_VOCAB = 256
    const val SMALL_DIMENSION = 128
    const val SMALL_LAYERS = 4
    const val SMALL_HEADS = 4

    fun small(seed: Int = 1): Transformer {
        val model = Transformer(SMALL_VOCAB, SMALL_DIMENSION, SMALL_LAYERS, SMALL_HEADS)
        val random = Random(seed)
        for (p in model.parameters()) {
            for (i in p.data.indices) p.data[i] = (random.nextFloat() * 2f - 1f) * 0.05f
        }
        return model
    }

    fun smolLM135M(quantization: Quantization? = null): Transformer {
        val model = Transformer(
            vocabularySize = 49152,
            dimension = 576,
            numLayers = 30,
            numHeads = 9,
            hiddenDimension = 1536,
            numKeyValueHeads = 3,
            normEps = 1e-5f,
            ropeTheta = 100000.0f
        )
        if (quantization != null) model.quantize(quantization)
        return model
    }
}
