package io.kotensor.core

import kotlin.math.exp
import kotlin.math.ln

fun mseLoss(prediction: Tensor, target: Tensor): Tensor {
    require(prediction.shape.contentEquals(target.shape)) { "shape mismatch" }
    val n = prediction.data.size.toFloat()

    var sumSq = 0.0f
    for (i in prediction.data.indices) {
        val diff = prediction.data[i] - target.data[i]
        sumSq += diff * diff
    }
    val lossValue = sumSq / n
    val outData = floatArrayOf(lossValue)
    val outShape = intArrayOf(1, 1)

    lateinit var out: Tensor
    out = Tensor(outData, outShape, listOf(prediction, target)) {
        val gradScale = out.grad[0] * (2.0f / n)
        for (i in prediction.data.indices) {
            val diff = prediction.data[i] - target.data[i]
            prediction.grad[i] += gradScale * diff
        }
    }
    return out
}

fun crossEntropyLoss(logits: Tensor, targets: IntArray): Tensor {
    val count = targets.size
    require(count > 0) { "targets cannot be empty" }
    require(logits.data.size % count == 0) { "logits size ${logits.data.size} must be divisible by target count $count" }
    val vocabularySize = logits.data.size / count

    val probabilities = FloatArray(logits.data.size)
    var totalLoss = 0.0f

    for (i in 0 until count) {
        val offset = i * vocabularySize
        val target = targets[i]
        require(target in 0 until vocabularySize) { "target $target out of bounds $vocabularySize" }

        var maxLogit = -1e9f
        for (v in 0 until vocabularySize) {
            val logit = logits.data[offset + v]
            if (logit > maxLogit) {
                maxLogit = logit
            }
        }

        var sumExp = 0.0f
        for (v in 0 until vocabularySize) {
            val expVal = exp(logits.data[offset + v] - maxLogit)
            probabilities[offset + v] = expVal
            sumExp += expVal
        }

        val invSumExp = 1.0f / sumExp
        for (v in 0 until vocabularySize) {
            probabilities[offset + v] *= invSumExp
        }

        val logSumExp = maxLogit + ln(sumExp)
        val sampleLoss = logSumExp - logits.data[offset + target]
        totalLoss += sampleLoss
    }

    val meanLoss = totalLoss / count.toFloat()
    val outData = floatArrayOf(meanLoss)
    val outShape = intArrayOf(1, 1)

    lateinit var out: Tensor
    out = Tensor(outData, outShape, listOf(logits)) {
        val upstreamGrad = out.grad[0]
        val scale = upstreamGrad / count.toFloat()

        for (i in 0 until count) {
            val offset = i * vocabularySize
            val target = targets[i]

            for (v in 0 until vocabularySize) {
                val prob = probabilities[offset + v]
                val indicator = if (v == target) 1.0f else 0.0f
                logits.grad[offset + v] += scale * (prob - indicator)
            }
        }
    }
    return out
}

fun crossEntropyLoss(logits: Tensor, targets: Tensor): Tensor {
    val targetIndices = IntArray(targets.data.size) { i -> targets.data[i].toInt() }
    return crossEntropyLoss(logits, targetIndices)
}
