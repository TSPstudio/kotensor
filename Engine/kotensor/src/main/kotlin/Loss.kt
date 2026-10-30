package io.kotensor.core

import kotlin.math.exp
import kotlin.math.ln

const val IGNORE_INDEX = -100

private const val FUSED_CHUNK_ELEMENTS = 1 shl 22

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
        if (prediction.requiresGrad) {
            val gradScale = out.grad[0] * (2.0f / n)
            val predictionGrad = prediction.grad
            for (i in prediction.data.indices) {
                val diff = prediction.data[i] - target.data[i]
                predictionGrad[i] += gradScale * diff
            }
        }
    }
    return out
}

@JvmOverloads
fun crossEntropyLoss(logits: Tensor, targets: IntArray, ignoreIndex: Int = IGNORE_INDEX): Tensor {
    val count = targets.size
    require(count > 0) { "targets cannot be empty" }
    require(logits.data.size % count == 0) { "logits size ${logits.data.size} must be divisible by target count $count" }
    val vocabularySize = logits.data.size / count

    val rows = activeRows(targets, ignoreIndex, vocabularySize)
    val active = rows.size
    if (active == 0) return emptyLoss()

    val probabilities = FloatArray(active * vocabularySize)
    var totalLoss = 0.0f

    for (r in 0 until active) {
        val offset = rows[r] * vocabularySize
        val probabilityOffset = r * vocabularySize
        val target = targets[rows[r]]

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
            probabilities[probabilityOffset + v] = expVal
            sumExp += expVal
        }

        val invSumExp = 1.0f / sumExp
        for (v in 0 until vocabularySize) {
            probabilities[probabilityOffset + v] *= invSumExp
        }

        val logSumExp = maxLogit + ln(sumExp)
        val sampleLoss = logSumExp - logits.data[offset + target]
        totalLoss += sampleLoss
    }

    val meanLoss = totalLoss / active.toFloat()
    val outData = floatArrayOf(meanLoss)
    val outShape = intArrayOf(1, 1)

    lateinit var out: Tensor
    out = Tensor(outData, outShape, listOf(logits)) {
        if (logits.requiresGrad) {
            val upstreamGrad = out.grad[0]
            val scale = upstreamGrad / active.toFloat()
            val logitsGrad = logits.grad

            for (r in 0 until active) {
                val offset = rows[r] * vocabularySize
                val probabilityOffset = r * vocabularySize
                val target = targets[rows[r]]

                for (v in 0 until vocabularySize) {
                    val prob = probabilities[probabilityOffset + v]
                    val indicator = if (v == target) 1.0f else 0.0f
                    logitsGrad[offset + v] += scale * (prob - indicator)
                }
            }
        }
    }
    return out
}

@JvmOverloads
fun crossEntropyLoss(logits: Tensor, targets: Tensor, ignoreIndex: Int = IGNORE_INDEX): Tensor {
    return crossEntropyLoss(logits, targetIndices(targets), ignoreIndex)
}

@JvmOverloads
fun fusedCrossEntropyLoss(hidden: Tensor, head: Projection, targets: IntArray, ignoreIndex: Int = IGNORE_INDEX): Tensor {
    val dimension = head.inSize
    val vocabularySize = head.outSize
    require(targets.isNotEmpty()) { "targets cannot be empty" }
    require(hidden.data.size == targets.size * dimension) {
        "hidden has ${hidden.data.size} elements, expected ${targets.size} × $dimension"
    }
    val weight = when (head) {
        is Linear -> head.weight
        is QuantizedLinear -> null
        else -> throw IllegalArgumentException("fusedCrossEntropyLoss needs a Linear or QuantizedLinear head, got ${head::class.simpleName}")
    }
    val quantized = (head as? QuantizedLinear)?.weight
    val bias = head.bias

    val rows = activeRows(targets, ignoreIndex, vocabularySize)
    val active = rows.size
    if (active == 0) return emptyLoss()

    val chunkRows = (FUSED_CHUNK_ELEMENTS / vocabularySize).coerceIn(1, active)
    val inputs = FloatArray(chunkRows * dimension)
    val logits = FloatArray(chunkRows * vocabularySize)
    val rowMax = FloatArray(active)
    val rowSum = FloatArray(active)
    val scale = 1.0f / active.toFloat()
    val hiddenGrad = if (hidden.requiresGrad) FloatArray(active * dimension) else null
    val chunkGrad = if (hiddenGrad != null) FloatArray(chunkRows * dimension) else null
    val biasGrad = if (bias.requiresGrad) FloatArray(vocabularySize) else null
    var totalLoss = 0.0f

    var start = 0
    while (start < active) {
        val count = minOf(chunkRows, active - start)
        projectRows(hidden.data, rows, start, count, dimension, vocabularySize, weight, quantized, bias, inputs, logits)
        for (r in 0 until count) {
            val offset = r * vocabularySize
            var maxLogit = -1e9f
            for (v in 0 until vocabularySize) {
                val logit = logits[offset + v]
                if (logit > maxLogit) {
                    maxLogit = logit
                }
            }
            var sumExp = 0.0f
            for (v in 0 until vocabularySize) {
                sumExp += exp(logits[offset + v] - maxLogit)
            }
            rowMax[start + r] = maxLogit
            rowSum[start + r] = sumExp
            totalLoss += maxLogit + ln(sumExp) - logits[offset + targets[rows[start + r]]]
        }
        if (hiddenGrad != null || biasGrad != null) {
            logitGradients(logits, count, vocabularySize, targets, rows, start, rowMax, rowSum, scale)
            if (biasGrad != null) {
                for (r in 0 until count) {
                    val offset = r * vocabularySize
                    for (v in 0 until vocabularySize) {
                        biasGrad[v] += logits[offset + v]
                    }
                }
            }
            if (hiddenGrad != null && chunkGrad != null) {
                if (weight != null) {
                    MatMul.multiplyTransposedB(logits, weight.data, chunkGrad, count, vocabularySize, dimension, accumulate = false)
                } else if (quantized != null) {
                    MatMul.multiplyByQuantizedTransposed(logits, quantized, chunkGrad, count, accumulate = false)
                }
                System.arraycopy(chunkGrad, 0, hiddenGrad, start * dimension, count * dimension)
            }
        }
        start += count
    }

    val meanLoss = totalLoss / active.toFloat()
    lateinit var out: Tensor
    out = Tensor(floatArrayOf(meanLoss), intArrayOf(1, 1), listOfNotNull(hidden, weight, bias)) {
        val upstreamGrad = out.grad[0]
        if (hiddenGrad != null && hidden.requiresGrad) {
            val target = hidden.grad
            for (r in 0 until active) {
                val destination = rows[r] * dimension
                val source = r * dimension
                for (d in 0 until dimension) {
                    target[destination + d] += upstreamGrad * hiddenGrad[source + d]
                }
            }
        }
        if (biasGrad != null && bias.requiresGrad) {
            val target = bias.grad
            for (v in 0 until vocabularySize) {
                target[v] += upstreamGrad * biasGrad[v]
            }
        }
        if (weight != null && weight.requiresGrad) {
            var chunkStart = 0
            while (chunkStart < active) {
                val count = minOf(chunkRows, active - chunkStart)
                projectRows(hidden.data, rows, chunkStart, count, dimension, vocabularySize, weight, null, bias, inputs, logits)
                logitGradients(logits, count, vocabularySize, targets, rows, chunkStart, rowMax, rowSum, scale)
                MatMul.multiplyTransposedA(inputs, logits, weight.grad, dimension, count, vocabularySize, accumulate = true, alpha = upstreamGrad)
                chunkStart += count
            }
        }
    }
    return out
}

@JvmOverloads
fun fusedCrossEntropyLoss(hidden: Tensor, head: Projection, targets: Tensor, ignoreIndex: Int = IGNORE_INDEX): Tensor {
    return fusedCrossEntropyLoss(hidden, head, targetIndices(targets), ignoreIndex)
}

internal fun targetIndices(targets: Tensor): IntArray = IntArray(targets.data.size) { i -> targets.data[i].toInt() }

internal fun countActiveTargets(targets: Tensor, ignoreIndex: Int = IGNORE_INDEX): Int {
    var active = 0
    for (value in targets.data) {
        if (value.toInt() != ignoreIndex) active++
    }
    return active
}

private fun activeRows(targets: IntArray, ignoreIndex: Int, vocabularySize: Int): IntArray {
    var active = 0
    for (target in targets) {
        if (target == ignoreIndex) continue
        require(target in 0 until vocabularySize) { "target $target out of bounds $vocabularySize" }
        active++
    }
    val rows = IntArray(active)
    var next = 0
    for (i in targets.indices) {
        if (targets[i] != ignoreIndex) rows[next++] = i
    }
    return rows
}

private fun emptyLoss(): Tensor = Tensor(floatArrayOf(0.0f), intArrayOf(1, 1)).also { it.requiresGrad = false }

private fun projectRows(
    source: FloatArray, rows: IntArray, start: Int, count: Int,
    dimension: Int, vocabularySize: Int,
    weight: Tensor?, quantized: QuantizedTensor?, bias: Tensor,
    inputs: FloatArray, logits: FloatArray
) {
    for (r in 0 until count) {
        System.arraycopy(source, rows[start + r] * dimension, inputs, r * dimension, dimension)
    }
    if (weight != null) {
        MatMul.multiply(inputs, weight.data, logits, count, dimension, vocabularySize, accumulate = false)
    } else if (quantized != null) {
        MatMul.multiplyQuantized(inputs, quantized, logits, count, accumulate = false)
    }
    for (r in 0 until count) {
        val offset = r * vocabularySize
        for (v in 0 until vocabularySize) {
            logits[offset + v] += bias.data[v]
        }
    }
}

private fun logitGradients(
    logits: FloatArray, count: Int, vocabularySize: Int,
    targets: IntArray, rows: IntArray, start: Int,
    rowMax: FloatArray, rowSum: FloatArray, scale: Float
) {
    for (r in 0 until count) {
        val offset = r * vocabularySize
        val maxLogit = rowMax[start + r]
        val invSumExp = 1.0f / rowSum[start + r]
        val target = targets[rows[start + r]]
        for (v in 0 until vocabularySize) {
            val prob = exp(logits[offset + v] - maxLogit) * invSumExp
            val indicator = if (v == target) 1.0f else 0.0f
            logits[offset + v] = scale * (prob - indicator)
        }
    }
}
