package io.kotensor.core


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