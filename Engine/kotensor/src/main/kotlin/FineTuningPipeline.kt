package io.kotensor.core

import java.io.File

class FineTuningPipeline(
    val model: Transformer,
    val adapter: LoRAAdapter,
    val optimizer: AdamW
) {
    private val trainedTensors = optimizer.parameters.toHashSet()

    fun trainStep(input: Tensor, target: Tensor): Float {
        if (countActiveTargets(target) == 0) return 0.0f
        val loss = model.loss(input, target)
        loss.backward()
        optimizer.step()
        optimizer.zeroGrad()
        clearFrozenGradients()
        return loss.data[0]
    }

    private fun clearFrozenGradients() {
        for (p in model.parameters()) {
            if (p.hasGrad && p !in trainedTensors) p.grad.fill(0.0f)
        }
        for (layer in adapter.layers.values) {
            for (p in layer.baseLinear.parameters()) {
                if (p.hasGrad && p !in trainedTensors) p.grad.fill(0.0f)
            }
        }
    }

    fun train(
        dataLoader: DataLoader,
        epochs: Int,
        onBatchEnd: (epoch: Int, batch: Int, totalBatches: Int, loss: Float) -> Unit = { _, _, _, _ -> },
        onEpochEnd: (epoch: Int, avgLoss: Float) -> Unit = { _, _ -> }
    ) {
        val totalBatches = dataLoader.totalBatches
        for (epoch in 1..epochs) {
            var totalLoss = 0.0f
            var trainedBatches = 0
            var batch = 0
            dataLoader.reset()
            while (dataLoader.hasNext()) {
                val (x, y) = dataLoader.nextBatch()
                batch++
                val loss = trainStep(x, y)
                if (countActiveTargets(y) > 0) {
                    totalLoss += loss
                    trainedBatches++
                }
                onBatchEnd(epoch, batch, totalBatches, loss)
            }
            val avgLoss = if (trainedBatches > 0) totalLoss / trainedBatches else 0.0f
            onEpochEnd(epoch, avgLoss)
        }
    }

    @Deprecated("Use train(dataLoader, epochs, onBatchEnd, onEpochEnd)", level = DeprecationLevel.HIDDEN)
    fun train(dataLoader: DataLoader, epochs: Int, onEpochEnd: (Int, Float) -> Unit = { _, _ -> }) {
        train(dataLoader, epochs, onEpochEnd = onEpochEnd)
    }

    fun exportAdapter(file: File) {
        adapter.saveDelta(file)
    }

    fun loadAdapter(file: File) {
        adapter.loadDelta(file)
    }

    fun mergeAdapter() {
        adapter.merge()
    }
}
