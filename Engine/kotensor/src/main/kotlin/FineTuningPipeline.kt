package io.kotensor.core

import java.io.File

class FineTuningPipeline(
    val model: Transformer,
    val adapter: LoRAAdapter,
    val optimizer: AdamW
) {

    fun trainStep(input: Tensor, target: Tensor): Float {
        val logits = model.forward(input)
        val loss = crossEntropyLoss(logits, target)
        loss.backward()
        optimizer.step()
        optimizer.zeroGrad()
        return loss.data[0]
    }

    fun train(
        dataLoader: DataLoader,
        epochs: Int,
        onEpochEnd: (Int, Float) -> Unit = { _, _ -> }
    ) {
        for (epoch in 1..epochs) {
            var totalLoss = 0.0f
            var batchCount = 0
            dataLoader.reset()
            while (dataLoader.hasNext()) {
                val (x, y) = dataLoader.nextBatch()
                val loss = trainStep(x, y)
                totalLoss += loss
                batchCount++
            }
            val avgLoss = if (batchCount > 0) totalLoss / batchCount else 0.0f
            onEpochEnd(epoch, avgLoss)
        }
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
