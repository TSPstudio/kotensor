package io.kotensor.sample

import android.os.Debug
import io.kotensor.core.Transformer

data class MemoryTelemetry(
    val heapUsedMb: Double = 0.0,
    val heapMaxMb: Double = 0.0,
    val nativeHeapMb: Double = 0.0,
    val modelWeightsMb: Double = 0.0,
    val cacheMb: Double = 0.0,
    val cachePositions: Int = 0,
    val parameters: Long = 0L
)

object Telemetry {
    private const val MB = 1024.0 * 1024.0

    fun cacheBytes(model: Transformer, positions: Int): Long {
        return 2L * model.numLayers * model.numKeyValueHeads * positions * model.headDim * 4L
    }

    fun sample(model: Transformer?, cachePositions: Int): MemoryTelemetry {
        val runtime = Runtime.getRuntime()
        return MemoryTelemetry(
            heapUsedMb = (runtime.totalMemory() - runtime.freeMemory()) / MB,
            heapMaxMb = runtime.maxMemory() / MB,
            nativeHeapMb = Debug.getNativeHeapAllocatedSize() / MB,
            modelWeightsMb = (model?.weightBytes() ?: 0L) / MB,
            cacheMb = (model?.let { cacheBytes(it, cachePositions) } ?: 0L) / MB,
            cachePositions = cachePositions,
            parameters = model?.parameterCount() ?: 0L
        )
    }
}
