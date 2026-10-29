package io.kotensor.benchmark

import io.kotensor.core.InferenceBuffers
import io.kotensor.core.KVCache
import io.kotensor.core.Quantization
import io.kotensor.core.Transformer
import java.util.concurrent.TimeUnit
import kotlin.random.Random
import org.openjdk.jmh.annotations.Benchmark
import org.openjdk.jmh.annotations.BenchmarkMode
import org.openjdk.jmh.annotations.Level
import org.openjdk.jmh.annotations.Mode
import org.openjdk.jmh.annotations.OutputTimeUnit
import org.openjdk.jmh.annotations.Param
import org.openjdk.jmh.annotations.Scope
import org.openjdk.jmh.annotations.Setup
import org.openjdk.jmh.annotations.State
import org.openjdk.jmh.infra.Blackhole

@State(Scope.Benchmark)
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.SECONDS)
open class DecodeBenchmark {
    @JvmField
    @Param("16", "128", "512", "2048")
    var sequenceLength: Int = 16

    @JvmField
    @Param("fp32", "int8", "int4")
    var weights: String = "fp32"

    private lateinit var model: Transformer
    private lateinit var cache: KVCache
    private lateinit var buffers: InferenceBuffers
    private var token = 1

    @Setup(Level.Trial)
    fun setup() {
        model = Models.smolLM135M(
            when (weights) {
                "int8" -> Quantization.INT8
                "int4" -> Quantization.INT4
                else -> null
            }
        )
        cache = model.newCache(sequenceLength + 1)
        buffers = model.newInferenceBuffers(sequenceLength + 1)
        val random = Random(3)
        for (i in cache.keys.indices) {
            cache.keys[i] = random.nextFloat() - 0.5f
            cache.values[i] = random.nextFloat() - 0.5f
        }
        cache.advance(sequenceLength)
    }

    @Benchmark
    fun decodeOneToken(blackhole: Blackhole) {
        val logits = model.decode(token, cache, buffers)
        token = (token * 17 + 3) % model.vocabularySize
        blackhole.consume(logits.data[0])
        cache.reset()
        cache.advance(sequenceLength)
    }
}
