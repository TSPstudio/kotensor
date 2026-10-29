package io.kotensor.benchmark

import io.kotensor.core.AdamW
import io.kotensor.core.LoRAAdapter
import io.kotensor.core.Quantization
import io.kotensor.core.Tensor
import io.kotensor.core.Transformer
import io.kotensor.core.crossEntropyLoss
import java.util.concurrent.TimeUnit
import org.openjdk.jmh.annotations.Benchmark
import org.openjdk.jmh.annotations.BenchmarkMode
import org.openjdk.jmh.annotations.Mode
import org.openjdk.jmh.annotations.OutputTimeUnit
import org.openjdk.jmh.annotations.Scope
import org.openjdk.jmh.annotations.Setup
import org.openjdk.jmh.annotations.State
import org.openjdk.jmh.infra.Blackhole

@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
open class TrainingBenchmark {
    companion object {
        const val BATCH = 4
        const val SEQUENCE = 32
    }

    private lateinit var model: Transformer
    private lateinit var optimizer: AdamW
    private lateinit var quantized: Transformer
    private lateinit var adapterOptimizer: AdamW
    private lateinit var inputs: Tensor
    private lateinit var targets: Tensor

    @Setup
    fun setup() {
        model = Models.small()
        optimizer = AdamW(model.parameters(), learningRate = 1e-3f)
        quantized = Models.small(seed = 2).quantize(Quantization.INT8)
        val adapter = LoRAAdapter(quantized, rank = 8)
        adapterOptimizer = AdamW(adapter.parameters(), learningRate = 1e-3f)
        inputs = Tensor(FloatArray(BATCH * SEQUENCE) { i -> ((i * 7) % Models.SMALL_VOCAB).toFloat() }, intArrayOf(BATCH, SEQUENCE))
        targets = Tensor(FloatArray(BATCH * SEQUENCE) { i -> ((i * 7 + 1) % Models.SMALL_VOCAB).toFloat() }, intArrayOf(BATCH, SEQUENCE))
    }

    @Benchmark
    fun fullFineTuningStep(blackhole: Blackhole) {
        val loss = crossEntropyLoss(model.forward(inputs), targets)
        loss.backward()
        optimizer.step()
        optimizer.zeroGrad()
        blackhole.consume(loss.data[0])
    }

    @Benchmark
    fun qloraStep(blackhole: Blackhole) {
        val loss = crossEntropyLoss(quantized.forward(inputs), targets)
        loss.backward()
        adapterOptimizer.step()
        adapterOptimizer.zeroGrad()
        for (p in quantized.parameters()) p.grad.fill(0.0f)
        blackhole.consume(loss.data[0])
    }
}
