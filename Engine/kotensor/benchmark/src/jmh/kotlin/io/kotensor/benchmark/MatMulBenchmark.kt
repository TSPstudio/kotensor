package io.kotensor.benchmark

import io.kotensor.core.MatMul
import io.kotensor.core.Quantization
import io.kotensor.core.QuantizedTensor
import java.util.concurrent.TimeUnit
import kotlin.random.Random
import org.openjdk.jmh.annotations.Benchmark
import org.openjdk.jmh.annotations.BenchmarkMode
import org.openjdk.jmh.annotations.Mode
import org.openjdk.jmh.annotations.OperationsPerInvocation
import org.openjdk.jmh.annotations.OutputTimeUnit
import org.openjdk.jmh.annotations.Scope
import org.openjdk.jmh.annotations.Setup
import org.openjdk.jmh.annotations.State
import org.openjdk.jmh.infra.Blackhole

@State(Scope.Benchmark)
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.SECONDS)
open class MatMulBenchmark {
    companion object {
        const val SQUARE = 512
        const val SQUARE_FLOPS = 2 * SQUARE * SQUARE * SQUARE
        const val PREFILL_M = 32
        const val DECODE_K = 2048
        const val DECODE_N = 8192
        const val DECODE_FLOPS = 2 * DECODE_K * DECODE_N
        const val PREFILL_FLOPS = 2 * PREFILL_M * DECODE_K * DECODE_N
    }

    private lateinit var squareA: FloatArray
    private lateinit var squareB: FloatArray
    private lateinit var squareC: FloatArray
    private lateinit var vector: FloatArray
    private lateinit var prefillRows: FloatArray
    private lateinit var weight: FloatArray
    private lateinit var weightInt8: QuantizedTensor
    private lateinit var weightInt4: QuantizedTensor
    private lateinit var decodeOut: FloatArray
    private lateinit var prefillOut: FloatArray

    @Setup
    fun setup() {
        val random = Random(42)
        squareA = FloatArray(SQUARE * SQUARE) { random.nextFloat() - 0.5f }
        squareB = FloatArray(SQUARE * SQUARE) { random.nextFloat() - 0.5f }
        squareC = FloatArray(SQUARE * SQUARE)
        vector = FloatArray(DECODE_K) { random.nextFloat() - 0.5f }
        prefillRows = FloatArray(PREFILL_M * DECODE_K) { random.nextFloat() - 0.5f }
        weight = FloatArray(DECODE_K * DECODE_N) { (random.nextFloat() - 0.5f) * 0.1f }
        weightInt8 = QuantizedTensor.quantize(weight, DECODE_K, DECODE_N, Quantization.INT8)
        weightInt4 = QuantizedTensor.quantize(weight, DECODE_K, DECODE_N, Quantization.INT4)
        decodeOut = FloatArray(DECODE_N)
        prefillOut = FloatArray(PREFILL_M * DECODE_N)
    }

    @Benchmark
    @OperationsPerInvocation(SQUARE_FLOPS)
    fun matmul512(blackhole: Blackhole) {
        MatMul.multiply(squareA, squareB, squareC, SQUARE, SQUARE, SQUARE, accumulate = false)
        blackhole.consume(squareC)
    }

    @Benchmark
    @OperationsPerInvocation(PREFILL_FLOPS)
    fun prefillFp32(blackhole: Blackhole) {
        MatMul.multiply(prefillRows, weight, prefillOut, PREFILL_M, DECODE_K, DECODE_N, accumulate = false)
        blackhole.consume(prefillOut)
    }

    @Benchmark
    @OperationsPerInvocation(DECODE_FLOPS)
    fun matvecFp32(blackhole: Blackhole) {
        MatMul.multiply(vector, weight, decodeOut, 1, DECODE_K, DECODE_N, accumulate = false)
        blackhole.consume(decodeOut)
    }

    @Benchmark
    @OperationsPerInvocation(DECODE_FLOPS)
    fun matvecInt8(blackhole: Blackhole) {
        MatMul.multiplyQuantized(vector, weightInt8, decodeOut, 1, accumulate = false)
        blackhole.consume(decodeOut)
    }

    @Benchmark
    @OperationsPerInvocation(DECODE_FLOPS)
    fun matvecInt4(blackhole: Blackhole) {
        MatMul.multiplyQuantized(vector, weightInt4, decodeOut, 1, accumulate = false)
        blackhole.consume(decodeOut)
    }
}
