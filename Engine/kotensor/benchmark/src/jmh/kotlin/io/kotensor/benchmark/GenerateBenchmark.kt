package io.kotensor.benchmark

import io.kotensor.core.SimpleTokenizer
import io.kotensor.core.StreamingSession
import io.kotensor.core.Transformer
import java.util.concurrent.TimeUnit
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
open class GenerateBenchmark {
    companion object {
        const val TOKENS = 128
    }

    private lateinit var model: Transformer
    private lateinit var prompt: IntArray
    private lateinit var tokenizer: SimpleTokenizer
    private lateinit var session: StreamingSession

    @Setup
    fun setup() {
        model = Models.small()
        prompt = intArrayOf(1, 2, 3, 4, 5, 6, 7, 8)
        tokenizer = SimpleTokenizer(String(CharArray(Models.SMALL_VOCAB) { c -> c.toChar() }))
        session = StreamingSession(model, tokenizer)
    }

    @Benchmark
    @OperationsPerInvocation(TOKENS)
    fun generateTokens(blackhole: Blackhole) {
        blackhole.consume(model.generate(prompt, TOKENS))
    }

    @Benchmark
    @OperationsPerInvocation(TOKENS)
    fun streamText(blackhole: Blackhole) {
        val state = session.stream("abcdefgh", TOKENS) { }
        blackhole.consume(state.generatedText)
    }

    @Benchmark
    fun prefillPrompt(blackhole: Blackhole) {
        val cache = model.newCache(prompt.size)
        blackhole.consume(model.forward(prompt, cache))
    }
}
