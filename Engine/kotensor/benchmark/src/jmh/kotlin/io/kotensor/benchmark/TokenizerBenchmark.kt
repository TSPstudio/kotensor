package io.kotensor.benchmark

import io.kotensor.core.BPETokenizer
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
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.SECONDS)
open class TokenizerBenchmark {
    private lateinit var tokenizer: BPETokenizer
    private lateinit var text: String
    private lateinit var ids: IntArray

    @Setup
    fun setup() {
        val stream = TokenizerBenchmark::class.java.getResourceAsStream("/tokenizer_qwenstyle.json")
            ?: throw IllegalStateException("tokenizer_qwenstyle.json missing from benchmark resources")
        tokenizer = BPETokenizer.fromJson(stream.use { s -> s.readBytes().toString(Charsets.UTF_8) })
        text = List(200) { i -> "Token $i of a benchmark sentence with punctuation, números y emoji 🚀." }.joinToString(" ")
        ids = tokenizer.encode(text)
    }

    @Benchmark
    fun encode(blackhole: Blackhole) {
        blackhole.consume(tokenizer.encode(text))
    }

    @Benchmark
    fun decode(blackhole: Blackhole) {
        blackhole.consume(tokenizer.decode(ids))
    }
}
