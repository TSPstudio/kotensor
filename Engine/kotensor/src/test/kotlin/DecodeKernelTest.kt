package io.kotensor.core

import java.io.File
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class DecodeKernelTest {

    private fun mappedCopy(weight: QuantizedTensor): QuantizedTensor {
        val holder = Transformer(vocabularySize = weight.columns, dimension = weight.rows, numLayers = 1, numHeads = 1, hiddenDimension = 4, quantization = Quantization(weight.bits, weight.groupSize))
        (holder.lmHead as QuantizedLinear).weight = weight
        val file = File.createTempFile("kotensor_decode", ".safetensors")
        file.deleteOnExit()
        holder.saveQuantized(file)
        val mapped = (Transformer.loadQuantized(file, memoryMapped = true).lmHead as QuantizedLinear).weight
        assertTrue(mapped.isMemoryMapped)
        return mapped
    }

    private fun maxAbsDiff(a: FloatArray, b: FloatArray): Float {
        var max = 0.0f
        for (i in a.indices) max = maxOf(max, kotlin.math.abs(a[i] - b[i]))
        return max
    }

    private fun <T> withSettings(threads: Int, lookup: Boolean, threshold: Long, block: () -> T): T {
        val savedThreads = MatMul.numThreads
        val savedLookup = MatMul.lookupInt4
        val savedThreshold = MatMul.parallelThreshold
        try {
            MatMul.numThreads = threads
            MatMul.lookupInt4 = lookup
            MatMul.parallelThreshold = threshold
            return block()
        } finally {
            MatMul.numThreads = savedThreads
            MatMul.lookupInt4 = savedLookup
            MatMul.parallelThreshold = savedThreshold
        }
    }

    @Test
    fun testSingleRowKernelsAgreeBitForBit() {
        val random = Random(8)
        val shapes = listOf(intArrayOf(130, 9001), intArrayOf(300, 2100), intArrayOf(200, 33), intArrayOf(70, 300), intArrayOf(450, 129), intArrayOf(2, 17))
        for (shape in shapes) {
            val k = shape[0]
            val n = shape[1]
            val w = FloatArray(k * n) { (random.nextFloat() - 0.5f) * 0.2f }
            val a = FloatArray(k) { random.nextFloat() - 0.5f }
            for (quantization in listOf(Quantization.INT4, Quantization(4, 32), Quantization.INT8)) {
                val heap = QuantizedTensor.quantize(w, k, n, quantization)
                val mapped = mappedCopy(heap)
                val reference = FloatArray(n)
                MatMul.multiply(a, heap.dequantize().data, reference, 1, k, n, accumulate = false)
                var first: FloatArray? = null
                for (weight in listOf(heap, mapped)) {
                    for (lookup in listOf(true, false)) {
                        for ((threads, threshold) in listOf(1 to Long.MAX_VALUE, 4 to 0L, 3 to 0L)) {
                            val actual = withSettings(threads, lookup, threshold) {
                                FloatArray(n).also { MatMul.multiplyQuantized(a, weight, it, 1, accumulate = false) }
                            }
                            val label = "[$k,$n] $quantization mapped=${weight.isMemoryMapped} lookup=$lookup threads=$threads"
                            assertTrue(maxAbsDiff(reference, actual) < 1e-4f, label)
                            val expected = first ?: actual.also { first = it }
                            assertContentEquals(expected, actual, label)
                        }
                    }
                }
            }
        }
    }

    @Test
    fun testSingleRowKernelAccumulatesWithAlpha() {
        val random = Random(9)
        val k = 96
        val n = 8200
        val weight = QuantizedTensor.quantize(FloatArray(k * n) { random.nextFloat() - 0.5f }, k, n, Quantization.INT4)
        val a = FloatArray(k) { random.nextFloat() - 0.5f }
        val plain = FloatArray(n).also { MatMul.multiplyQuantized(a, weight, it, 1, accumulate = false) }
        val accumulated = FloatArray(n) { 2.0f }
        MatMul.multiplyQuantized(a, weight, accumulated, 1, accumulate = true, alpha = -0.5f)
        for (j in 0 until n) assertEquals(2.0f - 0.5f * plain[j], accumulated[j], 1e-4f)
    }

    @Test
    fun testThreadCountCanChangeAtRuntime() {
        val saved = MatMul.numThreads
        try {
            assertFailsWith<IllegalArgumentException> { MatMul.numThreads = 0 }
            MatMul.numThreads = 2
            assertEquals(2, MatMul.numThreads)
            val random = Random(10)
            val weight = QuantizedTensor.quantize(FloatArray(256 * 4096) { random.nextFloat() - 0.5f }, 256, 4096, Quantization.INT4)
            val a = FloatArray(256) { random.nextFloat() - 0.5f }
            val two = FloatArray(4096).also { MatMul.multiplyQuantized(a, weight, it, 1, accumulate = false) }
            MatMul.numThreads = 5
            val five = FloatArray(4096).also { MatMul.multiplyQuantized(a, weight, it, 1, accumulate = false) }
            assertContentEquals(two, five)
        } finally {
            MatMul.numThreads = saved
        }
    }

    @Test
    fun testInt4KernelDefaultsToLookupOnAndroid() {
        assertTrue(MatMul.defaultLookupInt4(null, "Dalvik"))
        assertFalse(MatMul.defaultLookupInt4(null, "OpenJDK 64-Bit Server VM"))
        assertFalse(MatMul.defaultLookupInt4(null, null))
        assertTrue(MatMul.defaultLookupInt4(" Lookup ", "OpenJDK 64-Bit Server VM"))
        assertFalse(MatMul.defaultLookupInt4("unpack", "Dalvik"))
        assertTrue(MatMul.defaultLookupInt4("other", "Dalvik"))
        assertEquals(MatMul.defaultLookupInt4(System.getProperty("kotensor.int4Kernel"), System.getProperty("java.vm.name")), MatMul.lookupInt4)
    }
}
