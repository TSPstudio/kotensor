package io.kotensor.core

import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class MatMulTest {

    private fun naive(a: FloatArray, b: FloatArray, m: Int, k: Int, n: Int): FloatArray {
        val c = FloatArray(m * n)
        for (i in 0 until m) {
            for (j in 0 until n) {
                var sum = 0.0f
                for (l in 0 until k) {
                    sum += a[i * k + l] * b[l * n + j]
                }
                c[i * n + j] = sum
            }
        }
        return c
    }

    private fun assertClose(expected: FloatArray, actual: FloatArray, tolerance: Float, label: String) {
        assertEquals(expected.size, actual.size, "$label size")
        for (i in expected.indices) {
            assertEquals(expected[i], actual[i], tolerance, "$label element $i")
        }
    }

    @Test
    fun testTiledKernelMatchesNaiveOnOddShapes() {
        val random = Random(3)
        val shapes = listOf(
            Triple(1, 1, 1), Triple(1, 7, 300), Triple(3, 5, 4), Triple(32, 64, 256), Triple(33, 65, 257),
            Triple(70, 9, 513), Triple(4, 1000, 3), Triple(130, 17, 40)
        )
        for ((m, k, n) in shapes) {
            val a = FloatArray(m * k) { random.nextFloat() - 0.5f }
            val b = FloatArray(k * n) { random.nextFloat() - 0.5f }
            val expected = naive(a, b, m, k, n)

            val c = FloatArray(m * n)
            MatMul.multiply(a, b, c, m, k, n, accumulate = false)
            assertClose(expected, c, 1e-4f * k, "[$m,$k]×[$k,$n]")

            val accumulated = FloatArray(m * n) { 1.0f }
            MatMul.multiply(a, b, accumulated, m, k, n, accumulate = true, alpha = 2.0f)
            for (i in accumulated.indices) {
                assertEquals(1.0f + 2.0f * expected[i], accumulated[i], 2e-4f * k, "accumulate [$m,$k]×[$k,$n] element $i")
            }
        }
    }

    @Test
    fun testParallelTilesMatchSerial() {
        val random = Random(5)
        val m = 97
        val k = 50
        val n = 1030
        val a = FloatArray(m * k) { random.nextFloat() - 0.5f }
        val b = FloatArray(k * n) { random.nextFloat() - 0.5f }
        val expected = naive(a, b, m, k, n)

        val saved = MatMul.parallelThreshold
        try {
            MatMul.parallelThreshold = 0L
            val parallel = FloatArray(m * n)
            MatMul.multiply(a, b, parallel, m, k, n, accumulate = false)
            assertClose(expected, parallel, 1e-4f * k, "parallel")

            MatMul.parallelThreshold = Long.MAX_VALUE
            val serial = FloatArray(m * n)
            MatMul.multiply(a, b, serial, m, k, n, accumulate = false)
            assertClose(parallel, serial, 0.0f, "serial vs parallel")
        } finally {
            MatMul.parallelThreshold = saved
        }
        assertTrue(MatMul.numThreads >= 1)
    }

    @Test
    fun testTranspose() {
        val rows = 37
        val cols = 70
        val src = FloatArray(rows * cols) { i -> i.toFloat() }
        val dst = FloatArray(rows * cols)
        MatMul.transpose(src, rows, cols, dst)
        for (i in 0 until rows) {
            for (j in 0 until cols) {
                assertEquals(src[i * cols + j], dst[j * rows + i])
            }
        }
    }

    @Test
    fun testMatmulBackwardMatchesNaiveGradients() {
        val random = Random(9)
        val m = 5
        val k = 70
        val n = 300
        val a = Tensor(FloatArray(m * k) { random.nextFloat() - 0.5f }, intArrayOf(m, k))
        val b = Tensor(FloatArray(k * n) { random.nextFloat() - 0.5f }, intArrayOf(k, n))
        val out = a matmul b
        for (i in out.grad.indices) out.grad[i] = random.nextFloat() - 0.5f
        val upstream = out.grad.copyOf()
        out.backwardOp()

        for (i in 0 until m) {
            for (l in 0 until k) {
                var expected = 0.0f
                for (j in 0 until n) expected += upstream[i * n + j] * b.data[l * n + j]
                assertEquals(expected, a.grad[i * k + l], 1e-3f, "dA[$i,$l]")
            }
        }
        for (l in 0 until k) {
            for (j in 0 until n) {
                var expected = 0.0f
                for (i in 0 until m) expected += a.data[i * k + l] * upstream[i * n + j]
                assertEquals(expected, b.grad[l * n + j], 1e-3f, "dB[$l,$j]")
            }
        }
    }

    @Test
    fun testInPlaceOps() {
        val a = Tensor(floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f), intArrayOf(2, 3))
        val b = Tensor(floatArrayOf(1f, 0f, 0f, 1f, 1f, 1f), intArrayOf(3, 2))
        val out = Tensor(FloatArray(4), intArrayOf(2, 2))

        a.matmulInto(b, out)
        assertEquals(listOf(4f, 5f, 10f, 11f), out.data.toList())
        a.matmulInto(b, out, accumulate = true, scale = 0.5f)
        assertEquals(listOf(6f, 7.5f, 15f, 16.5f), out.data.toList())
        assertTrue(out.parents.isEmpty())

        val row = Tensor(floatArrayOf(10f, 20f), intArrayOf(1, 2))
        val sum = Tensor(FloatArray(4), intArrayOf(2, 2))
        out.addInto(row, sum)
        assertEquals(listOf(16f, 27.5f, 25f, 36.5f), sum.data.toList())
        sum.addInPlace(sum)
        assertEquals(listOf(32f, 55f, 50f, 73f), sum.data.toList())

        val x = Tensor(floatArrayOf(0f, 1f, -1f), intArrayOf(1, 3))
        val reference = x.silu().data.copyOf()
        x.siluInPlace()
        for (i in reference.indices) assertEquals(reference[i], x.data[i], 1e-6f)

        x.mulInPlace(Tensor(floatArrayOf(2f, 2f, 2f), intArrayOf(1, 3)))
        for (i in reference.indices) assertEquals(2f * reference[i], x.data[i], 1e-6f)

        x.copyFrom(Tensor(floatArrayOf(7f, 8f, 9f), intArrayOf(3, 1)))
        assertEquals(listOf(7f, 8f, 9f), x.data.toList())
    }
}

