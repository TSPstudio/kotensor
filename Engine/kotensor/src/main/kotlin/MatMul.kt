package io.kotensor.core

import java.util.concurrent.ForkJoinPool
import java.util.concurrent.RecursiveAction
import kotlin.math.abs

internal interface VectorKernels {
    fun axpy(y: FloatArray, x: FloatArray, a: Float, len: Int)
    fun axpy4(y0: FloatArray, y1: FloatArray, y2: FloatArray, y3: FloatArray, x: FloatArray, a0: Float, a1: Float, a2: Float, a3: Float, len: Int)
    fun dot(a: FloatArray, aOffset: Int, b: FloatArray, bOffset: Int, len: Int): Float
    fun dot4(a: FloatArray, aOffset: Int, b: FloatArray, b0: Int, b1: Int, b2: Int, b3: Int, len: Int, out: FloatArray)
}

object MatMul {
    const val TILE_M = 32
    const val TILE_N = 256
    private const val TILE_DOT = 64
    private const val CHUNK_K = 1024
    private const val TRANSPOSE_CHUNK = 1 shl 20

    val numThreads: Int = Runtime.getRuntime().availableProcessors()

    @Volatile
    var parallelThreshold: Long = 1L shl 18

    private val pool: ForkJoinPool by lazy { ForkJoinPool(numThreads) }

    private val vectorKernels: VectorKernels? = loadVectorKernels()

    val vectorApiAvailable: Boolean
        get() = vectorKernels != null

    @Volatile
    var useVectorApi: Boolean = vectorKernels != null
        set(value) {
            require(!value || vectorKernels != null) { "The Vector API is not available: run on JDK 17+ with --add-modules jdk.incubator.vector" }
            field = value
        }

    private fun simd(): VectorKernels? = if (useVectorApi) vectorKernels else null

    private fun loadVectorKernels(): VectorKernels? {
        if (System.getProperty("kotensor.vectorApi").equals("false", ignoreCase = true)) return null
        return try {
            Class.forName("jdk.incubator.vector.FloatVector")
            val kernels = Class.forName("io.kotensor.core.SimdKernels").getDeclaredConstructor().newInstance() as VectorKernels
            if (agreesWithScalar(kernels)) kernels else null
        } catch (e: ReflectiveOperationException) {
            null
        } catch (e: LinkageError) {
            null
        } catch (e: RuntimeException) {
            null
        }
    }

    private fun agreesWithScalar(kernels: VectorKernels): Boolean {
        val len = 67
        val x = FloatArray(len) { i -> i * 0.25f - 8.0f }
        val y = FloatArray(len) { 1.0f }
        kernels.axpy(y, x, 0.5f, len)
        var expectedDot = 0.0f
        for (j in 0 until len) {
            if (abs(y[j] - (1.0f + 0.5f * x[j])) > 1e-5f) return false
            expectedDot += x[j] * x[j]
        }
        return abs(kernels.dot(x, 0, x, 0, len) - expectedDot) <= 1e-4f * expectedDot
    }

    private class Scratch {
        val rows = Array(TILE_M) { FloatArray(TILE_N) }
        val groupRows = Array(TILE_M) { FloatArray(TILE_N) }
        val bRow = FloatArray(TILE_N)
        val bytes = ByteArray(TILE_N)
        val nibbles = ByteArray(TILE_N)
        val unpacked = FloatArray(TILE_N)
        val scaleRow = FloatArray(TILE_N)
        val zeroRow = FloatArray(TILE_N)
        val rowSums = FloatArray(TILE_M)
        val dots = FloatArray(4)
        val chunks = ArrayList<FloatArray>()
        val accChunks = ArrayList<FloatArray>()
    }

    private val scratch: ThreadLocal<Scratch> = ThreadLocal.withInitial { Scratch() }

    private class TileRange(val from: Int, val to: Int, val grain: Int, val body: (Int) -> Unit) : RecursiveAction() {
        override fun compute() {
            if (to - from <= grain) {
                for (tile in from until to) body(tile)
            } else {
                val mid = (from + to) ushr 1
                invokeAll(TileRange(from, mid, grain, body), TileRange(mid, to, grain, body))
            }
        }
    }

    private inline fun forEachTile(tileCount: Int, work: Long, crossinline body: (Int) -> Unit) {
        if (tileCount == 1 || work < parallelThreshold || numThreads == 1) {
            for (tile in 0 until tileCount) body(tile)
        } else {
            val grain = maxOf(1, tileCount / (4 * numThreads))
            pool.invoke(TileRange(0, tileCount, grain) { tile -> body(tile) })
        }
    }

    private inline fun forEachStripe(n: Int, work: Long, crossinline body: (Int, Int) -> Unit) {
        val chunkCount = (n + TILE_N - 1) / TILE_N
        val stripeCount = if (work < parallelThreshold || numThreads == 1) 1 else minOf(numThreads, chunkCount)
        forEachTile(stripeCount, work) { stripe ->
            val t0 = (stripe.toLong() * chunkCount / stripeCount).toInt()
            val t1 = ((stripe + 1).toLong() * chunkCount / stripeCount).toInt()
            body(t0, t1)
        }
    }

    fun multiply(
        a: FloatArray,
        b: FloatArray,
        c: FloatArray,
        m: Int,
        k: Int,
        n: Int,
        accumulate: Boolean = true,
        alpha: Float = 1.0f
    ) {
        require(m >= 0 && k >= 0 && n >= 0) { "Negative dimension in [$m, $k] × [$k, $n]" }
        require(a.size >= m * k) { "a has ${a.size} elements, needs $m × $k" }
        require(b.size >= k * n) { "b has ${b.size} elements, needs $k × $n" }
        require(c.size >= m * n) { "c has ${c.size} elements, needs $m × $n" }
        multiplyStrided(a, 0, k, 1, b, c, m, k, n, accumulate, alpha, simd())
    }

    fun multiplyTransposedA(
        a: FloatArray,
        b: FloatArray,
        c: FloatArray,
        m: Int,
        k: Int,
        n: Int,
        accumulate: Boolean = true,
        alpha: Float = 1.0f
    ) {
        require(m >= 0 && k >= 0 && n >= 0) { "Negative dimension in [$k, $m]ᵀ × [$k, $n]" }
        require(a.size >= k * m) { "a has ${a.size} elements, needs $k × $m" }
        require(b.size >= k * n) { "b has ${b.size} elements, needs $k × $n" }
        require(c.size >= m * n) { "c has ${c.size} elements, needs $m × $n" }
        multiplyStrided(a, 0, 1, m, b, c, m, k, n, accumulate, alpha, simd())
    }

    fun multiplyTransposedB(
        a: FloatArray,
        b: FloatArray,
        c: FloatArray,
        m: Int,
        k: Int,
        n: Int,
        accumulate: Boolean = true,
        alpha: Float = 1.0f
    ) {
        require(m >= 0 && k >= 0 && n >= 0) { "Negative dimension in [$m, $k] × [$n, $k]ᵀ" }
        require(a.size >= m * k) { "a has ${a.size} elements, needs $m × $k" }
        require(b.size >= n * k) { "b has ${b.size} elements, needs $n × $k" }
        require(c.size >= m * n) { "c has ${c.size} elements, needs $m × $n" }
        if (m == 0 || n == 0) return
        if (k == 0) {
            if (!accumulate) c.fill(0.0f, 0, m * n)
            return
        }
        val simd = simd()
        if (simd != null) {
            multiplyByRows(a, b, c, m, k, n, accumulate, alpha, simd)
        } else {
            multiplyByTransposedChunks(a, b, c, m, k, n, accumulate, alpha)
        }
    }

    private fun multiplyStrided(
        a: FloatArray, aOffset: Int, rowStride: Int, colStride: Int,
        b: FloatArray, c: FloatArray,
        m: Int, k: Int, n: Int,
        accumulate: Boolean, alpha: Float, simd: VectorKernels?
    ) {
        if (m == 0 || n == 0) return
        if (m == 1) {
            forEachStripe(n, k.toLong() * n) { t0, t1 ->
                computeStripe(a, aOffset, colStride, b, c, k, n, accumulate, alpha, t0, t1)
            }
            return
        }
        val tilesN = (n + TILE_N - 1) / TILE_N
        val tileCount = ((m + TILE_M - 1) / TILE_M) * tilesN
        forEachTile(tileCount, m.toLong() * k * n) { tile ->
            computeTile(a, aOffset, rowStride, colStride, b, c, m, k, n, accumulate, alpha, tile, tilesN, simd)
        }
    }

    private fun computeStripe(
        a: FloatArray, aOffset: Int, colStride: Int,
        b: FloatArray, c: FloatArray,
        k: Int, n: Int, accumulate: Boolean, alpha: Float,
        t0: Int, t1: Int
    ) {
        val s = scratch.get()
        val bRow = s.bRow
        loadChunks(c, n, t0, t1, s.chunks, accumulate)
        for (l in 0 until k) {
            val al = alpha * a[aOffset + l * colStride]
            val rowOffset = l * n
            for (t in t0 until t1) {
                val j0 = t * TILE_N
                val len = minOf(TILE_N, n - j0)
                System.arraycopy(b, rowOffset + j0, bRow, 0, len)
                axpy(s.chunks[t - t0], bRow, al, len, null)
            }
        }
        storeChunks(c, n, t0, t1, s.chunks)
    }

    private fun loadChunks(c: FloatArray, n: Int, t0: Int, t1: Int, chunks: ArrayList<FloatArray>, accumulate: Boolean) {
        for (t in t0 until t1) {
            val j0 = t * TILE_N
            val len = minOf(TILE_N, n - j0)
            val chunk = chunkAt(chunks, t - t0)
            if (accumulate) System.arraycopy(c, j0, chunk, 0, len) else chunk.fill(0.0f, 0, len)
        }
    }

    private fun storeChunks(c: FloatArray, n: Int, t0: Int, t1: Int, chunks: ArrayList<FloatArray>) {
        for (t in t0 until t1) {
            val j0 = t * TILE_N
            System.arraycopy(chunks[t - t0], 0, c, j0, minOf(TILE_N, n - j0))
        }
    }

    private fun chunkAt(list: ArrayList<FloatArray>, index: Int): FloatArray {
        while (list.size <= index) list.add(FloatArray(TILE_N))
        return list[index]
    }

    private fun computeTile(
        a: FloatArray, aOffset: Int, rowStride: Int, colStride: Int,
        b: FloatArray, c: FloatArray,
        m: Int, k: Int, n: Int,
        accumulate: Boolean, alpha: Float,
        tile: Int, tilesN: Int, simd: VectorKernels?
    ) {
        val i0 = (tile / tilesN) * TILE_M
        val j0 = (tile % tilesN) * TILE_N
        val rows = minOf(TILE_M, m - i0)
        val len = minOf(TILE_N, n - j0)
        val s = scratch.get()
        val ct = s.rows
        val bRow = s.bRow

        loadTile(c, n, i0, j0, rows, len, ct, accumulate)

        val rowBase = aOffset + i0 * rowStride
        for (l in 0 until k) {
            System.arraycopy(b, l * n + j0, bRow, 0, len)
            accumulateRows(a, rowBase + l * colStride, rowStride, rows, bRow, ct, alpha, len, simd)
        }

        storeTile(c, n, i0, j0, rows, len, ct)
    }

    private fun loadTile(c: FloatArray, n: Int, i0: Int, j0: Int, rows: Int, len: Int, ct: Array<FloatArray>, accumulate: Boolean) {
        for (r in 0 until rows) {
            if (accumulate) {
                System.arraycopy(c, (i0 + r) * n + j0, ct[r], 0, len)
            } else {
                ct[r].fill(0.0f, 0, len)
            }
        }
    }

    private fun storeTile(c: FloatArray, n: Int, i0: Int, j0: Int, rows: Int, len: Int, ct: Array<FloatArray>) {
        for (r in 0 until rows) {
            System.arraycopy(ct[r], 0, c, (i0 + r) * n + j0, len)
        }
    }

    private fun accumulateRows(
        a: FloatArray, aIndex: Int, rowStride: Int, rows: Int,
        x: FloatArray, target: Array<FloatArray>, alpha: Float, len: Int, simd: VectorKernels?
    ) {
        var r = 0
        while (r + 4 <= rows) {
            val index = aIndex + r * rowStride
            axpy4(
                target[r], target[r + 1], target[r + 2], target[r + 3], x,
                alpha * a[index], alpha * a[index + rowStride], alpha * a[index + 2 * rowStride], alpha * a[index + 3 * rowStride],
                len, simd
            )
            r += 4
        }
        while (r < rows) {
            axpy(target[r], x, alpha * a[aIndex + r * rowStride], len, simd)
            r++
        }
    }

    private fun axpy(y: FloatArray, x: FloatArray, a: Float, len: Int, simd: VectorKernels?) {
        if (simd != null) {
            simd.axpy(y, x, a, len)
            return
        }
        for (j in 0 until len) {
            y[j] += a * x[j]
        }
    }

    private fun axpy4(
        y0: FloatArray, y1: FloatArray, y2: FloatArray, y3: FloatArray, x: FloatArray,
        a0: Float, a1: Float, a2: Float, a3: Float, len: Int, simd: VectorKernels?
    ) {
        if (simd != null) {
            simd.axpy4(y0, y1, y2, y3, x, a0, a1, a2, a3, len)
            return
        }
        for (j in 0 until len) {
            val xv = x[j]
            y0[j] += a0 * xv
            y1[j] += a1 * xv
            y2[j] += a2 * xv
            y3[j] += a3 * xv
        }
    }

    private fun multiplyByRows(
        a: FloatArray, b: FloatArray, c: FloatArray,
        m: Int, k: Int, n: Int,
        accumulate: Boolean, alpha: Float, simd: VectorKernels
    ) {
        val tilesN = (n + TILE_DOT - 1) / TILE_DOT
        val tileCount = ((m + TILE_M - 1) / TILE_M) * tilesN
        forEachTile(tileCount, m.toLong() * k * n) { tile ->
            val i0 = (tile / tilesN) * TILE_M
            val j0 = (tile % tilesN) * TILE_DOT
            val i1 = minOf(i0 + TILE_M, m)
            val j1 = minOf(j0 + TILE_DOT, n)
            if (!accumulate) {
                for (i in i0 until i1) c.fill(0.0f, i * n + j0, i * n + j1)
            }
            val dots = scratch.get().dots
            var l0 = 0
            while (l0 < k) {
                val len = minOf(CHUNK_K, k - l0)
                var j = j0
                while (j + 4 <= j1) {
                    val b0 = j * k + l0
                    for (i in i0 until i1) {
                        simd.dot4(a, i * k + l0, b, b0, b0 + k, b0 + 2 * k, b0 + 3 * k, len, dots)
                        val ci = i * n + j
                        c[ci] += alpha * dots[0]
                        c[ci + 1] += alpha * dots[1]
                        c[ci + 2] += alpha * dots[2]
                        c[ci + 3] += alpha * dots[3]
                    }
                    j += 4
                }
                while (j < j1) {
                    for (i in i0 until i1) {
                        c[i * n + j] += alpha * simd.dot(a, i * k + l0, b, j * k + l0, len)
                    }
                    j++
                }
                l0 += len
            }
        }
    }

    private fun multiplyByTransposedChunks(
        a: FloatArray, b: FloatArray, c: FloatArray,
        m: Int, k: Int, n: Int,
        accumulate: Boolean, alpha: Float
    ) {
        val chunk = (TRANSPOSE_CHUNK / n).coerceIn(16, CHUNK_K).coerceAtMost(k)
        val transposed = FloatArray(chunk * n)
        var l0 = 0
        while (l0 < k) {
            val len = minOf(chunk, k - l0)
            transposeColumns(b, k, l0, len, n, transposed)
            multiplyStrided(a, l0, k, 1, transposed, c, m, len, n, accumulate || l0 > 0, alpha, null)
            l0 += len
        }
    }

    private fun transposeColumns(b: FloatArray, k: Int, l0: Int, len: Int, n: Int, dst: FloatArray) {
        val blocks = (n + 31) / 32
        forEachTile(blocks, len.toLong() * n) { block ->
            val j0 = block * 32
            val j1 = minOf(j0 + 32, n)
            var lb0 = 0
            while (lb0 < len) {
                val lb1 = minOf(lb0 + 32, len)
                for (j in j0 until j1) {
                    val source = j * k + l0
                    for (l in lb0 until lb1) {
                        dst[l * n + j] = b[source + l]
                    }
                }
                lb0 = lb1
            }
        }
    }

    fun multiplyQuantized(
        a: FloatArray,
        weight: QuantizedTensor,
        c: FloatArray,
        m: Int,
        accumulate: Boolean = true,
        alpha: Float = 1.0f
    ) {
        val k = weight.rows
        val n = weight.columns
        require(m >= 0) { "Negative row count $m" }
        require(a.size >= m * k) { "a has ${a.size} elements, needs $m × $k" }
        require(c.size >= m * n) { "c has ${c.size} elements, needs $m × $n" }
        if (m == 0) return

        if (m == 1) {
            forEachStripe(n, k.toLong() * n) { t0, t1 ->
                computeQuantizedStripe(a, weight, c, accumulate, alpha, t0, t1)
            }
            return
        }
        val simd = simd()
        val tilesN = (n + TILE_N - 1) / TILE_N
        val tileCount = ((m + TILE_M - 1) / TILE_M) * tilesN
        forEachTile(tileCount, m.toLong() * k * n) { tile ->
            computeQuantizedTile(a, weight, c, m, accumulate, alpha, tile, tilesN, simd)
        }
    }

    private fun computeQuantizedStripe(a: FloatArray, weight: QuantizedTensor, c: FloatArray, accumulate: Boolean, alpha: Float, t0: Int, t1: Int) {
        val k = weight.rows
        val n = weight.columns
        val s = scratch.get()
        val zeros = weight.zeros
        loadChunks(c, n, t0, t1, s.chunks, accumulate)
        for (t in t0 until t1) chunkAt(s.accChunks, t - t0)

        for (group in 0 until weight.groups) {
            val k0 = group * weight.groupSize
            val k1 = minOf(k0 + weight.groupSize, k)
            for (t in t0 until t1) s.accChunks[t - t0].fill(0.0f, 0, minOf(TILE_N, n - t * TILE_N))
            var rowSum = 0.0f
            for (l in k0 until k1) {
                val al = a[l]
                rowSum += al
                for (t in t0 until t1) {
                    val j0 = t * TILE_N
                    val len = minOf(TILE_N, n - j0)
                    unpackRow(weight, l, j0, len, s)
                    axpy(s.accChunks[t - t0], s.bRow, al, len, null)
                }
            }
            for (t in t0 until t1) {
                val j0 = t * TILE_N
                val len = minOf(TILE_N, n - j0)
                val groupOffset = group * n + j0
                System.arraycopy(weight.scales, groupOffset, s.scaleRow, 0, len)
                if (zeros != null) {
                    System.arraycopy(zeros, groupOffset, s.bytes, 0, len)
                    bytesToFloats(s.bytes, s.zeroRow, len)
                    combineWithZeroPoint(s.chunks[t - t0], s.accChunks[t - t0], s.scaleRow, s.zeroRow, alpha, rowSum, len)
                } else {
                    combine(s.chunks[t - t0], s.accChunks[t - t0], s.scaleRow, alpha, len)
                }
            }
        }
        storeChunks(c, n, t0, t1, s.chunks)
    }

    private fun computeQuantizedTile(
        a: FloatArray, weight: QuantizedTensor, c: FloatArray,
        m: Int, accumulate: Boolean, alpha: Float,
        tile: Int, tilesN: Int, simd: VectorKernels?
    ) {
        val k = weight.rows
        val n = weight.columns
        val i0 = (tile / tilesN) * TILE_M
        val j0 = (tile % tilesN) * TILE_N
        val rows = minOf(TILE_M, m - i0)
        val len = minOf(TILE_N, n - j0)
        val s = scratch.get()
        val ct = s.rows
        val acc = s.groupRows
        val rowSums = s.rowSums
        val zeros = weight.zeros

        loadTile(c, n, i0, j0, rows, len, ct, accumulate)

        for (group in 0 until weight.groups) {
            val k0 = group * weight.groupSize
            val k1 = minOf(k0 + weight.groupSize, k)
            for (r in 0 until rows) {
                acc[r].fill(0.0f, 0, len)
                rowSums[r] = 0.0f
            }
            for (l in k0 until k1) {
                unpackRow(weight, l, j0, len, s)
                val aColumn = i0 * k + l
                accumulateRows(a, aColumn, k, rows, s.bRow, acc, 1.0f, len, simd)
                for (r in 0 until rows) {
                    rowSums[r] += a[aColumn + r * k]
                }
            }
            val groupOffset = group * n + j0
            System.arraycopy(weight.scales, groupOffset, s.scaleRow, 0, len)
            if (zeros != null) {
                System.arraycopy(zeros, groupOffset, s.bytes, 0, len)
                bytesToFloats(s.bytes, s.zeroRow, len)
                for (r in 0 until rows) {
                    combineWithZeroPoint(ct[r], acc[r], s.scaleRow, s.zeroRow, alpha, rowSums[r], len)
                }
            } else {
                for (r in 0 until rows) {
                    combine(ct[r], acc[r], s.scaleRow, alpha, len)
                }
            }
        }

        storeTile(c, n, i0, j0, rows, len, ct)
    }

    private fun unpackRow(weight: QuantizedTensor, row: Int, j0: Int, len: Int, s: Scratch) {
        val bytesPerRow = weight.bytesPerRow
        if (weight.bits == 8) {
            System.arraycopy(weight.qdata, row * bytesPerRow + j0, s.bytes, 0, len)
            bytesToFloats(s.bytes, s.bRow, len)
            return
        }
        val half = bytesPerRow
        val j1 = j0 + len
        if (j0 < half) {
            val count = minOf(j1, half) - j0
            System.arraycopy(weight.qdata, row * half + j0, s.bytes, 0, count)
            lowNibbles(s.bytes, s.nibbles, count)
            bytesToFloats(s.nibbles, s.bRow, count)
        }
        if (j1 > half) {
            val start = maxOf(j0, half)
            val count = j1 - start
            System.arraycopy(weight.qdata, row * half + (start - half), s.bytes, 0, count)
            highNibbles(s.bytes, s.nibbles, count)
            bytesToFloats(s.nibbles, s.unpacked, count)
            System.arraycopy(s.unpacked, 0, s.bRow, start - j0, count)
        }
    }

    private fun bytesToFloats(bytes: ByteArray, floats: FloatArray, len: Int) {
        for (j in 0 until len) {
            floats[j] = bytes[j].toFloat()
        }
    }

    private fun lowNibbles(src: ByteArray, dst: ByteArray, len: Int) {
        for (j in 0 until len) {
            dst[j] = (src[j].toInt() and 0xF).toByte()
        }
    }

    private fun highNibbles(src: ByteArray, dst: ByteArray, len: Int) {
        for (j in 0 until len) {
            dst[j] = ((src[j].toInt() shr 4) and 0xF).toByte()
        }
    }

    private fun combine(y: FloatArray, acc: FloatArray, scale: FloatArray, alpha: Float, len: Int) {
        for (j in 0 until len) {
            y[j] += alpha * scale[j] * acc[j]
        }
    }

    private fun combineWithZeroPoint(y: FloatArray, acc: FloatArray, scale: FloatArray, zero: FloatArray, alpha: Float, rowSum: Float, len: Int) {
        for (j in 0 until len) {
            y[j] += alpha * scale[j] * (acc[j] - zero[j] * rowSum)
        }
    }

    fun multiplyByQuantizedTransposed(
        dy: FloatArray,
        weight: QuantizedTensor,
        dx: FloatArray,
        m: Int,
        accumulate: Boolean = true
    ) {
        val k = weight.rows
        val n = weight.columns
        require(dy.size >= m * n) { "dy has ${dy.size} elements, needs $m × $n" }
        require(dx.size >= m * k) { "dx has ${dx.size} elements, needs $m × $k" }
        if (m == 0) return
        if (!accumulate) dx.fill(0.0f, 0, m * k)
        val simd = simd()

        val rowTiles = (k + TILE_M - 1) / TILE_M
        forEachTile(rowTiles, m.toLong() * k * n) { tile ->
            val wRow = FloatArray(n)
            val k0 = tile * TILE_M
            val k1 = minOf(k0 + TILE_M, k)
            for (l in k0 until k1) {
                weight.dequantizeRow(l, wRow)
                for (i in 0 until m) {
                    dx[i * k + l] += dot(dy, i * n, wRow, 0, n, simd)
                }
            }
        }
    }

    private fun dot(a: FloatArray, aOffset: Int, b: FloatArray, bOffset: Int, len: Int, simd: VectorKernels?): Float {
        if (simd != null) return simd.dot(a, aOffset, b, bOffset, len)
        var s0 = 0.0f
        var s1 = 0.0f
        var s2 = 0.0f
        var s3 = 0.0f
        var j = 0
        while (j + 4 <= len) {
            s0 += a[aOffset + j] * b[bOffset + j]
            s1 += a[aOffset + j + 1] * b[bOffset + j + 1]
            s2 += a[aOffset + j + 2] * b[bOffset + j + 2]
            s3 += a[aOffset + j + 3] * b[bOffset + j + 3]
            j += 4
        }
        var sum = (s0 + s1) + (s2 + s3)
        while (j < len) {
            sum += a[aOffset + j] * b[bOffset + j]
            j++
        }
        return sum
    }

    fun transpose(src: FloatArray, rows: Int, cols: Int, dst: FloatArray) {
        require(src.size >= rows * cols && dst.size >= rows * cols) { "transpose buffers too small for $rows × $cols" }
        val block = 32
        var i0 = 0
        while (i0 < rows) {
            val i1 = minOf(i0 + block, rows)
            var j0 = 0
            while (j0 < cols) {
                val j1 = minOf(j0 + block, cols)
                for (i in i0 until i1) {
                    val rowOffset = i * cols
                    for (j in j0 until j1) {
                        dst[j * rows + i] = src[rowOffset + j]
                    }
                }
                j0 = j1
            }
            i0 = i1
        }
    }
}
