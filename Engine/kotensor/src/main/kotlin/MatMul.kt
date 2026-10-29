package io.kotensor.core

import java.util.concurrent.ForkJoinPool
import java.util.concurrent.RecursiveAction

object MatMul {
    const val TILE_M = 32
    const val TILE_N = 256

    val numThreads: Int = Runtime.getRuntime().availableProcessors()

    @Volatile
    var parallelThreshold: Long = 1L shl 18

    private val pool: ForkJoinPool by lazy { ForkJoinPool(numThreads) }

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
        if (m == 0 || n == 0) return

        if (m == 1) {
            forEachStripe(n, k.toLong() * n) { t0, t1 ->
                computeStripe(a, b, c, k, n, accumulate, alpha, t0, t1)
            }
            return
        }
        val tilesN = (n + TILE_N - 1) / TILE_N
        val tileCount = ((m + TILE_M - 1) / TILE_M) * tilesN
        forEachTile(tileCount, m.toLong() * k * n) { tile ->
            computeTile(a, b, c, m, k, n, accumulate, alpha, tile, tilesN)
        }
    }

    private fun computeStripe(a: FloatArray, b: FloatArray, c: FloatArray, k: Int, n: Int, accumulate: Boolean, alpha: Float, t0: Int, t1: Int) {
        val s = scratch.get()
        val bRow = s.bRow
        loadChunks(c, n, t0, t1, s.chunks, accumulate)
        for (l in 0 until k) {
            val al = alpha * a[l]
            val rowOffset = l * n
            for (t in t0 until t1) {
                val j0 = t * TILE_N
                val len = minOf(TILE_N, n - j0)
                System.arraycopy(b, rowOffset + j0, bRow, 0, len)
                axpy(s.chunks[t - t0], bRow, al, len)
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
        a: FloatArray, b: FloatArray, c: FloatArray,
        m: Int, k: Int, n: Int,
        accumulate: Boolean, alpha: Float,
        tile: Int, tilesN: Int
    ) {
        val i0 = (tile / tilesN) * TILE_M
        val j0 = (tile % tilesN) * TILE_N
        val rows = minOf(TILE_M, m - i0)
        val len = minOf(TILE_N, n - j0)
        val s = scratch.get()
        val ct = s.rows
        val bRow = s.bRow

        loadTile(c, n, i0, j0, rows, len, ct, accumulate)

        for (l in 0 until k) {
            System.arraycopy(b, l * n + j0, bRow, 0, len)
            accumulateRows(a, k, i0 * k + l, rows, bRow, ct, alpha, len)
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

    private fun accumulateRows(a: FloatArray, k: Int, aColumn: Int, rows: Int, x: FloatArray, target: Array<FloatArray>, alpha: Float, len: Int) {
        var r = 0
        while (r + 4 <= rows) {
            val aIndex = aColumn + r * k
            axpy4(
                target[r], target[r + 1], target[r + 2], target[r + 3], x,
                alpha * a[aIndex], alpha * a[aIndex + k], alpha * a[aIndex + 2 * k], alpha * a[aIndex + 3 * k],
                len
            )
            r += 4
        }
        while (r < rows) {
            axpy(target[r], x, alpha * a[aColumn + r * k], len)
            r++
        }
    }

    private fun axpy(y: FloatArray, x: FloatArray, a: Float, len: Int) {
        for (j in 0 until len) {
            y[j] += a * x[j]
        }
    }

    private fun axpy4(
        y0: FloatArray, y1: FloatArray, y2: FloatArray, y3: FloatArray, x: FloatArray,
        a0: Float, a1: Float, a2: Float, a3: Float, len: Int
    ) {
        for (j in 0 until len) {
            val xv = x[j]
            y0[j] += a0 * xv
            y1[j] += a1 * xv
            y2[j] += a2 * xv
            y3[j] += a3 * xv
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
        val tilesN = (n + TILE_N - 1) / TILE_N
        val tileCount = ((m + TILE_M - 1) / TILE_M) * tilesN
        forEachTile(tileCount, m.toLong() * k * n) { tile ->
            computeQuantizedTile(a, weight, c, m, accumulate, alpha, tile, tilesN)
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
                    axpy(s.accChunks[t - t0], s.bRow, al, len)
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
        tile: Int, tilesN: Int
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
                accumulateRows(a, k, aColumn, rows, s.bRow, acc, 1.0f, len)
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

        val rowTiles = (k + TILE_M - 1) / TILE_M
        forEachTile(rowTiles, m.toLong() * k * n) { tile ->
            val wRow = FloatArray(n)
            val k0 = tile * TILE_M
            val k1 = minOf(k0 + TILE_M, k)
            for (l in k0 until k1) {
                weight.dequantizeRow(l, wRow)
                for (i in 0 until m) {
                    dx[i * k + l] += dot(dy, i * n, wRow, n)
                }
            }
        }
    }

    private fun dot(a: FloatArray, aOffset: Int, b: FloatArray, len: Int): Float {
        var s0 = 0.0f
        var s1 = 0.0f
        var s2 = 0.0f
        var s3 = 0.0f
        var j = 0
        while (j + 4 <= len) {
            s0 += a[aOffset + j] * b[j]
            s1 += a[aOffset + j + 1] * b[j + 1]
            s2 += a[aOffset + j + 2] * b[j + 2]
            s3 += a[aOffset + j + 3] * b[j + 3]
            j += 4
        }
        var sum = (s0 + s1) + (s2 + s3)
        while (j < len) {
            sum += a[aOffset + j] * b[j]
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

