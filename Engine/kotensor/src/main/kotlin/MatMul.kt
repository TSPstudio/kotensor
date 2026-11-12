package io.kotensor.core

import java.util.concurrent.ForkJoinPool
import java.util.concurrent.RecursiveAction
import java.util.concurrent.atomic.AtomicInteger
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

    private const val MAX_ROW_UNITS = 32
    private const val TARGET_UNITS = 48
    private const val UNIT_WORK = 1L shl 16
    private const val COLUMN_SPLIT_COLUMNS = 8192
    private const val MIN_BLOCK_BYTES = 128
    private const val BLOCK_ALIGN = 64
    private const val ROW_BATCH = 8

    @Volatile
    var numThreads: Int = defaultThreads()
        set(value) {
            require(value >= 1) { "numThreads must be at least 1, got $value" }
            synchronized(this) {
                if (value == field) return
                field = value
                currentPool?.shutdown()
                currentPool = null
            }
        }

    @Volatile
    var parallelThreshold: Long = 1L shl 18

    @Volatile
    var lookupInt4: Boolean = defaultLookupInt4(System.getProperty("kotensor.int4Kernel"), System.getProperty("java.vm.name"))

    @Volatile
    private var currentPool: ForkJoinPool? = null

    private val pool: ForkJoinPool
        get() {
            currentPool?.let { return it }
            synchronized(this) {
                return currentPool ?: ForkJoinPool(numThreads).also { currentPool = it }
            }
        }

    private fun defaultThreads(): Int {
        val available = Runtime.getRuntime().availableProcessors()
        val configured = System.getProperty("kotensor.threads")?.trim()?.toIntOrNull() ?: return available
        return configured.coerceIn(1, maxOf(1, available))
    }

    internal fun defaultLookupInt4(configured: String?, vmName: String?): Boolean {
        val kernel = configured?.trim()
        if (kernel.equals("lookup", ignoreCase = true)) return true
        if (kernel.equals("unpack", ignoreCase = true)) return false
        return vmName.orEmpty().contains("Dalvik", ignoreCase = true)
    }

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
        val scaleRow = FloatArray(TILE_N)
        val zeroRow = FloatArray(TILE_N)
        val rowSums = FloatArray(TILE_M)
        val dots = FloatArray(4)
        val chunks = ArrayList<FloatArray>()
    }

    private val scratch: ThreadLocal<Scratch> = object : ThreadLocal<Scratch>() {
        override fun initialValue(): Scratch = Scratch()
    }

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

    internal fun runsInParallel(count: Int, work: Long): Boolean {
        return count > 1 && work >= parallelThreshold && numThreads > 1
    }

    internal fun parallelFor(count: Int, work: Long, body: (Int) -> Unit) {
        forEachTile(count, work) { index -> body(index) }
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
            multiplyQuantizedRow(a, weight, c, accumulate, alpha)
            return
        }
        val simd = simd()
        val width = quantizedTileWidth(n)
        val tilesN = (n + width - 1) / width
        val tileCount = ((m + TILE_M - 1) / TILE_M) * tilesN
        forEachTile(tileCount, m.toLong() * k * n) { tile ->
            computeQuantizedTile(a, weight, c, m, accumulate, alpha, tile, tilesN, width, simd)
        }
    }

    private fun quantizedTileWidth(n: Int): Int {
        if (n >= TARGET_UNITS * TILE_N) return TILE_N
        val wanted = (n + TARGET_UNITS - 1) / TARGET_UNITS
        return ((wanted + BLOCK_ALIGN - 1) / BLOCK_ALIGN * BLOCK_ALIGN).coerceIn(BLOCK_ALIGN, TILE_N)
    }

    private fun multiplyQuantizedRow(a: FloatArray, weight: QuantizedTensor, c: FloatArray, accumulate: Boolean, alpha: Float) {
        val k = weight.rows
        val n = weight.columns
        val rowBytes = weight.bytesPerRow
        val targetUnits = (k.toLong() * n / UNIT_WORK).toInt().coerceIn(1, TARGET_UNITS)
        val rowsPerUnit: Int
        val wantedColumnUnits: Int
        if (n >= COLUMN_SPLIT_COLUMNS) {
            rowsPerUnit = k
            wantedColumnUnits = targetUnits
        } else {
            val groupsPerUnit = (weight.groups + MAX_ROW_UNITS - 1) / MAX_ROW_UNITS
            rowsPerUnit = minOf(k, groupsPerUnit * weight.groupSize)
            wantedColumnUnits = (targetUnits * rowsPerUnit + k - 1) / k
        }
        val rowUnits = (k + rowsPerUnit - 1) / rowsPerUnit
        val columnUnits = wantedColumnUnits.coerceIn(1, maxOf(1, rowBytes / MIN_BLOCK_BYTES))
        val blockBytes = minOf(rowBytes, ((rowBytes + columnUnits - 1) / columnUnits + BLOCK_ALIGN - 1) / BLOCK_ALIGN * BLOCK_ALIGN)
        val blocks = (rowBytes + blockBytes - 1) / blockBytes
        val partials = if (rowUnits > 1) rowScratch.get().partialBuffer(rowUnits * n) else null
        partials?.fill(0.0f, 0, rowUnits * n)
        if (!accumulate) c.fill(0.0f, 0, n)
        runUnits(rowUnits * blocks, k.toLong() * n) { unit ->
            val rowUnit = unit / blocks
            val r0 = rowUnit * rowsPerUnit
            val r1 = minOf(r0 + rowsPerUnit, k)
            val b0 = (unit % blocks) * blockBytes
            val b1 = minOf(b0 + blockBytes, rowBytes)
            if (partials == null) {
                rowUnitInto(a, weight, c, 0, r0, r1, b0, b1, alpha)
            } else {
                rowUnitInto(a, weight, partials, rowUnit * n, r0, r1, b0, b1, alpha)
            }
        }
        if (partials != null) {
            for (u in 0 until rowUnits) {
                val offset = u * n
                for (j in 0 until n) {
                    c[j] += partials[offset + j]
                }
            }
        }
    }

    private class RowScratch {
        var bytes = ByteArray(0)
        var low = Array(0) { FloatArray(TILE_N) }
        var high = Array(0) { FloatArray(TILE_N) }
        var scales = FloatArray(0)
        var zeros = ByteArray(0)
        val unpacked = Array(8) { FloatArray(TILE_N) }
        val nibbles = ByteArray(TILE_N)
        val chunk = ByteArray(TILE_N)
        val table = FloatArray(64)
        private var partials = FloatArray(0)

        fun ensure(width: Int, batchBytes: Int) {
            val chunks = (width + TILE_N - 1) / TILE_N
            if (low.size < chunks) {
                low = Array(chunks) { FloatArray(TILE_N) }
                high = Array(chunks) { FloatArray(TILE_N) }
            }
            if (scales.size < width) {
                scales = FloatArray(width)
                zeros = ByteArray(width)
            }
            if (bytes.size < batchBytes) bytes = ByteArray(batchBytes)
        }

        fun partialBuffer(size: Int): FloatArray {
            if (partials.size < size) partials = FloatArray(size)
            return partials
        }
    }

    private val rowScratch: ThreadLocal<RowScratch> = object : ThreadLocal<RowScratch>() {
        override fun initialValue(): RowScratch = RowScratch()
    }

    private class UnitWorker(val units: Int, val next: AtomicInteger, val body: (Int) -> Unit) : RecursiveAction() {
        override fun compute() {
            while (true) {
                val unit = next.getAndIncrement()
                if (unit >= units) return
                body(unit)
            }
        }
    }

    private class UnitWorkers(val workers: Int, val units: Int, val body: (Int) -> Unit) : RecursiveAction() {
        override fun compute() {
            val next = AtomicInteger()
            invokeAll(List(workers) { UnitWorker(units, next, body) })
        }
    }

    private inline fun runUnits(units: Int, work: Long, crossinline body: (Int) -> Unit) {
        val threads = numThreads
        if (units == 1 || work < parallelThreshold || threads == 1) {
            for (unit in 0 until units) body(unit)
        } else {
            pool.invoke(UnitWorkers(minOf(threads, units), units) { unit -> body(unit) })
        }
    }

    private fun rowUnitInto(a: FloatArray, weight: QuantizedTensor, out: FloatArray, outOffset: Int, r0: Int, r1: Int, b0: Int, b1: Int, alpha: Float) {
        val n = weight.columns
        val rowBytes = weight.bytesPerRow
        val groupSize = weight.groupSize
        val width = b1 - b0
        val chunks = (width + TILE_N - 1) / TILE_N
        val packed = weight.bits == 4
        val lookup = packed && lookupInt4
        val highColumns = if (packed) minOf(rowBytes + b1, n) - (rowBytes + b0) else 0
        val fullRows = width == rowBytes
        val views = weight.mappedViews()
        val heap = if (views == null) weight.qdata else null
        val s = rowScratch.get()
        s.ensure(width, ROW_BATCH * width)
        val low = s.low
        val high = s.high
        var row = r0
        while (row < r1) {
            val group = row / groupSize
            val groupEnd = minOf(r1, (group + 1) * groupSize)
            for (chunk in 0 until chunks) {
                low[chunk].fill(0.0f)
                if (packed) high[chunk].fill(0.0f)
            }
            var rowSum = 0.0f
            var l = row
            while (l < groupEnd) {
                val batch = minOf(ROW_BATCH, groupEnd - l)
                val q: ByteArray
                val base: Int
                val stride: Int
                if (heap != null) {
                    q = heap
                    base = l * rowBytes + b0
                    stride = rowBytes
                } else if (fullRows) {
                    views!!.readQData(l * rowBytes, s.bytes, 0, batch * rowBytes)
                    q = s.bytes
                    base = 0
                    stride = rowBytes
                } else {
                    for (i in 0 until batch) views!!.readQData((l + i) * rowBytes + b0, s.bytes, i * width, width)
                    q = s.bytes
                    base = 0
                    stride = width
                }
                var i = 0
                while (i + 4 <= batch) {
                    val a0 = a[l + i]
                    val a1 = a[l + i + 1]
                    val a2 = a[l + i + 2]
                    val a3 = a[l + i + 3]
                    rowSum += a0
                    rowSum += a1
                    rowSum += a2
                    rowSum += a3
                    val offset = base + i * stride
                    if (lookup) {
                        lookup4(q, offset, stride, width, a0, a1, a2, a3, s)
                    } else {
                        unpack4(q, offset, stride, width, packed, a0, a1, a2, a3, s)
                    }
                    i += 4
                }
                while (i < batch) {
                    val al = a[l + i]
                    rowSum += al
                    val offset = base + i * stride
                    if (lookup) {
                        lookup1(q, offset, width, al, s)
                    } else {
                        unpack1(q, offset, width, packed, al, s)
                    }
                    i++
                }
                l += batch
            }
            combineGroup(weight, views, group * n + b0, width, rowSum, alpha, low, out, outOffset + b0, s)
            if (highColumns > 0) {
                combineGroup(weight, views, group * n + rowBytes + b0, highColumns, rowSum, alpha, high, out, outOffset + rowBytes + b0, s)
            }
            row = groupEnd
        }
    }

    private fun lookup4(q: ByteArray, offset: Int, stride: Int, width: Int, a0: Float, a1: Float, a2: Float, a3: Float, s: RowScratch) {
        val table = s.table
        for (v in 0 until 16) {
            val f = v.toFloat()
            table[v] = a0 * f
            table[16 + v] = a1 * f
            table[32 + v] = a2 * f
            table[48 + v] = a3 * f
        }
        var c0 = 0
        var chunk = 0
        while (c0 < width) {
            val len = minOf(TILE_N, width - c0)
            val low = s.low[chunk]
            val high = s.high[chunk]
            val o0 = offset + c0
            val o1 = o0 + stride
            val o2 = o1 + stride
            val o3 = o2 + stride
            for (c in 0 until len) {
                val q0 = q[o0 + c].toInt()
                val q1 = q[o1 + c].toInt()
                val q2 = q[o2 + c].toInt()
                val q3 = q[o3 + c].toInt()
                var lo = low[c]
                lo += table[q0 and 15]
                lo += table[16 + (q1 and 15)]
                lo += table[32 + (q2 and 15)]
                lo += table[48 + (q3 and 15)]
                low[c] = lo
                var hi = high[c]
                hi += table[(q0 shr 4) and 15]
                hi += table[16 + ((q1 shr 4) and 15)]
                hi += table[32 + ((q2 shr 4) and 15)]
                hi += table[48 + ((q3 shr 4) and 15)]
                high[c] = hi
            }
            c0 += len
            chunk++
        }
    }

    private fun lookup1(q: ByteArray, offset: Int, width: Int, a0: Float, s: RowScratch) {
        val table = s.table
        for (v in 0 until 16) {
            table[v] = a0 * v.toFloat()
        }
        var c0 = 0
        var chunk = 0
        while (c0 < width) {
            val len = minOf(TILE_N, width - c0)
            val low = s.low[chunk]
            val high = s.high[chunk]
            val o = offset + c0
            for (c in 0 until len) {
                val q0 = q[o + c].toInt()
                low[c] += table[q0 and 15]
                high[c] += table[(q0 shr 4) and 15]
            }
            c0 += len
            chunk++
        }
    }

    private fun unpack4(q: ByteArray, offset: Int, stride: Int, width: Int, packed: Boolean, a0: Float, a1: Float, a2: Float, a3: Float, s: RowScratch) {
        val x = s.unpacked
        val nib = s.nibbles
        val bytes = s.chunk
        var c0 = 0
        var chunk = 0
        while (c0 < width) {
            val len = minOf(TILE_N, width - c0)
            for (r in 0 until 4) {
                System.arraycopy(q, offset + r * stride + c0, bytes, 0, len)
                if (packed) {
                    lowNibbles(bytes, nib, len)
                    bytesToFloats(nib, x[r], 0, len)
                } else {
                    bytesToFloats(bytes, x[r], 0, len)
                }
            }
            accumulate4(s.low[chunk], x[0], x[1], x[2], x[3], a0, a1, a2, a3, len)
            if (packed) {
                for (r in 0 until 4) {
                    System.arraycopy(q, offset + r * stride + c0, bytes, 0, len)
                    highNibbles(bytes, nib, len)
                    bytesToFloats(nib, x[r], 0, len)
                }
                accumulate4(s.high[chunk], x[0], x[1], x[2], x[3], a0, a1, a2, a3, len)
            }
            c0 += len
            chunk++
        }
    }

    private fun unpack1(q: ByteArray, offset: Int, width: Int, packed: Boolean, a0: Float, s: RowScratch) {
        val x = s.unpacked[0]
        val nib = s.nibbles
        val bytes = s.chunk
        var c0 = 0
        var chunk = 0
        while (c0 < width) {
            val len = minOf(TILE_N, width - c0)
            System.arraycopy(q, offset + c0, bytes, 0, len)
            if (packed) {
                lowNibbles(bytes, nib, len)
                bytesToFloats(nib, x, 0, len)
                axpy(s.low[chunk], x, a0, len, null)
                highNibbles(bytes, nib, len)
                bytesToFloats(nib, x, 0, len)
                axpy(s.high[chunk], x, a0, len, null)
            } else {
                bytesToFloats(bytes, x, 0, len)
                axpy(s.low[chunk], x, a0, len, null)
            }
            c0 += len
            chunk++
        }
    }

    private fun combineGroup(weight: QuantizedTensor, views: MappedViews?, scaleOffset: Int, count: Int, rowSum: Float, alpha: Float, acc: Array<FloatArray>, out: FloatArray, outStart: Int, s: RowScratch) {
        val scales = s.scales
        weight.readScales(views, scaleOffset, scales, 0, count)
        val zeros = s.zeros
        val hasZeros = weight.hasZeroPoints
        if (hasZeros) weight.readZeros(views, scaleOffset, zeros, 0, count)
        var c0 = 0
        var chunk = 0
        while (c0 < count) {
            val len = minOf(TILE_N, count - c0)
            val values = acc[chunk]
            val o = outStart + c0
            if (hasZeros) {
                for (c in 0 until len) {
                    out[o + c] += alpha * scales[c0 + c] * (values[c] - zeros[c0 + c].toFloat() * rowSum)
                }
            } else {
                for (c in 0 until len) {
                    out[o + c] += alpha * scales[c0 + c] * values[c]
                }
            }
            c0 += len
            chunk++
        }
    }

    private fun computeQuantizedTile(
        a: FloatArray, weight: QuantizedTensor, c: FloatArray,
        m: Int, accumulate: Boolean, alpha: Float,
        tile: Int, tilesN: Int, width: Int, simd: VectorKernels?
    ) {
        val k = weight.rows
        val n = weight.columns
        val i0 = (tile / tilesN) * TILE_M
        val j0 = (tile % tilesN) * width
        val rows = minOf(TILE_M, m - i0)
        val len = minOf(width, n - j0)
        val s = scratch.get()
        val ct = s.rows
        val acc = s.groupRows
        val rowSums = s.rowSums
        val views = weight.mappedViews()
        val hasZeros = weight.hasZeroPoints

        loadTile(c, n, i0, j0, rows, len, ct, accumulate)

        for (group in 0 until weight.groups) {
            val k0 = group * weight.groupSize
            val k1 = minOf(k0 + weight.groupSize, k)
            for (r in 0 until rows) {
                acc[r].fill(0.0f, 0, len)
                rowSums[r] = 0.0f
            }
            for (l in k0 until k1) {
                unpackRow(weight, views, l, j0, len, s, s.bRow)
                val aColumn = i0 * k + l
                accumulateRows(a, aColumn, k, rows, s.bRow, acc, 1.0f, len, simd)
                for (r in 0 until rows) {
                    rowSums[r] += a[aColumn + r * k]
                }
            }
            val groupOffset = group * n + j0
            weight.readScales(views, groupOffset, s.scaleRow, 0, len)
            if (hasZeros) {
                weight.readZeros(views, groupOffset, s.bytes, 0, len)
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

    private fun unpackRow(weight: QuantizedTensor, views: MappedViews?, row: Int, j0: Int, len: Int, s: Scratch, out: FloatArray) {
        val bytesPerRow = weight.bytesPerRow
        if (weight.bits == 8) {
            weight.readQData(views, row * bytesPerRow + j0, s.bytes, 0, len)
            bytesToFloats(s.bytes, out, 0, len)
            return
        }
        val half = bytesPerRow
        val j1 = j0 + len
        if (j0 < half) {
            val count = minOf(j1, half) - j0
            weight.readQData(views, row * half + j0, s.bytes, 0, count)
            lowNibbles(s.bytes, s.nibbles, count)
            bytesToFloats(s.nibbles, out, 0, count)
        }
        if (j1 > half) {
            val start = maxOf(j0, half)
            val count = j1 - start
            weight.readQData(views, row * half + (start - half), s.bytes, 0, count)
            highNibbles(s.bytes, s.nibbles, count)
            bytesToFloats(s.nibbles, out, start - j0, count)
        }
    }

    private fun accumulate4(acc: FloatArray, x0: FloatArray, x1: FloatArray, x2: FloatArray, x3: FloatArray, a0: Float, a1: Float, a2: Float, a3: Float, len: Int) {
        for (j in 0 until len) {
            var v = acc[j]
            v += a0 * x0[j]
            v += a1 * x1[j]
            v += a2 * x2[j]
            v += a3 * x3[j]
            acc[j] = v
        }
    }

    private fun bytesToFloats(bytes: ByteArray, floats: FloatArray, len: Int) {
        for (j in 0 until len) {
            floats[j] = bytes[j].toFloat()
        }
    }

    private fun bytesToFloats(bytes: ByteArray, floats: FloatArray, offset: Int, len: Int) {
        for (j in 0 until len) {
            floats[offset + j] = bytes[j].toFloat()
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
