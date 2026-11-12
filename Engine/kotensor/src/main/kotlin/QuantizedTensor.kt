package io.kotensor.core

import java.nio.Buffer
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.abs
import kotlin.math.roundToInt

class Quantization(
    val bits: Int,
    val groupSize: Int = 64
) {
    init {
        require(bits == 4 || bits == 8) { "bits must be 4 or 8, got $bits" }
        require(groupSize > 0) { "groupSize must be positive" }
    }

    override fun toString(): String = "int$bits/g$groupSize"

    companion object {
        val INT8 = Quantization(8)
        val INT4 = Quantization(4)
    }
}

class QuantizedTensor private constructor(
    qdata: ByteArray?,
    scales: FloatArray?,
    zeros: ByteArray?,
    val shape: IntArray,
    val bits: Int,
    val groupSize: Int,
    internal val hasZeroPoints: Boolean
) {
    constructor(
        qdata: ByteArray,
        scales: FloatArray,
        zeros: ByteArray?,
        shape: IntArray,
        bits: Int,
        groupSize: Int
    ) : this(qdata, scales, zeros, shape, bits, groupSize, zeros != null)

    val rows = shape[0]
    val columns = shape[1]
    val groups = (rows + groupSize - 1) / groupSize
    val storedColumns = if (bits == 4) columns + columns % 2 else columns
    val bytesPerRow = if (bits == 4) storedColumns / 2 else storedColumns

    @Volatile
    private var heapQData: ByteArray? = qdata

    @Volatile
    private var heapScales: FloatArray? = scales

    @Volatile
    private var heapZeros: ByteArray? = zeros

    @Volatile
    private var mapped: MappedQuantizedData? = null

    init {
        require(shape.size == 2 && rows > 0 && columns > 0) { "QuantizedTensor needs a 2-D shape, got ${shape.contentToString()}" }
        require(bits == 4 || bits == 8) { "bits must be 4 or 8" }
        require(qdata == null || qdata.size == rows * bytesPerRow) { "qdata has ${qdata?.size} bytes, expected ${rows * bytesPerRow}" }
        require(scales == null || scales.size == groups * columns) { "scales has ${scales?.size} entries, expected ${groups * columns}" }
        require(zeros == null || zeros.size == groups * columns) { "zeros has ${zeros?.size} entries, expected ${groups * columns}" }
        require(bits == 8 || hasZeroPoints) { "4-bit tensors need zero points" }
    }

    val qdata: ByteArray
        get() {
            heapQData?.let { return it }
            moveToHeap()
            return heapQData!!
        }

    val scales: FloatArray
        get() {
            heapScales?.let { return it }
            moveToHeap()
            return heapScales!!
        }

    val zeros: ByteArray?
        get() {
            if (!hasZeroPoints) return null
            heapZeros?.let { return it }
            moveToHeap()
            return heapZeros
        }

    val isMemoryMapped: Boolean
        get() = mapped != null

    val sizeInBytes: Long
        get() = rows.toLong() * bytesPerRow + groups.toLong() * columns * 4L + (if (hasZeroPoints) groups.toLong() * columns else 0L)

    @Synchronized
    private fun moveToHeap() {
        val source = mapped
        if (heapQData == null) heapQData = ByteArray(rows * bytesPerRow).also { array -> source?.copyQData(array) }
        if (heapScales == null) heapScales = FloatArray(groups * columns).also { array -> source?.copyScales(array) }
        if (hasZeroPoints && heapZeros == null) heapZeros = ByteArray(groups * columns).also { array -> source?.copyZeros(array) }
        mapped = null
    }

    internal fun mapFrom(reader: SafeTensorsFile, name: String) {
        val qweight = reader.entry("$name.qweight")
        require(qweight.byteLength == rows.toLong() * bytesPerRow) { "$name.qweight has ${qweight.byteLength} bytes, expected ${rows.toLong() * bytesPerRow}" }
        val scaleEntry = reader.entry("$name.scales")
        require(scaleEntry.dtype == "F32" && scaleEntry.byteLength == groups.toLong() * columns * 4L) {
            "$name.scales is ${scaleEntry.dtype} with ${scaleEntry.byteLength} bytes, expected F32 with ${groups.toLong() * columns * 4L}"
        }
        val zeroEntry = if (hasZeroPoints) reader.entry("$name.zeros") else null
        require(zeroEntry == null || zeroEntry.byteLength == groups.toLong() * columns) { "$name.zeros has ${zeroEntry?.byteLength} bytes, expected ${groups.toLong() * columns}" }
        val data = MappedQuantizedData(reader.map(qweight.name), reader.map(scaleEntry.name), zeroEntry?.let { entry -> reader.map(entry.name) })
        synchronized(this) {
            heapQData = null
            heapScales = null
            heapZeros = null
            mapped = data
        }
    }

    internal fun mappedViews(): MappedViews? = mapped?.views()

    internal fun readQData(views: MappedViews?, offset: Int, dest: ByteArray, destOffset: Int, length: Int) {
        if (views != null) views.readQData(offset, dest, destOffset, length) else System.arraycopy(qdata, offset, dest, destOffset, length)
    }

    internal fun readScales(views: MappedViews?, offset: Int, dest: FloatArray, destOffset: Int, length: Int) {
        if (views != null) views.readScales(offset, dest, destOffset, length) else System.arraycopy(scales, offset, dest, destOffset, length)
    }

    internal fun readZeros(views: MappedViews?, offset: Int, dest: ByteArray, destOffset: Int, length: Int) {
        if (views != null) views.readZeros(offset, dest, destOffset, length) else System.arraycopy(zeros!!, offset, dest, destOffset, length)
    }

    internal fun qdataSnapshot(): ByteArray {
        val source = mapped ?: return qdata
        return ByteArray(rows * bytesPerRow).also { array -> source.copyQData(array) }
    }

    internal fun scalesSnapshot(): FloatArray {
        val source = mapped ?: return scales
        return FloatArray(groups * columns).also { array -> source.copyScales(array) }
    }

    internal fun zerosSnapshot(): ByteArray? {
        if (!hasZeroPoints) return null
        val source = mapped ?: return zeros
        return ByteArray(groups * columns).also { array -> source.copyZeros(array) }
    }

    private fun storedByte(index: Int): Int {
        val source = mapped
        return if (source != null) source.qdata.get(index).toInt() else qdata[index].toInt()
    }

    private fun scaleAt(index: Int): Float {
        val source = mapped
        return if (source != null) source.scales.get(index) else scales[index]
    }

    private fun zeroAt(index: Int): Int {
        val source = mapped
        return if (source != null) source.zeros!!.get(index).toInt() else zeros!![index].toInt()
    }

    fun rawValue(row: Int, column: Int): Int {
        return if (bits == 8) {
            storedByte(row * bytesPerRow + column)
        } else {
            val half = bytesPerRow
            val byte = storedByte(row * bytesPerRow + column % half)
            if (column < half) byte and 0xF else (byte shr 4) and 0xF
        }
    }

    fun value(row: Int, column: Int): Float {
        val index = (row / groupSize) * columns + column
        val zero = if (hasZeroPoints) zeroAt(index) else 0
        return scaleAt(index) * (rawValue(row, column) - zero)
    }

    fun dequantizeRow(row: Int, dest: FloatArray, destOffset: Int = 0) {
        require(destOffset + columns <= dest.size) { "dest too small for a row of $columns" }
        val groupOffset = (row / groupSize) * columns
        val source = mapped
        if (source != null) {
            dequantizeMappedRow(source, row, groupOffset, dest, destOffset)
            return
        }
        val qdata = qdata
        val scales = scales
        if (bits == 8) {
            val rowOffset = row * bytesPerRow
            for (c in 0 until columns) {
                dest[destOffset + c] = scales[groupOffset + c] * qdata[rowOffset + c]
            }
        } else {
            val half = bytesPerRow
            val rowOffset = row * half
            val zeroPoints = zeros!!
            for (c in 0 until columns) {
                val byte = qdata[rowOffset + c % half].toInt()
                val raw = if (c < half) byte and 0xF else (byte shr 4) and 0xF
                dest[destOffset + c] = scales[groupOffset + c] * (raw - zeroPoints[groupOffset + c])
            }
        }
    }

    private fun dequantizeMappedRow(source: MappedQuantizedData, row: Int, groupOffset: Int, dest: FloatArray, destOffset: Int) {
        val qdata = source.qdata
        val scales = source.scales
        if (bits == 8) {
            val rowOffset = row * bytesPerRow
            for (c in 0 until columns) {
                dest[destOffset + c] = scales.get(groupOffset + c) * qdata.get(rowOffset + c)
            }
        } else {
            val half = bytesPerRow
            val rowOffset = row * half
            val zeroPoints = source.zeros!!
            for (c in 0 until columns) {
                val byte = qdata.get(rowOffset + c % half).toInt()
                val raw = if (c < half) byte and 0xF else (byte shr 4) and 0xF
                dest[destOffset + c] = scales.get(groupOffset + c) * (raw - zeroPoints.get(groupOffset + c))
            }
        }
    }

    fun dequantizeColumn(column: Int, dest: FloatArray, destOffset: Int = 0) {
        require(column in 0 until columns) { "column $column out of range for $columns columns" }
        require(destOffset >= 0 && destOffset + rows <= dest.size) { "dest too small for a column of $rows" }
        for (r in 0 until rows) {
            dest[destOffset + r] = value(r, column)
        }
    }

    fun dequantize(): Tensor {
        val data = FloatArray(rows * columns)
        for (r in 0 until rows) {
            dequantizeRow(r, data, r * columns)
        }
        return Tensor(data, shape.copyOf())
    }

    companion object {
        fun empty(rows: Int, columns: Int, quantization: Quantization): QuantizedTensor {
            return QuantizedTensor(null, null, null, intArrayOf(rows, columns), quantization.bits, quantization.groupSize, quantization.bits == 4)
        }

        fun quantize(tensor: Tensor, quantization: Quantization): QuantizedTensor {
            require(tensor.shape.size == 2) { "Only 2-D tensors can be quantized, got ${tensor.shape.contentToString()}" }
            return quantize(tensor.data, tensor.shape[0], tensor.shape[1], quantization)
        }

        fun quantize(data: FloatArray, rows: Int, columns: Int, quantization: Quantization): QuantizedTensor {
            require(data.size == rows * columns) { "data has ${data.size} elements, expected $rows × $columns" }
            val result = empty(rows, columns, quantization)
            val groupSize = quantization.groupSize
            for (group in 0 until result.groups) {
                val rowStart = group * groupSize
                val rowEnd = minOf(rowStart + groupSize, rows)
                val groupOffset = group * columns
                for (c in 0 until columns) {
                    if (quantization.bits == 8) {
                        quantizeSymmetric(data, rows, columns, rowStart, rowEnd, c, result, groupOffset + c)
                    } else {
                        quantizeAsymmetric(data, columns, rowStart, rowEnd, c, result, groupOffset + c)
                    }
                }
            }
            return result
        }

        private fun quantizeSymmetric(data: FloatArray, rows: Int, columns: Int, rowStart: Int, rowEnd: Int, column: Int, result: QuantizedTensor, scaleIndex: Int) {
            var maxAbs = 0.0f
            for (r in rowStart until rowEnd) {
                val v = abs(data[r * columns + column])
                if (v > maxAbs) maxAbs = v
            }
            val scale = if (maxAbs > 0.0f) maxAbs / 127.0f else 1.0f
            result.scales[scaleIndex] = scale
            val inverse = 1.0f / scale
            val qdata = result.qdata
            for (r in rowStart until rowEnd) {
                val q = (data[r * columns + column] * inverse).roundToInt().coerceIn(-127, 127)
                qdata[r * result.bytesPerRow + column] = q.toByte()
            }
        }

        private fun quantizeAsymmetric(data: FloatArray, columns: Int, rowStart: Int, rowEnd: Int, column: Int, result: QuantizedTensor, scaleIndex: Int) {
            var min = Float.MAX_VALUE
            var max = -Float.MAX_VALUE
            for (r in rowStart until rowEnd) {
                val v = data[r * columns + column]
                if (v < min) min = v
                if (v > max) max = v
            }
            if (min > 0.0f) min = 0.0f
            if (max < 0.0f) max = 0.0f
            val range = max - min
            val scale = if (range > 0.0f) range / 15.0f else 1.0f
            val zero = (-min / scale).roundToInt().coerceIn(0, 15)
            result.scales[scaleIndex] = scale
            result.zeros!![scaleIndex] = zero.toByte()
            val inverse = 1.0f / scale
            val half = result.bytesPerRow
            val qdata = result.qdata
            for (r in rowStart until rowEnd) {
                val q = ((data[r * columns + column] * inverse).roundToInt() + zero).coerceIn(0, 15)
                val index = r * half + column % half
                val existing = qdata[index].toInt()
                qdata[index] = if (column < half) {
                    ((existing and 0xF0) or q).toByte()
                } else {
                    ((existing and 0x0F) or (q shl 4)).toByte()
                }
            }
        }
    }
}

internal class MappedQuantizedData(
    val qdata: ByteBuffer,
    scales: ByteBuffer,
    val zeros: ByteBuffer?
) {
    val scales: FloatBuffer = scales.order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()

    private val views = object : ThreadLocal<MappedViews>() {
        override fun initialValue(): MappedViews = MappedViews(qdata.duplicate(), this@MappedQuantizedData.scales.duplicate(), zeros?.duplicate())
    }

    fun views(): MappedViews = views.get()

    fun copyQData(dest: ByteArray) = views().readQData(0, dest, 0, dest.size)

    fun copyScales(dest: FloatArray) = views().readScales(0, dest, 0, dest.size)

    fun copyZeros(dest: ByteArray) = views().readZeros(0, dest, 0, dest.size)
}

internal class MappedViews(
    private val qdata: ByteBuffer,
    private val scales: FloatBuffer,
    private val zeros: ByteBuffer?
) {
    fun readQData(offset: Int, dest: ByteArray, destOffset: Int, length: Int) {
        (qdata as Buffer).position(offset)
        qdata.get(dest, destOffset, length)
    }

    fun readScales(offset: Int, dest: FloatArray, destOffset: Int, length: Int) {
        (scales as Buffer).position(offset)
        scales.get(dest, destOffset, length)
    }

    fun readZeros(offset: Int, dest: ByteArray, destOffset: Int, length: Int) {
        val buffer = zeros!!
        (buffer as Buffer).position(offset)
        buffer.get(dest, destOffset, length)
    }
}
