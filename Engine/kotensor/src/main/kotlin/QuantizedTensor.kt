package io.kotensor.core

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

class QuantizedTensor(
    val qdata: ByteArray,
    val scales: FloatArray,
    val zeros: ByteArray?,
    val shape: IntArray,
    val bits: Int,
    val groupSize: Int
) {
    val rows = shape[0]
    val columns = shape[1]
    val groups = (rows + groupSize - 1) / groupSize
    val storedColumns = if (bits == 4) columns + columns % 2 else columns
    val bytesPerRow = if (bits == 4) storedColumns / 2 else storedColumns

    init {
        require(shape.size == 2 && rows > 0 && columns > 0) { "QuantizedTensor needs a 2-D shape, got ${shape.contentToString()}" }
        require(bits == 4 || bits == 8) { "bits must be 4 or 8" }
        require(qdata.size == rows * bytesPerRow) { "qdata has ${qdata.size} bytes, expected ${rows * bytesPerRow}" }
        require(scales.size == groups * columns) { "scales has ${scales.size} entries, expected ${groups * columns}" }
        require(zeros == null || zeros.size == groups * columns) { "zeros has ${zeros?.size} entries, expected ${groups * columns}" }
        require(bits == 8 || zeros != null) { "4-bit tensors need zero points" }
    }

    val sizeInBytes: Long
        get() = qdata.size.toLong() + scales.size.toLong() * 4L + (zeros?.size?.toLong() ?: 0L)

    fun rawValue(row: Int, column: Int): Int {
        return if (bits == 8) {
            qdata[row * bytesPerRow + column].toInt()
        } else {
            val half = bytesPerRow
            val byte = qdata[row * bytesPerRow + column % half].toInt()
            if (column < half) byte and 0xF else (byte shr 4) and 0xF
        }
    }

    fun value(row: Int, column: Int): Float {
        val group = row / groupSize
        val scale = scales[group * columns + column]
        val zero = zeros?.get(group * columns + column)?.toInt() ?: 0
        return scale * (rawValue(row, column) - zero)
    }

    fun dequantizeRow(row: Int, dest: FloatArray, destOffset: Int = 0) {
        require(destOffset + columns <= dest.size) { "dest too small for a row of $columns" }
        val groupOffset = (row / groupSize) * columns
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

    fun dequantize(): Tensor {
        val data = FloatArray(rows * columns)
        for (r in 0 until rows) {
            dequantizeRow(r, data, r * columns)
        }
        return Tensor(data, shape.copyOf())
    }

    companion object {
        fun empty(rows: Int, columns: Int, quantization: Quantization): QuantizedTensor {
            val groups = (rows + quantization.groupSize - 1) / quantization.groupSize
            val storedColumns = if (quantization.bits == 4) columns + columns % 2 else columns
            val bytesPerRow = if (quantization.bits == 4) storedColumns / 2 else storedColumns
            return QuantizedTensor(
                qdata = ByteArray(rows * bytesPerRow),
                scales = FloatArray(groups * columns),
                zeros = if (quantization.bits == 4) ByteArray(groups * columns) else null,
                shape = intArrayOf(rows, columns),
                bits = quantization.bits,
                groupSize = quantization.groupSize
            )
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
            for (r in rowStart until rowEnd) {
                val q = (data[r * columns + column] * inverse).roundToInt().coerceIn(-127, 127)
                result.qdata[r * result.bytesPerRow + column] = q.toByte()
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
            for (r in rowStart until rowEnd) {
                val q = ((data[r * columns + column] * inverse).roundToInt() + zero).coerceIn(0, 15)
                val index = r * half + column % half
                val existing = result.qdata[index].toInt()
                result.qdata[index] = if (column < half) {
                    ((existing and 0xF0) or q).toByte()
                } else {
                    ((existing and 0x0F) or (q shl 4)).toByte()
                }
            }
        }
    }
}

