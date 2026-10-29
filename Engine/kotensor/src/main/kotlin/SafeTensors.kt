package io.kotensor.core

import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

object SafeTensors {

    fun save(file: File, tensors: Map<String, Tensor>) {
        var currentOffset = 0L
        val headerEntries = mutableListOf<String>()

        for ((name, tensor) in tensors) {
            val byteCount = tensor.data.size.toLong() * 4L
            val startOffset = currentOffset
            val endOffset = currentOffset + byteCount
            val shapeString = tensor.shape.joinToString(",")
            val entry = "\"$name\":{\"dtype\":\"F32\",\"shape\":[$shapeString],\"data_offsets\":[$startOffset,$endOffset]}"
            headerEntries.add(entry)
            currentOffset = endOffset
        }

        val jsonString = "{" + headerEntries.joinToString(",") + "}"
        val headerBytes = jsonString.toByteArray(Charsets.UTF_8)
        val headerSize = headerBytes.size.toLong()

        FileOutputStream(file).use { out ->
            val sizeBuffer = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN)
            sizeBuffer.putLong(headerSize)
            out.write(sizeBuffer.array())
            out.write(headerBytes)

            for ((_, tensor) in tensors) {
                val floatBuffer = ByteBuffer.allocate(tensor.data.size * 4).order(ByteOrder.LITTLE_ENDIAN)
                for (v in tensor.data) {
                    floatBuffer.putFloat(v)
                }
                out.write(floatBuffer.array())
            }
        }
    }

    fun load(file: File): Map<String, Tensor> {
        val fileBytes = file.readBytes()
        val buffer = ByteBuffer.wrap(fileBytes).order(ByteOrder.LITTLE_ENDIAN)
        val headerSize = buffer.long.toInt()

        val headerBytes = ByteArray(headerSize)
        buffer.get(headerBytes)
        val headerJson = String(headerBytes, Charsets.UTF_8)

        val dataStart = 8 + headerSize
        val result = mutableMapOf<String, Tensor>()

        val tensorRegex = Regex("\"([^\"]+)\"\\s*:\\s*\\{([^}]+)\\}")
        for (match in tensorRegex.findAll(headerJson)) {
            val name = match.groupValues[1]
            if (name == "__metadata__") continue

            val body = match.groupValues[2]

            val shapeRegex = Regex("\"shape\"\\s*:\\s*\\[([^\\]]*)\\]")
            val shapeMatch = shapeRegex.find(body) ?: continue
            val shapeStr = shapeMatch.groupValues[1].trim()
            val shape = if (shapeStr.isEmpty()) {
                intArrayOf()
            } else {
                shapeStr.split(",").map { s -> s.trim().toInt() }.toIntArray()
            }

            val offsetRegex = Regex("\"data_offsets\"\\s*:\\s*\\[(\\d+)\\s*,\\s*(\\d+)\\]")
            val offsetMatch = offsetRegex.find(body) ?: continue
            val startOffset = offsetMatch.groupValues[1].toInt()
            val endOffset = offsetMatch.groupValues[2].toInt()

            val byteCount = endOffset - startOffset
            val floatCount = byteCount / 4
            val floatData = FloatArray(floatCount)

            val tensorBuffer = ByteBuffer.wrap(fileBytes, dataStart + startOffset, byteCount).order(ByteOrder.LITTLE_ENDIAN)
            for (i in 0 until floatCount) {
                floatData[i] = tensorBuffer.float
            }

            result[name] = Tensor(floatData, shape)
        }

        return result
    }
}
