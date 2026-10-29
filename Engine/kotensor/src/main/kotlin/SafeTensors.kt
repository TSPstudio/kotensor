package io.kotensor.core

import java.io.Closeable
import java.io.File
import java.io.FileOutputStream
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder

class SafeTensorsFile(val file: File) : Closeable {

    class Entry(val name: String, val dtype: String, val shape: IntArray, val byteOffset: Long, val byteLength: Long) {
        val elementCount: Int
            get() {
                var count = 1L
                for (dim in shape) count *= dim
                require(count <= Int.MAX_VALUE) { "Tensor $name has $count elements, more than a FloatArray can hold" }
                return count.toInt()
            }

        val bytesPerElement: Int
            get() = when (dtype) {
                "F32", "I32", "U32" -> 4
                "F16", "BF16", "I16", "U16" -> 2
                "F64", "I64", "U64" -> 8
                "I8", "U8", "BOOL" -> 1
                else -> throw IllegalArgumentException("Unsupported safetensors dtype '$dtype' for tensor $name")
            }
    }

    private val input = RandomAccessFile(file, "r")
    private val dataStart: Long
    val entries: Map<String, Entry>
    val metadata: Map<String, String>

    init {
        val sizeBytes = ByteArray(8)
        input.readFully(sizeBytes)
        val headerSize = ByteBuffer.wrap(sizeBytes).order(ByteOrder.LITTLE_ENDIAN).long
        require(headerSize in 1..(100L shl 20)) { "Invalid safetensors header size $headerSize in $file" }
        val headerBytes = ByteArray(headerSize.toInt())
        input.readFully(headerBytes)
        dataStart = 8 + headerSize

        val header = JsonParser.parse(String(headerBytes, Charsets.UTF_8)) as? Map<*, *>
            ?: throw IllegalArgumentException("safetensors header of $file is not a JSON object")
        val parsedEntries = LinkedHashMap<String, Entry>()
        val parsedMetadata = LinkedHashMap<String, String>()
        for ((key, value) in header) {
            val name = key as String
            val body = value as? Map<*, *> ?: continue
            if (name == "__metadata__") {
                for ((metaKey, metaValue) in body) parsedMetadata[metaKey as String] = metaValue.toString()
                continue
            }
            val dtype = body["dtype"] as? String ?: throw IllegalArgumentException("Tensor $name has no dtype")
            val shape = (body["shape"] as? List<*> ?: emptyList<Any?>()).map { dim -> (dim as Number).toInt() }.toIntArray()
            val offsets = body["data_offsets"] as? List<*> ?: throw IllegalArgumentException("Tensor $name has no data_offsets")
            val start = (offsets[0] as Number).toLong()
            val end = (offsets[1] as Number).toLong()
            require(end >= start && dataStart + end <= file.length()) { "Tensor $name data range [$start, $end) exceeds file ${file.length()}" }
            parsedEntries[name] = Entry(name, dtype, shape, start, end - start)
        }
        entries = parsedEntries
        metadata = parsedMetadata
    }

    val names: List<String>
        get() = entries.keys.toList()

    operator fun contains(name: String): Boolean = entries.containsKey(name)

    fun entry(name: String): Entry = entries[name] ?: throw IllegalArgumentException("No tensor named '$name' in $file")

    fun load(name: String): Tensor {
        val entry = entry(name)
        val data = FloatArray(entry.elementCount)
        readInto(entry, data, 0)
        return Tensor(data, entry.shape)
    }

    fun readInto(name: String, dest: FloatArray, destOffset: Int = 0) {
        readInto(entry(name), dest, destOffset)
    }

    fun readRaw(name: String, dest: ByteArray, destOffset: Int = 0) {
        val entry = entry(name)
        require(entry.byteLength <= dest.size - destOffset) { "Destination too small for ${entry.name}: needs ${entry.byteLength} bytes at $destOffset, has ${dest.size}" }
        synchronized(input) {
            input.seek(dataStart + entry.byteOffset)
            input.readFully(dest, destOffset, entry.byteLength.toInt())
        }
    }

    fun readRaw(name: String): ByteArray {
        val bytes = ByteArray(entry(name).byteLength.toInt())
        readRaw(name, bytes)
        return bytes
    }

    private fun readInto(entry: Entry, dest: FloatArray, destOffset: Int) {
        val count = entry.elementCount
        require(destOffset >= 0 && destOffset + count <= dest.size) { "Destination too small for ${entry.name}: needs $count elements at $destOffset, has ${dest.size}" }
        val bytesPerElement = entry.bytesPerElement
        require(entry.byteLength == count.toLong() * bytesPerElement) { "Tensor ${entry.name} has ${entry.byteLength} bytes for $count ${entry.dtype} elements" }

        val chunkElements = minOf(count, CHUNK_BYTES / bytesPerElement)
        val chunk = ByteBuffer.allocate(chunkElements * bytesPerElement).order(ByteOrder.LITTLE_ENDIAN)
        var position = dataStart + entry.byteOffset
        var written = destOffset
        var remaining = count
        while (remaining > 0) {
            val elements = minOf(remaining, chunkElements)
            chunk.clear()
            chunk.limit(elements * bytesPerElement)
            synchronized(input) {
                input.seek(position)
                input.readFully(chunk.array(), 0, elements * bytesPerElement)
            }
            convert(entry.dtype, chunk, elements, dest, written)
            position += elements.toLong() * bytesPerElement
            written += elements
            remaining -= elements
        }
    }

    private fun convert(dtype: String, chunk: ByteBuffer, elements: Int, dest: FloatArray, destOffset: Int) {
        when (dtype) {
            "F32" -> chunk.asFloatBuffer().get(dest, destOffset, elements)
            "BF16" -> {
                val shorts = chunk.asShortBuffer()
                for (i in 0 until elements) {
                    dest[destOffset + i] = Float.fromBits((shorts.get(i).toInt() and 0xFFFF) shl 16)
                }
            }
            "F16" -> {
                val shorts = chunk.asShortBuffer()
                for (i in 0 until elements) {
                    dest[destOffset + i] = halfToFloat(shorts.get(i).toInt() and 0xFFFF)
                }
            }
            "F64" -> {
                val doubles = chunk.asDoubleBuffer()
                for (i in 0 until elements) dest[destOffset + i] = doubles.get(i).toFloat()
            }
            "I8" -> for (i in 0 until elements) dest[destOffset + i] = chunk.get(i).toFloat()
            "U8", "BOOL" -> for (i in 0 until elements) dest[destOffset + i] = (chunk.get(i).toInt() and 0xFF).toFloat()
            "I16" -> {
                val shorts = chunk.asShortBuffer()
                for (i in 0 until elements) dest[destOffset + i] = shorts.get(i).toFloat()
            }
            "I32" -> {
                val ints = chunk.asIntBuffer()
                for (i in 0 until elements) dest[destOffset + i] = ints.get(i).toFloat()
            }
            "I64" -> {
                val longs = chunk.asLongBuffer()
                for (i in 0 until elements) dest[destOffset + i] = longs.get(i).toFloat()
            }
            else -> throw IllegalArgumentException("Unsupported safetensors dtype '$dtype'")
        }
    }

    override fun close() {
        input.close()
    }

    companion object {
        private const val CHUNK_BYTES = 4 shl 20

        fun halfToFloat(half: Int): Float {
            val sign = (half ushr 15) and 1
            val exponent = (half ushr 10) and 0x1F
            val mantissa = half and 0x3FF
            val bits = when {
                exponent == 0x1F -> (sign shl 31) or 0x7F800000 or (mantissa shl 13)
                exponent != 0 -> (sign shl 31) or ((exponent - 15 + 127) shl 23) or (mantissa shl 13)
                mantissa == 0 -> sign shl 31
                else -> {
                    var shifted = mantissa
                    var shifts = -1
                    do {
                        shifts++
                        shifted = shifted shl 1
                    } while (shifted and 0x400 == 0)
                    (sign shl 31) or ((127 - 15 - shifts) shl 23) or ((shifted and 0x3FF) shl 13)
                }
            }
            return Float.fromBits(bits)
        }
    }
}

class SafeTensorsEntry(val name: String, val dtype: String, val shape: IntArray, val bytes: ByteArray) {
    companion object {
        fun of(name: String, tensor: Tensor): SafeTensorsEntry {
            val buffer = ByteBuffer.allocate(tensor.data.size * 4).order(ByteOrder.LITTLE_ENDIAN)
            buffer.asFloatBuffer().put(tensor.data)
            return SafeTensorsEntry(name, "F32", tensor.shape.copyOf(), buffer.array())
        }
    }
}

object SafeTensors {

    fun save(file: File, tensors: Map<String, Tensor>, metadata: Map<String, String> = emptyMap()) {
        saveEntries(file, tensors.map { (name, tensor) -> SafeTensorsEntry.of(name, tensor) }, metadata)
    }

    fun saveEntries(file: File, entries: List<SafeTensorsEntry>, metadata: Map<String, String> = emptyMap()) {
        var currentOffset = 0L
        val headerEntries = mutableListOf<String>()
        if (metadata.isNotEmpty()) {
            val pairs = metadata.entries.joinToString(",") { (key, value) -> "${jsonString(key)}:${jsonString(value)}" }
            headerEntries.add("\"__metadata__\":{$pairs}")
        }
        for (entry in entries) {
            val startOffset = currentOffset
            val endOffset = currentOffset + entry.bytes.size
            val shapeString = entry.shape.joinToString(",")
            headerEntries.add("${jsonString(entry.name)}:{\"dtype\":\"${entry.dtype}\",\"shape\":[$shapeString],\"data_offsets\":[$startOffset,$endOffset]}")
            currentOffset = endOffset
        }

        val headerBytes = ("{" + headerEntries.joinToString(",") + "}").toByteArray(Charsets.UTF_8)
        val padding = (8 - headerBytes.size % 8) % 8
        val headerSize = (headerBytes.size + padding).toLong()

        FileOutputStream(file).use { out ->
            val sizeBuffer = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN)
            sizeBuffer.putLong(headerSize)
            out.write(sizeBuffer.array())
            out.write(headerBytes)
            repeat(padding) { out.write(' '.code) }
            for (entry in entries) {
                out.write(entry.bytes)
            }
        }
    }

    private fun jsonString(text: String): String {
        val sb = StringBuilder("\"")
        for (c in text) {
            when (c) {
                '"' -> sb.append("\\\"")
                '\\' -> sb.append("\\\\")
                '\n' -> sb.append("\\n")
                '\r' -> sb.append("\\r")
                '\t' -> sb.append("\\t")
                else -> if (c.code < 0x20) sb.append(String.format("\\u%04x", c.code)) else sb.append(c)
            }
        }
        return sb.append('"').toString()
    }

    fun open(file: File): SafeTensorsFile = SafeTensorsFile(file)

    fun load(file: File): Map<String, Tensor> {
        return SafeTensorsFile(file).use { reader ->
            val result = LinkedHashMap<String, Tensor>()
            for (name in reader.names) {
                result[name] = reader.load(name)
            }
            result
        }
    }
}

