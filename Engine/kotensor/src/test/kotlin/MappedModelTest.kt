package io.kotensor.core

import java.io.File
import java.lang.management.ManagementFactory
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class MappedModelTest {

    private fun seededModel(vocabularySize: Int = 48, dimension: Int = 32, numLayers: Int = 2, numHeads: Int = 4, numKeyValueHeads: Int = 2, hiddenDimension: Int = 80, seed: Int = 31): Transformer {
        val model = Transformer(vocabularySize, dimension, numLayers, numHeads, hiddenDimension, numKeyValueHeads)
        val random = Random(seed)
        for (p in model.parameters()) {
            for (i in p.data.indices) p.data[i] = (random.nextFloat() * 2f - 1f) * 0.2f
        }
        return model
    }

    private fun savedModel(quantization: Quantization, tied: Boolean = false): File {
        val model = seededModel().quantize(quantization)
        if (tied) model.tieEmbeddingToHead()
        val file = File.createTempFile("kotensor_mapped_${quantization.bits}", ".safetensors")
        file.deleteOnExit()
        model.saveQuantized(file)
        return file
    }

    private fun decodeLogits(model: Transformer, tokens: IntArray): List<FloatArray> {
        val cache = model.newCache(tokens.size)
        val buffers = model.newInferenceBuffers(tokens.size)
        val result = ArrayList<FloatArray>()
        result.add(model.prefill(tokens.copyOfRange(0, tokens.size - 3), cache, buffers).data.copyOf())
        for (t in tokens.size - 3 until tokens.size) {
            result.add(model.decode(tokens[t], cache, buffers).data.copyOf())
        }
        return result
    }

    @Test
    fun testMemoryMappedModelMatchesHeapModelExactly() {
        for (quantization in listOf(Quantization.INT8, Quantization.INT4)) {
            val file = savedModel(quantization)
            val heap = Transformer.loadQuantized(file)
            val mapped = Transformer.loadQuantized(file, memoryMapped = true)

            for ((name, tensor) in mapped.namedQuantizedTensors()) {
                assertTrue(tensor.isMemoryMapped, "$name should be mapped")
                assertFalse(heap.namedQuantizedTensors().getValue(name).isMemoryMapped)
            }
            assertEquals(heap.weightBytes(), mapped.weightBytes())

            val tokens = intArrayOf(4, 9, 33, 2, 17, 40, 8, 1, 25, 12)
            val expected = decodeLogits(heap, tokens)
            val actual = decodeLogits(mapped, tokens)
            for (step in expected.indices) assertContentEquals(expected[step], actual[step], "$quantization step $step")
            assertContentEquals(heap.generate(tokens, maxTokens = 6), mapped.generate(tokens, maxTokens = 6))

            val batch = Tensor(FloatArray(tokens.size) { i -> tokens[i].toFloat() }, intArrayOf(1, tokens.size))
            assertContentEquals(heap.forward(batch).data, mapped.forward(batch).data)
        }
    }

    @Test
    fun testMappedTensorScalarAccessMatchesHeap() {
        val file = savedModel(Quantization.INT4)
        val heap = Transformer.loadQuantized(file).namedQuantizedTensors().getValue("block.0.ffn.wDown")
        val mapped = Transformer.loadQuantized(file, memoryMapped = true).namedQuantizedTensors().getValue("block.0.ffn.wDown")
        val heapRow = FloatArray(heap.columns)
        val mappedRow = FloatArray(mapped.columns)
        for (r in 0 until heap.rows) {
            heap.dequantizeRow(r, heapRow)
            mapped.dequantizeRow(r, mappedRow)
            assertContentEquals(heapRow, mappedRow)
            for (c in 0 until heap.columns) {
                assertEquals(heap.rawValue(r, c), mapped.rawValue(r, c))
                assertEquals(heap.value(r, c), mapped.value(r, c))
            }
        }
        assertTrue(mapped.isMemoryMapped)
    }

    @Test
    fun testMappedDecodeAllocatesNothing() {
        val threadBean = ManagementFactory.getThreadMXBean() as? com.sun.management.ThreadMXBean ?: return
        if (!threadBean.isThreadAllocatedMemorySupported) return
        threadBean.isThreadAllocatedMemoryEnabled = true
        val model = Transformer.loadQuantized(savedModel(Quantization.INT4), memoryMapped = true)
        val cache = model.newCache(512)
        val buffers = model.newInferenceBuffers(512)
        repeat(400) { model.decode(it % 48, cache, buffers) }
        var allocated = -1L
        var window = 0
        while (allocated != 0L && window < 10) {
            cache.reset()
            val before = threadBean.currentThreadAllocatedBytes
            repeat(100) { model.decode(it % 48, cache, buffers) }
            allocated = threadBean.currentThreadAllocatedBytes - before
            window++
        }
        assertEquals(0L, allocated)
    }

    @Test
    fun testSavingMappedModelReproducesTheFileAndStaysMapped() {
        for (quantization in listOf(Quantization.INT8, Quantization.INT4)) {
            val file = savedModel(quantization)
            val mapped = Transformer.loadQuantized(file, memoryMapped = true)
            val copy = File.createTempFile("kotensor_mapped_copy", ".safetensors")
            copy.deleteOnExit()
            mapped.saveQuantized(copy)
            assertContentEquals(file.readBytes(), copy.readBytes())
            for (tensor in mapped.namedQuantizedTensors().values) assertTrue(tensor.isMemoryMapped)
        }
    }

    @Test
    fun testReadingArraysMovesAMappedTensorToTheHeap() {
        val file = savedModel(Quantization.INT4)
        val heap = Transformer.loadQuantized(file)
        val mapped = Transformer.loadQuantized(file, memoryMapped = true)
        val tensor = mapped.namedQuantizedTensors().getValue("block.1.attention.wQ")
        val reference = heap.namedQuantizedTensors().getValue("block.1.attention.wQ")
        assertTrue(tensor.isMemoryMapped)
        assertContentEquals(reference.scales, tensor.scales)
        assertFalse(tensor.isMemoryMapped)
        assertContentEquals(reference.qdata, tensor.qdata)
        assertContentEquals(reference.zeros, tensor.zeros)

        tensor.scales[0] = tensor.scales[0] * 2f
        val row = FloatArray(tensor.columns)
        tensor.dequantizeRow(0, row)
        val expected = FloatArray(reference.columns)
        reference.dequantizeRow(0, expected)
        assertEquals(expected[0] * 2f, row[0], 1e-6f)
    }

    @Test
    fun testMapReturnsTheTensorBytes() {
        val file = savedModel(Quantization.INT8)
        SafeTensorsFile(file).use { reader ->
            for (name in reader.names) {
                val mapped = reader.map(name)
                val bytes = ByteArray(mapped.remaining())
                mapped.get(bytes)
                assertContentEquals(reader.readRaw(name), bytes, name)
            }
        }
    }

    @Test
    fun testLoadingWithoutMappingStillAllocatesOnTheHeap() {
        val file = savedModel(Quantization.INT8)
        val model = Transformer.loadQuantized(file, memoryMapped = false)
        for (tensor in model.namedQuantizedTensors().values) assertFalse(tensor.isMemoryMapped)
    }

    @Test
    fun testTiedEmbeddingReadsColumnsOfTheHead() {
        val model = seededModel().quantize(Quantization.INT4)
        val untiedBytes = model.weightBytes()
        val untiedParameters = model.parameterCount()
        val embeddingBytes = (model.embedding as QuantizedEmbedding).sizeInBytes
        model.tieEmbeddingToHead()
        assertTrue(model.isEmbeddingTied)

        val head = (model.lmHead as QuantizedLinear).weight
        val tokens = intArrayOf(0, 7, 47)
        val embedded = model.embedding.forward(tokens)
        assertFalse(embedded.requiresGrad)
        val row = Tensor(FloatArray(model.dimension), intArrayOf(1, model.dimension))
        for ((t, token) in tokens.withIndex()) {
            model.embedding.lookupInto(token, row)
            for (d in 0 until model.dimension) {
                assertEquals(head.value(d, token), embedded.data[t * model.dimension + d])
                assertEquals(head.value(d, token), row.data[d])
            }
        }
        assertEquals(untiedBytes - embeddingBytes, model.weightBytes())
        assertEquals(untiedParameters - model.vocabularySize.toLong() * model.dimension, model.parameterCount())
    }

    @Test
    fun testTiedModelRoundTripsThroughFilesWithoutAnEmbeddingCopy() {
        val model = seededModel().quantize(Quantization.INT4).tieEmbeddingToHead()
        val file = File.createTempFile("kotensor_tied", ".safetensors")
        file.deleteOnExit()
        model.saveQuantized(file)
        SafeTensorsFile(file).use { reader ->
            assertEquals("true", reader.metadata["kotensor.tiedEmbedding"])
            assertTrue(reader.names.none { it.startsWith("embedding.") })
        }

        val tokens = intArrayOf(3, 30, 12, 45, 6, 21, 9)
        val expected = model.generate(tokens, maxTokens = 5)
        for (memoryMapped in listOf(false, true)) {
            val loaded = Transformer.loadQuantized(file, memoryMapped)
            assertTrue(loaded.isEmbeddingTied)
            assertContentEquals(expected, loaded.generate(tokens, maxTokens = 5))
            assertContentEquals(decodeLogits(model, tokens).last(), decodeLogits(loaded, tokens).last())
        }
    }

    @Test
    fun testUntiedFilesStillLoadWithTheirOwnEmbedding() {
        val loaded = Transformer.loadQuantized(savedModel(Quantization.INT4), memoryMapped = true)
        assertFalse(loaded.isEmbeddingTied)
        assertTrue(loaded.embedding is QuantizedEmbedding)
        assertTrue((loaded.embedding as QuantizedEmbedding).weight.isMemoryMapped)
    }

    @Test
    fun testTyingNeedsAQuantizedHead() {
        assertFailsWith<IllegalStateException> { seededModel().tieEmbeddingToHead() }
    }

    @Test
    fun testDequantizeColumnMatchesValues() {
        val random = Random(8)
        val data = FloatArray(70 * 9) { random.nextFloat() - 0.5f }
        for (quantization in listOf(Quantization.INT8, Quantization.INT4)) {
            val tensor = QuantizedTensor.quantize(data, 70, 9, quantization)
            val column = FloatArray(72)
            tensor.dequantizeColumn(4, column, 2)
            for (r in 0 until 70) assertEquals(tensor.value(r, 4), column[2 + r])
            assertFailsWith<IllegalArgumentException> { tensor.dequantizeColumn(9, column) }
        }
    }
}
