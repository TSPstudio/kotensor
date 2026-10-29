package io.kotensor.core

import java.io.File
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull
import kotlin.test.assertTrue

class QuantizationTest {

    private fun seededModel(vocabularySize: Int = 48, dimension: Int = 32, numLayers: Int = 2, numHeads: Int = 4, numKeyValueHeads: Int = 2, hiddenDimension: Int = 80, seed: Int = 17): Transformer {
        val model = Transformer(vocabularySize, dimension, numLayers, numHeads, hiddenDimension, numKeyValueHeads)
        val random = Random(seed)
        for (p in model.parameters()) {
            for (i in p.data.indices) p.data[i] = (random.nextFloat() * 2f - 1f) * 0.2f
        }
        return model
    }

    private fun maxAbsDiff(a: FloatArray, b: FloatArray): Float {
        assertEquals(a.size, b.size)
        var max = 0.0f
        for (i in a.indices) max = maxOf(max, abs(a[i] - b[i]))
        return max
    }

    private fun maxAbs(a: FloatArray): Float {
        var max = 0.0f
        for (v in a) max = maxOf(max, abs(v))
        return max
    }

    @Test
    fun testQuantizeDequantizeErrorBounds() {
        val random = Random(1)
        val rows = 100
        val columns = 37
        val tensor = Tensor(FloatArray(rows * columns) { (random.nextFloat() * 2f - 1f) * 0.3f }, intArrayOf(rows, columns))

        val int8 = QuantizedTensor.quantize(tensor, Quantization.INT8)
        assertEquals(8, int8.bits)
        assertEquals(2, int8.groups)
        assertNull(int8.zeros)
        assertEquals(rows * columns, int8.qdata.size)
        val int8Error = maxAbsDiff(tensor.data, int8.dequantize().data)
        assertTrue(int8Error <= 0.3f / 127f + 1e-6f, "int8 error $int8Error")

        val int4 = QuantizedTensor.quantize(tensor, Quantization(4, 32))
        assertEquals(4, int4.groups)
        assertEquals(38, int4.storedColumns)
        assertEquals(19, int4.bytesPerRow)
        assertEquals(rows * 19, int4.qdata.size)
        assertEquals(4 * columns, int4.zeros!!.size)
        val int4Error = maxAbsDiff(tensor.data, int4.dequantize().data)
        assertTrue(int4Error <= 0.6f / 15f / 2f + 1e-6f, "int4 error $int4Error")
        for (r in 0 until rows) {
            for (c in 0 until columns) {
                assertTrue(int4.rawValue(r, c) in 0..15)
                assertEquals(int4.value(r, c), int4.dequantize().data[r * columns + c])
            }
        }

        val zerosTensor = Tensor(FloatArray(rows * columns), intArrayOf(rows, columns))
        assertEquals(0.0f, maxAbs(QuantizedTensor.quantize(zerosTensor, Quantization.INT4).dequantize().data))
        assertEquals(0.0f, maxAbs(QuantizedTensor.quantize(zerosTensor, Quantization.INT8).dequantize().data))

        assertEquals(rows.toLong() * columns + 2L * columns * 4L, int8.sizeInBytes)
        assertEquals(rows.toLong() * 19 + 4L * columns * 4L + 4L * columns, int4.sizeInBytes)
        assertFailsWith<IllegalArgumentException> { Quantization(3) }
    }

    @Test
    fun testDequantizingMatmulMatchesDequantizedWeights() {
        val random = Random(2)
        for ((m, k, n) in listOf(Triple(1, 70, 300), Triple(1, 64, 64), Triple(5, 130, 577), Triple(40, 96, 33), Triple(1, 256, 1537))) {
            val a = FloatArray(m * k) { random.nextFloat() - 0.5f }
            val w = FloatArray(k * n) { (random.nextFloat() - 0.5f) * 0.2f }
            for (quantization in listOf(Quantization.INT8, Quantization.INT4, Quantization(4, 32), Quantization(8, 1000))) {
                val quantized = QuantizedTensor.quantize(w, k, n, quantization)
                val dequantized = quantized.dequantize().data

                val expected = FloatArray(m * n)
                MatMul.multiply(a, dequantized, expected, m, k, n, accumulate = false)
                val actual = FloatArray(m * n)
                MatMul.multiplyQuantized(a, quantized, actual, m, accumulate = false)
                assertTrue(maxAbsDiff(expected, actual) < 1e-4f, "[$m,$k]×[$k,$n] $quantization forward")

                val accumulated = FloatArray(m * n) { 1.0f }
                MatMul.multiplyQuantized(a, quantized, accumulated, m, accumulate = true, alpha = 0.5f)
                for (i in accumulated.indices) {
                    assertEquals(1.0f + 0.5f * expected[i], accumulated[i], 1e-4f)
                }

                val dy = FloatArray(m * n) { random.nextFloat() - 0.5f }
                val transposed = FloatArray(k * n)
                MatMul.transpose(dequantized, k, n, transposed)
                val expectedDx = FloatArray(m * k)
                MatMul.multiply(dy, transposed, expectedDx, m, n, k, accumulate = false)
                val dx = FloatArray(m * k)
                MatMul.multiplyByQuantizedTransposed(dy, quantized, dx, m, accumulate = false)
                assertTrue(maxAbsDiff(expectedDx, dx) < 1e-3f, "[$m,$k]×[$k,$n] $quantization backward")
            }
        }
    }

    @Test
    fun testParallelQuantizedKernelMatchesSerial() {
        val random = Random(4)
        val k = 300
        val n = 2100
        val a = FloatArray(k) { random.nextFloat() - 0.5f }
        val quantized = QuantizedTensor.quantize(FloatArray(k * n) { random.nextFloat() - 0.5f }, k, n, Quantization.INT4)
        val saved = MatMul.parallelThreshold
        try {
            MatMul.parallelThreshold = 0L
            val parallel = FloatArray(n)
            MatMul.multiplyQuantized(a, quantized, parallel, 1, accumulate = false)
            MatMul.parallelThreshold = Long.MAX_VALUE
            val serial = FloatArray(n)
            MatMul.multiplyQuantized(a, quantized, serial, 1, accumulate = false)
            assertContentEquals(serial, parallel)
            val batched = FloatArray(3 * n)
            MatMul.multiplyQuantized(FloatArray(3 * k) { i -> a[i % k] }, quantized, batched, 3, accumulate = false)
            for (row in 0 until 3) {
                for (j in 0 until n) assertEquals(serial[j], batched[row * n + j], 1e-4f)
            }
        } finally {
            MatMul.parallelThreshold = saved
        }
    }

    @Test
    fun testMatmulDequantGraphGradientFlowsToInput() {
        val random = Random(6)
        val x = Tensor(FloatArray(3 * 40) { random.nextFloat() - 0.5f }, intArrayOf(3, 40))
        val w = Tensor(FloatArray(40 * 24) { (random.nextFloat() - 0.5f) * 0.3f }, intArrayOf(40, 24))
        val quantized = QuantizedTensor.quantize(w, Quantization.INT8)

        val out = x.matmulDequant(quantized)
        val reference = x matmul quantized.dequantize()
        assertTrue(maxAbsDiff(reference.data, out.data) < 1e-5f)
        assertTrue(out.shape.contentEquals(intArrayOf(3, 24)))

        for (i in out.grad.indices) out.grad[i] = random.nextFloat() - 0.5f
        reference.grad.indices.forEach { i -> reference.grad[i] = out.grad[i] }
        val xReference = Tensor(x.data.copyOf(), intArrayOf(3, 40))
        val referenceOut = xReference matmul quantized.dequantize()
        for (i in referenceOut.grad.indices) referenceOut.grad[i] = out.grad[i]
        out.backwardOp()
        referenceOut.backwardOp()
        assertTrue(maxAbsDiff(xReference.grad, x.grad) < 1e-4f)
    }

    @Test
    fun testQuantizedModelMatchesFloatModelInAllPaths() {
        val model = seededModel()
        val tokens = intArrayOf(3, 11, 7, 40, 0, 25, 9)
        val reference = model.forward(tokens)
        val referenceScale = maxAbs(reference.data)

        for ((quantization, tolerance) in listOf(Quantization.INT8 to 0.02f, Quantization.INT4 to 0.25f)) {
            val quantized = seededModel().quantize(quantization)
            assertEquals(quantization, quantized.quantization)
            assertTrue(quantized.lmHead is QuantizedLinear)
            assertTrue(quantized.embedding is QuantizedEmbedding)
            assertTrue(quantized.blocks[0].ffn.wGate is QuantizedLinear)
            assertEquals(15, quantized.namedQuantizedLayers().size)
            assertTrue(quantized.weightBytes() < model.weightBytes())
            assertEquals(model.parameterCount(), quantized.parameterCount())

            val full = quantized.forward(tokens)
            val error = maxAbsDiff(reference.data, full.data)
            assertTrue(error < tolerance * referenceScale, "$quantization full forward error $error of $referenceScale")

            val cache = quantized.newCache(tokens.size)
            val buffers = quantized.newInferenceBuffers(tokens.size)
            val prefill = quantized.forward(tokens.sliceArray(0 until 3), cache)
            for (t in 0 until 3) {
                for (v in 0 until 48) assertEquals(full.data[t * 48 + v], prefill.data[t * 48 + v], 1e-4f)
            }
            for (t in 3 until tokens.size) {
                val step = quantized.decode(tokens[t], cache, buffers)
                for (v in 0 until 48) assertEquals(full.data[t * 48 + v], step.data[v], 1e-4f, "$quantization decode position $t vocab $v")
            }

            val batch = Tensor(FloatArray(14) { i -> tokens[i % 7].toFloat() }, intArrayOf(2, 7))
            val batched = quantized.forward(batch)
            for (i in 0 until full.data.size) {
                assertEquals(full.data[i], batched.data[i], 1e-4f)
                assertEquals(full.data[i], batched.data[full.data.size + i], 1e-4f)
            }
            assertEquals(tokens.size + 4, quantized.generate(tokens, maxTokens = 4).size)
        }
    }

    @Test
    fun testDecodeOnQuantizedModelAllocatesNothing() {
        val threadBean = java.lang.management.ManagementFactory.getThreadMXBean() as? com.sun.management.ThreadMXBean ?: return
        if (!threadBean.isThreadAllocatedMemorySupported) return
        threadBean.isThreadAllocatedMemoryEnabled = true
        val model = seededModel().quantize(Quantization.INT4)
        val cache = model.newCache(512)
        val buffers = model.newInferenceBuffers(512)
        repeat(400) { model.decode(it % 48, cache, buffers) }
        cache.reset()
        val before = threadBean.currentThreadAllocatedBytes
        repeat(100) { model.decode(it % 48, cache, buffers) }
        assertEquals(0L, threadBean.currentThreadAllocatedBytes - before)
    }

    @Test
    fun testLoRAOnQuantizedBaseTrainsAndDecodes() {
        val model = seededModel(numLayers = 1).quantize(Quantization.INT8)
        val adapter = LoRAAdapter(model, rank = 2, alpha = 1.0f)
        assertEquals(2, adapter.layers.size)
        assertTrue(adapter.layers.getValue("block.0.attention.wQ").baseLinear is QuantizedLinear)
        assertTrue(model.namedQuantizedLayers().containsKey("block.0.attention.wQ"))

        val optimizer = AdamW(adapter.parameters(), learningRate = 0.05f)
        val inputs = intArrayOf(1, 5, 9, 2, 7)
        val targets = intArrayOf(5, 9, 2, 7, 3)
        var first = 0.0f
        var last = 0.0f
        for (epoch in 0 until 25) {
            val loss = crossEntropyLoss(model.forward(inputs), targets)
            if (epoch == 0) first = loss.data[0]
            last = loss.data[0]
            loss.backward()
            optimizer.step()
            optimizer.zeroGrad()
        }
        assertTrue(last < first, "loss $first -> $last")

        val full = model.forward(inputs)
        val cache = model.newCache(inputs.size)
        val buffers = model.newInferenceBuffers(inputs.size)
        for (t in inputs.indices) {
            val step = model.decode(inputs[t], cache, buffers)
            for (v in 0 until 48) assertEquals(full.data[t * 48 + v], step.data[v], 1e-4f)
        }

        assertFailsWith<IllegalStateException> { adapter.merge() }
        val adapterFile = File.createTempFile("qlora", ".safetensors")
        adapterFile.deleteOnExit()
        adapter.saveDelta(adapterFile)
        assertTrue(SafeTensors.load(adapterFile).containsKey("block.0.attention.wQ.loraA"))
        assertFailsWith<IllegalStateException> { model.quantize(Quantization.INT4) }
    }

    @Test
    fun testSaveAndLoadQuantizedModelRoundTrip() {
        for (quantization in listOf(Quantization.INT8, Quantization(4, 16))) {
            val model = seededModel(numLayers = 2).quantize(quantization)
            val file = File.createTempFile("kotensor_${quantization.bits}bit", ".safetensors")
            file.deleteOnExit()
            model.saveQuantized(file)
            assertTrue(file.length() < model.parameterCount() * 4L)

            val loaded = Transformer.loadQuantized(file)
            assertEquals(quantization.bits, loaded.quantization!!.bits)
            assertEquals(quantization.groupSize, loaded.quantization!!.groupSize)
            assertEquals(model.numKeyValueHeads, loaded.numKeyValueHeads)
            assertEquals(model.weightBytes(), loaded.weightBytes())
            assertEquals(16, model.namedQuantizedTensors().size)
            for ((name, tensor) in model.namedQuantizedTensors()) {
                val other = loaded.namedQuantizedTensors().getValue(name)
                assertContentEquals(tensor.qdata, other.qdata, name)
                assertContentEquals(tensor.scales, other.scales, name)
                assertContentEquals(tensor.zeros, other.zeros, name)
            }
            for ((name, layer) in model.namedQuantizedLayers()) {
                assertContentEquals(layer.bias.data, loaded.namedQuantizedLayers().getValue(name).bias.data, name)
            }
            assertTrue(loaded.embedding is QuantizedEmbedding)
            val tokens = intArrayOf(4, 8, 15, 16, 23, 42)
            assertContentEquals(model.forward(tokens).data, loaded.forward(tokens).data)

            SafeTensorsFile(file).use { reader ->
                assertEquals("kotensor", reader.metadata.keys.first().substringBefore('.'))
                assertEquals(if (quantization.bits == 8) "I8" else "U8", reader.entry("block.0.attention.wQ.qweight").dtype)
            }
        }
        assertFailsWith<IllegalStateException> { seededModel().saveQuantized(File.createTempFile("fp32", ".safetensors").also { it.deleteOnExit() }) }
    }

    @Test
    fun testHuggingFaceImportWithQuantizationMatchesQuantizeAfterImport() {
        val url = QuantizationTest::class.java.getResource("/hf/qwen_tiny_sharded/expected.json")!!
        val directory = File(url.toURI()).parentFile
        val quantization = Quantization(4, 32)

        val direct = HuggingFaceModelLoader.load(directory, quantization)
        val afterwards = HuggingFaceModelLoader.load(directory).quantize(quantization)
        assertTrue(direct.lmHead is QuantizedLinear)
        assertEquals(afterwards.weightBytes(), direct.weightBytes())
        assertTrue(direct.embedding is QuantizedEmbedding)
        for ((name, tensor) in direct.namedQuantizedTensors()) {
            val other = afterwards.namedQuantizedTensors().getValue(name)
            assertContentEquals(other.qdata, tensor.qdata, name)
            assertContentEquals(other.scales, tensor.scales, name)
        }
        for ((name, layer) in direct.namedQuantizedLayers()) {
            assertContentEquals(afterwards.namedQuantizedLayers().getValue(name).bias.data, layer.bias.data, name)
        }
        val tokens = intArrayOf(1, 2, 3, 5, 8)
        assertContentEquals(afterwards.forward(tokens).data, direct.forward(tokens).data)

        val fp32 = HuggingFaceModelLoader.load(directory)
        val reference = fp32.forward(tokens).data
        val int8 = HuggingFaceModelLoader.load(directory, Quantization.INT8).forward(tokens).data
        assertTrue(maxAbsDiff(reference, int8) < 0.05f * maxAbs(reference))
        assertTrue(HuggingFaceModelLoader.load(directory, Quantization.INT8).weightBytes() < fp32.weightBytes())
    }

    @Test
    fun testMemoryFootprintFollowsBitsPerWeight() {
        val model = seededModel(vocabularySize = 64, dimension = 64, numLayers = 2, numHeads = 4, numKeyValueHeads = 4, hiddenDimension = 128)
        val fp32 = model.weightBytes()
        var projectionParams = 0L
        for (tensor in seededModel(vocabularySize = 64, dimension = 64, numLayers = 2, numHeads = 4, numKeyValueHeads = 4, hiddenDimension = 128).quantize(Quantization.INT8).namedQuantizedTensors().values) {
            projectionParams += tensor.rows.toLong() * tensor.columns
        }
        val other = fp32 - projectionParams * 4L

        val int8 = seededModel(vocabularySize = 64, dimension = 64, numLayers = 2, numHeads = 4, numKeyValueHeads = 4, hiddenDimension = 128).quantize(Quantization.INT8).weightBytes()
        val int4 = seededModel(vocabularySize = 64, dimension = 64, numLayers = 2, numHeads = 4, numKeyValueHeads = 4, hiddenDimension = 128).quantize(Quantization.INT4).weightBytes()
        val int8PerWeight = (int8 - other).toDouble() / projectionParams
        val int4PerWeight = (int4 - other).toDouble() / projectionParams
        assertTrue(int8PerWeight in 1.0..1.2, "int8 bytes per weight $int8PerWeight")
        assertTrue(int4PerWeight in 0.5..0.7, "int4 bytes per weight $int4PerWeight")
    }

    @Test
    fun testGradientsAreAllocatedLazily() {
        val threadBean = java.lang.management.ManagementFactory.getThreadMXBean() as? com.sun.management.ThreadMXBean ?: return
        if (!threadBean.isThreadAllocatedMemorySupported) return
        threadBean.isThreadAllocatedMemoryEnabled = true
        val before = threadBean.currentThreadAllocatedBytes
        val tensor = Tensor(FloatArray(1 shl 20), intArrayOf(1024, 1024))
        val afterData = threadBean.currentThreadAllocatedBytes
        tensor.grad[0] = 1.0f
        val afterGrad = threadBean.currentThreadAllocatedBytes
        assertTrue(afterData - before < 2L * (1 shl 22), "data allocation ${afterData - before}")
        assertTrue(afterGrad - afterData >= (1 shl 22).toLong(), "grad allocated on first use")
        assertEquals(1.0f, tensor.grad[0])
    }
}

