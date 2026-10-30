package io.kotensor.core

import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class TrainingTest {

    private fun assertClose(expected: FloatArray, actual: FloatArray, tolerance: Float, label: String) {
        assertEquals(expected.size, actual.size, "$label size")
        for (i in expected.indices) {
            assertEquals(expected[i], actual[i], tolerance, "$label element $i")
        }
    }

    private fun randomTensor(random: Random, rows: Int, cols: Int, scale: Float = 1.0f): Tensor {
        return Tensor(FloatArray(rows * cols) { (random.nextFloat() - 0.5f) * scale }, intArrayOf(rows, cols))
    }

    private fun naive(a: FloatArray, b: FloatArray, m: Int, k: Int, n: Int): FloatArray {
        val c = FloatArray(m * n)
        for (i in 0 until m) {
            for (j in 0 until n) {
                var sum = 0.0f
                for (l in 0 until k) sum += a[i * k + l] * b[l * n + j]
                c[i * n + j] = sum
            }
        }
        return c
    }

    private fun transposed(src: FloatArray, rows: Int, cols: Int): FloatArray {
        val dst = FloatArray(rows * cols)
        MatMul.transpose(src, rows, cols, dst)
        return dst
    }

    private fun <T> withVectorApi(enabled: Boolean, body: () -> T): T {
        val saved = MatMul.useVectorApi
        try {
            MatMul.useVectorApi = enabled
            return body()
        } finally {
            MatMul.useVectorApi = saved
        }
    }

    private fun vectorModes(): List<Boolean> = if (MatMul.vectorApiAvailable) listOf(false, true) else listOf(false)

    private fun seededModel(seed: Int, vocabularySize: Int = 11, numLayers: Int = 2): Transformer {
        val model = Transformer(vocabularySize, dimension = 16, numLayers = numLayers, numHeads = 4, numKeyValueHeads = 2)
        val random = Random(seed)
        for (tensor in model.namedParameters().values) {
            for (i in tensor.data.indices) tensor.data[i] = (random.nextFloat() - 0.5f) * 0.4f
        }
        return model
    }

    private fun seedAdapter(adapter: LoRAAdapter, seed: Int) {
        val random = Random(seed)
        for (p in adapter.parameters()) {
            for (i in p.data.indices) p.data[i] = (random.nextFloat() - 0.5f) * 0.2f
        }
    }

    @Test
    fun testMaskedCrossEntropyMatchesTheActiveRows() {
        val random = Random(1)
        val vocabularySize = 7
        val logits = randomTensor(random, 5, vocabularySize, 4.0f)
        val targets = intArrayOf(3, IGNORE_INDEX, 0, IGNORE_INDEX, 6)
        val loss = crossEntropyLoss(logits, targets)
        loss.backward()

        val activeRows = intArrayOf(0, 2, 4)
        val subset = Tensor(FloatArray(activeRows.size * vocabularySize) { i -> logits.data[activeRows[i / vocabularySize] * vocabularySize + i % vocabularySize] }, intArrayOf(activeRows.size, vocabularySize))
        val reference = crossEntropyLoss(subset, intArrayOf(3, 0, 6))
        reference.backward()

        assertEquals(reference.data[0], loss.data[0], 1e-6f)
        for (row in 0 until 5) {
            for (v in 0 until vocabularySize) {
                val subsetRow = activeRows.indexOf(row)
                val expected = if (subsetRow < 0) 0.0f else subset.grad[subsetRow * vocabularySize + v]
                assertEquals(expected, logits.grad[row * vocabularySize + v], 1e-7f, "row $row column $v")
            }
        }

        val allMasked = crossEntropyLoss(logits, IntArray(5) { IGNORE_INDEX })
        assertEquals(0.0f, allMasked.data[0])
        assertFalse(allMasked.requiresGrad)
        assertFailsWith<IllegalArgumentException> { crossEntropyLoss(logits, intArrayOf(3, -1, 0, 1, 2)) }
        assertEquals(loss.data[0], crossEntropyLoss(logits, Tensor(targets.map { it.toFloat() }.toFloatArray(), intArrayOf(5))).data[0])
    }

    @Test
    fun testFusedLossMatchesProjectionPlusCrossEntropy() {
        val random = Random(2)
        val dimension = 8
        val vocabularySize = 13
        val targets = intArrayOf(4, IGNORE_INDEX, 12, 0, IGNORE_INDEX, 7)
        for (simd in vectorModes()) {
            withVectorApi(simd) {
                val head = Linear(dimension, vocabularySize)
                for (i in head.bias.data.indices) head.bias.data[i] = random.nextFloat() - 0.5f
                val hidden = randomTensor(random, targets.size, dimension, 2.0f)

                val fused = fusedCrossEntropyLoss(hidden, head, targets)
                fused.backward()
                val fusedHidden = hidden.grad.copyOf()
                val fusedWeight = head.weight.grad.copyOf()
                val fusedBias = head.bias.grad.copyOf()
                hidden.grad.fill(0.0f)
                head.weight.grad.fill(0.0f)
                head.bias.grad.fill(0.0f)

                val reference = crossEntropyLoss(head(hidden), targets)
                reference.backward()

                assertEquals(reference.data[0], fused.data[0], 1e-6f, "loss (simd=$simd)")
                assertClose(hidden.grad, fusedHidden, 1e-6f, "hidden gradient (simd=$simd)")
                assertClose(head.weight.grad, fusedWeight, 1e-6f, "weight gradient (simd=$simd)")
                assertClose(head.bias.grad, fusedBias, 1e-6f, "bias gradient (simd=$simd)")
                for (d in 0 until dimension) {
                    assertEquals(0.0f, fusedHidden[1 * dimension + d], "masked row 1 gets no gradient")
                    assertEquals(0.0f, fusedHidden[4 * dimension + d], "masked row 4 gets no gradient")
                }
            }
        }
    }

    @Test
    fun testFusedLossWithQuantizedHeadAndFrozenWeights() {
        val random = Random(3)
        val dimension = 16
        val vocabularySize = 37
        val targets = intArrayOf(IGNORE_INDEX, 5, 36, IGNORE_INDEX, 1)
        val float = Linear(dimension, vocabularySize)
        val head = QuantizedLinear.fromLinear(float, Quantization.INT4)
        for (i in head.bias.data.indices) head.bias.data[i] = random.nextFloat() - 0.5f
        head.requiresGrad = false
        val hidden = randomTensor(random, targets.size, dimension)

        val fused = fusedCrossEntropyLoss(hidden, head, targets)
        fused.backward()
        val fusedHidden = hidden.grad.copyOf()
        hidden.grad.fill(0.0f)

        val reference = crossEntropyLoss(head(hidden), targets)
        reference.backward()

        assertEquals(reference.data[0], fused.data[0], 1e-5f)
        assertClose(hidden.grad, fusedHidden, 1e-6f, "hidden gradient")
        assertFalse(head.bias.hasGrad, "a frozen bias gets no gradient array")

        val frozenLinear = Linear(dimension, vocabularySize)
        frozenLinear.requiresGrad = false
        val frozenHidden = randomTensor(random, targets.size, dimension)
        fusedCrossEntropyLoss(frozenHidden, frozenLinear, targets).backward()
        assertFalse(frozenLinear.weight.hasGrad, "a frozen head skips its weight gradient")
        assertTrue(frozenHidden.grad.any { it != 0.0f })
    }

    @Test
    fun testFusedLossStreamsLargeVocabulariesInChunks() {
        val random = Random(4)
        val dimension = 4
        val vocabularySize = 1 shl 19
        val head = Linear(dimension, vocabularySize)
        val targets = IntArray(11) { i -> if (i == 3) IGNORE_INDEX else (i * 48611) % vocabularySize }
        val hidden = randomTensor(random, targets.size, dimension, 2.0f)

        val fused = fusedCrossEntropyLoss(hidden, head, targets)
        fused.backward()
        val fusedHidden = hidden.grad.copyOf()
        val fusedWeight = head.weight.grad.copyOf()
        hidden.grad.fill(0.0f)
        head.weight.grad.fill(0.0f)

        val reference = crossEntropyLoss(head(hidden), targets)
        reference.backward()

        assertEquals(reference.data[0], fused.data[0], 1e-5f)
        assertClose(hidden.grad, fusedHidden, 1e-6f, "hidden gradient")
        assertClose(head.weight.grad, fusedWeight, 1e-6f, "weight gradient")
    }

    @Test
    fun testTransformerLossMatchesForwardAndKeepsTheBaseFrozen() {
        val model = seededModel(5)
        val adapter = LoRAAdapter(model, rank = 2)
        seedAdapter(adapter, 6)
        val input = Tensor(FloatArray(10) { i -> ((i * 3) % 11).toFloat() }, intArrayOf(2, 5))
        val targets = Tensor(floatArrayOf(-100f, -100f, 4f, 9f, 1f, -100f, 2f, 2f, 10f, 0f), intArrayOf(2, 5))

        for (simd in vectorModes()) {
            withVectorApi(simd) {
                val fused = model.loss(input, targets)
                fused.backward()
                val fusedGrads = adapter.parameters().map { it.grad.copyOf() }
                adapter.parameters().forEach { it.grad.fill(0.0f) }

                val reference = crossEntropyLoss(model.forward(input), targets)
                reference.backward()

                assertEquals(reference.data[0], fused.data[0], 1e-5f, "loss (simd=$simd)")
                for ((p, expected) in adapter.parameters().zip(fusedGrads)) {
                    assertClose(p.grad, expected, 1e-5f, "adapter gradient (simd=$simd)")
                }
                adapter.parameters().forEach { it.grad.fill(0.0f) }
            }
        }

        for ((name, tensor) in model.namedParameters()) {
            assertFalse(tensor.requiresGrad, "$name is frozen while the adapter is attached")
            assertFalse(tensor.hasGrad, "$name never allocates a gradient")
        }
        adapter.merge()
        assertTrue(model.namedParameters().values.all { it.requiresGrad }, "merge unfreezes the base weights")
    }

    @Test
    fun testFreezingLeavesAdapterGradientsUnchanged() {
        val frozenModel = seededModel(7)
        val trainableModel = seededModel(7)
        val frozenAdapter = LoRAAdapter(frozenModel, rank = 3)
        val trainableAdapter = LoRAAdapter(trainableModel, rank = 3)
        seedAdapter(frozenAdapter, 8)
        seedAdapter(trainableAdapter, 8)
        trainableModel.namedParameters().values.forEach { it.requiresGrad = true }

        val inputs = intArrayOf(1, 4, 2, 8, 5, 7)
        val targets = intArrayOf(4, 2, 8, 5, 7, 3)
        crossEntropyLoss(frozenModel.forward(inputs), targets).backward()
        crossEntropyLoss(trainableModel.forward(inputs), targets).backward()

        for ((frozen, trainable) in frozenAdapter.parameters().zip(trainableAdapter.parameters())) {
            assertClose(trainable.grad, frozen.grad, 1e-6f, "adapter gradient")
        }
        assertTrue(trainableModel.blocks[0].attention.wK.parameters()[0].hasGrad)
        assertFalse(frozenModel.blocks[0].attention.wK.parameters()[0].hasGrad)
    }

    @Test
    fun testBackwardStopsAtFrozenSubgraphs() {
        val frozen = Tensor(floatArrayOf(1f, 2f), intArrayOf(1, 2)).also { it.requiresGrad = false }
        val trainable = Tensor(floatArrayOf(3f, 4f), intArrayOf(1, 2))
        var frozenBranchVisited = false
        var trainableBranchVisited = false
        val frozenBranch = Tensor(frozen.data.copyOf(), intArrayOf(1, 2), listOf(frozen)) { frozenBranchVisited = true }
        val trainableBranch = Tensor(trainable.data.copyOf(), intArrayOf(1, 2), listOf(trainable)) { trainableBranchVisited = true }
        assertFalse(frozenBranch.requiresGrad, "outputs of frozen inputs do not require gradients")
        assertTrue(trainableBranch.requiresGrad)

        (frozenBranch + trainableBranch).backward()
        assertFalse(frozenBranchVisited, "backward never runs a frozen subgraph")
        assertTrue(trainableBranchVisited)
        assertFalse(frozenBranch.hasGrad)
        assertFalse(frozen.hasGrad)

        val a = Tensor(floatArrayOf(1f, 2f, 3f, 4f), intArrayOf(2, 2))
        val b = Tensor(floatArrayOf(5f, 6f, 7f, 8f), intArrayOf(2, 2)).also { it.requiresGrad = false }
        (a matmul b).backward()
        assertFalse(b.hasGrad, "a frozen matmul operand gets no gradient")
        assertEquals(listOf(11f, 15f, 11f, 15f), a.grad.toList())

        val model = seededModel(9, numLayers = 3)
        LoRAAdapter(model, rank = 2, blockIndices = listOf(2))
        val embedded = model.embedding.forward(intArrayOf(1, 2, 3))
        val lower = model.blocks[1].forward(model.blocks[0].forward(embedded, 3), 3)
        assertFalse(lower.requiresGrad, "blocks below the lowest adapter are left out of the backward pass")
        assertTrue(model.blocks[2].forward(lower, 3).requiresGrad)
        assertFailsWith<IllegalArgumentException> { LoRAAdapter(seededModel(9), blockIndices = listOf(5)) }
    }

    @Test
    fun testModuleFlagsAndOptimizersRespectFrozenParameters() {
        val linear = Linear(3, 2)
        assertTrue(linear.requiresGrad)
        linear.requiresGrad = false
        assertFalse(linear.weight.requiresGrad)
        assertFalse(linear.bias.requiresGrad)
        assertFalse(linear.requiresGrad)

        val before = linear.weight.data.copyOf()
        val adamW = AdamW(linear.parameters(), learningRate = 0.1f, weightDecay = 0.5f)
        adamW.step()
        adamW.zeroGrad()
        SGD(linear.parameters(), learningRate = 0.1f).step()
        assertClose(before, linear.weight.data, 0.0f, "frozen weights")
        assertFalse(linear.weight.hasGrad)

        linear.requiresGrad = true
        assertTrue(linear.requiresGrad)
        val loss = mseLoss(linear(Tensor(floatArrayOf(1f, 2f, 3f), intArrayOf(1, 3))), Tensor(floatArrayOf(1f, -1f), intArrayOf(1, 2)))
        loss.backward()
        assertTrue(linear.weight.grad.any { it != 0.0f })
    }

    @Test
    fun testTransposedKernelsMatchNaive() {
        val random = Random(10)
        val shapes = listOf(Triple(1, 1, 1), Triple(1, 37, 300), Triple(5, 70, 3), Triple(33, 65, 257), Triple(70, 9, 513), Triple(4, 1500, 6), Triple(40, 2100, 33))
        val savedThreshold = MatMul.parallelThreshold
        try {
            for (simd in vectorModes()) {
                withVectorApi(simd) {
                    for ((m, k, n) in shapes) {
                        val a = FloatArray(m * k) { random.nextFloat() - 0.5f }
                        val b = FloatArray(k * n) { random.nextFloat() - 0.5f }
                        val expected = naive(a, b, m, k, n)
                        val tolerance = 2e-4f * k
                        val label = "[$m,$k]×[$k,$n] simd=$simd"

                        val aT = transposed(a, m, k)
                        val bT = transposed(b, k, n)
                        MatMul.parallelThreshold = 0L
                        val left = FloatArray(m * n)
                        MatMul.multiplyTransposedA(aT, b, left, m, k, n, accumulate = false)
                        assertClose(expected, left, tolerance, "Aᵀ·B $label")
                        val right = FloatArray(m * n)
                        MatMul.multiplyTransposedB(a, bT, right, m, k, n, accumulate = false)
                        assertClose(expected, right, tolerance, "A·Bᵀ $label")

                        MatMul.parallelThreshold = Long.MAX_VALUE
                        val serial = FloatArray(m * n)
                        MatMul.multiplyTransposedB(a, bT, serial, m, k, n, accumulate = false)
                        assertClose(right, serial, 0.0f, "serial vs parallel $label")

                        val accumulated = FloatArray(m * n) { 1.0f }
                        MatMul.multiplyTransposedB(a, bT, accumulated, m, k, n, accumulate = true, alpha = -0.5f)
                        val accumulatedLeft = FloatArray(m * n) { 1.0f }
                        MatMul.multiplyTransposedA(aT, b, accumulatedLeft, m, k, n, accumulate = true, alpha = -0.5f)
                        for (i in expected.indices) {
                            assertEquals(1.0f - 0.5f * expected[i], accumulated[i], tolerance, "accumulate A·Bᵀ $label element $i")
                            assertEquals(1.0f - 0.5f * expected[i], accumulatedLeft[i], tolerance, "accumulate Aᵀ·B $label element $i")
                        }
                    }
                }
            }
        } finally {
            MatMul.parallelThreshold = savedThreshold
        }
    }

    @Test
    fun testVectorApiBackendAgreesWithScalarKernels() {
        if (!MatMul.vectorApiAvailable) {
            assertFalse(MatMul.useVectorApi)
            assertFailsWith<IllegalArgumentException> { MatMul.useVectorApi = true }
            return
        }
        val random = Random(11)
        val m = 19
        val k = 203
        val n = 517
        val a = FloatArray(m * k) { random.nextFloat() - 0.5f }
        val b = FloatArray(k * n) { random.nextFloat() - 0.5f }
        val quantized = QuantizedTensor.quantize(b, k, n, Quantization.INT8)
        val dy = FloatArray(m * n) { random.nextFloat() - 0.5f }

        fun run(simd: Boolean): List<FloatArray> = withVectorApi(simd) {
            val product = FloatArray(m * n).also { MatMul.multiply(a, b, it, m, k, n, accumulate = false) }
            val row = FloatArray(n).also { MatMul.multiply(a, b, it, 1, k, n, accumulate = false) }
            val dequant = FloatArray(m * n).also { MatMul.multiplyQuantized(a, quantized, it, m, accumulate = false) }
            val backward = FloatArray(m * k).also { MatMul.multiplyByQuantizedTransposed(dy, quantized, it, m, accumulate = false) }
            val transposedRight = FloatArray(m * k).also { MatMul.multiplyTransposedB(dy, b, it, m, n, k, accumulate = false) }
            listOf(product, row, dequant, backward, transposedRight)
        }

        val scalar = run(false)
        val vector = run(true)
        val labels = listOf("multiply", "single row", "dequantizing multiply", "quantized transposed", "transposed B")
        for (i in scalar.indices) {
            assertClose(scalar[i], vector[i], 1e-3f, labels[i])
        }
    }

    @Test
    fun testDataLoaderTakesTargetsFromLabels() {
        val tokens = intArrayOf(10, 11, 12, 13, 14, 15, 16, 17, 18)
        val labels = intArrayOf(IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 13, 14, IGNORE_INDEX, 16, 17, 18)
        val loader = DataLoader(tokens, batchSize = 2, sequenceLength = 4, labels = labels)
        val (x, y) = loader.nextBatch()
        assertEquals(listOf(10f, 11f, 12f, 13f, 14f, 15f, 16f, 17f), x.data.toList())
        assertEquals(listOf(-100f, -100f, 13f, 14f, -100f, 16f, 17f, 18f), y.data.toList())
        assertFailsWith<IllegalArgumentException> { DataLoader(tokens, 1, 4, labels = IntArray(3)) }

        val prompted = DataLoader.promptMasked(listOf(intArrayOf(1, 2, 3) to intArrayOf(4, 5), intArrayOf(6) to intArrayOf(7, 8)), batchSize = 1, sequenceLength = 7)
        assertEquals(listOf(1, 2, 3, 4, 5, 6, 7, 8), prompted.tokens.toList())
        assertEquals(listOf(-100, -100, -100, 4, 5, -100, 7, 8), prompted.labels.toList())
        val (_, targets) = prompted.nextBatch()
        assertEquals(listOf(-100f, -100f, 4f, 5f, -100f, 7f, 8f), targets.data.toList())
    }

    @Test
    fun testPipelineReportsEveryBatchAndLearnsMaskedResponses() {
        val model = Transformer(vocabularySize = 12, dimension = 16, numLayers = 2, numHeads = 4, numKeyValueHeads = 2)
        val adapter = LoRAAdapter(model, rank = 4, alpha = 4.0f)
        val pipeline = FineTuningPipeline(model, adapter, AdamW(adapter.parameters(), learningRate = 0.05f, weightDecay = 0.0f))
        val examples = List(6) { i -> intArrayOf(1, 2 + i % 3) to intArrayOf(8 + i % 3, 11) }
        val loader = DataLoader.promptMasked(examples, batchSize = 1, sequenceLength = 4)

        val reports = mutableListOf<IntArray>()
        val epochLosses = mutableListOf<Float>()
        pipeline.train(
            loader,
            epochs = 30,
            onBatchEnd = { epoch, batch, totalBatches, loss ->
                assertTrue(loss.isFinite())
                reports.add(intArrayOf(epoch, batch, totalBatches))
            },
            onEpochEnd = { _, averageLoss -> epochLosses.add(averageLoss) }
        )

        assertEquals(30 * loader.totalBatches, reports.size)
        assertEquals(listOf(1, 1, loader.totalBatches), reports.first().toList())
        assertEquals(listOf(30, loader.totalBatches, loader.totalBatches), reports.last().toList())
        assertEquals(30, epochLosses.size)
        assertTrue(epochLosses.last() < epochLosses.first() * 0.8f, "loss ${epochLosses.first()} → ${epochLosses.last()}")
        assertTrue(model.namedParameters().values.none { it.hasGrad }, "the frozen base never holds gradients")

        var called = 0
        pipeline.train(loader, epochs = 1) { _, _ -> called++ }
        assertEquals(1, called)

        val before = adapter.parameters().map { it.data.copyOf() }
        val silent = Tensor(FloatArray(4) { -100f }, intArrayOf(1, 4))
        assertEquals(0.0f, pipeline.trainStep(Tensor(FloatArray(4) { 1f }, intArrayOf(1, 4)), silent))
        for ((p, saved) in adapter.parameters().zip(before)) {
            assertClose(saved, p.data, 0.0f, "a batch without targets leaves the adapter unchanged")
        }
    }
}
