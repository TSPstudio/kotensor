package io.kotensor.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class RegressionTest {

    @Test
    fun testBatchedSequencesAreIndependent() {
        val model = Transformer(vocabularySize = 8, dimension = 16, numLayers = 2, numHeads = 2)
        val first = intArrayOf(1, 2, 3, 4)
        val second = intArrayOf(5, 6, 7, 0)

        val batchData = FloatArray(8) { i -> (if (i < 4) first[i] else second[i - 4]).toFloat() }
        val batched = model.forward(Tensor(batchData, intArrayOf(2, 4)))

        val alone1 = model.forward(first)
        val alone2 = model.forward(second)
        val half = alone1.data.size

        assertEquals(half * 2, batched.data.size)
        for (i in 0 until half) {
            assertEquals(alone1.data[i], batched.data[i], 1e-5f)
            assertEquals(alone2.data[i], batched.data[half + i], 1e-5f)
        }
    }

    @Test
    fun testLoRAMergeAppliesDeltaOnce() {
        val model = Transformer(vocabularySize = 8, dimension = 16, numLayers = 1, numHeads = 2)
        val adapter = LoRAAdapter(model, rank = 2, alpha = 1.0f)
        val optimizer = AdamW(adapter.parameters(), learningRate = 0.05f)
        val inputs = intArrayOf(0, 1, 2, 3)
        val targets = intArrayOf(1, 2, 3, 4)

        repeat(5) {
            val loss = crossEntropyLoss(model.forward(inputs), targets)
            loss.backward()
            optimizer.step()
            optimizer.zeroGrad()
        }

        val before = model.forward(inputs).data.copyOf()
        adapter.merge()
        val after = model.forward(inputs).data

        assertTrue(model.blocks[0].attention.wQ is Linear)
        assertTrue(model.blocks[0].attention.wV is Linear)
        assertTrue(adapter.parameters().isEmpty())
        for (i in before.indices) {
            assertEquals(before[i], after[i], 1e-4f)
        }
    }

    @Test
    fun testNamedParametersIncludeLoRAWrappedProjections() {
        val model = Transformer(vocabularySize = 8, dimension = 16, numLayers = 1, numHeads = 2)
        val namesBefore = model.namedParameters().keys
        LoRAAdapter(model, rank = 2)
        val namesAfter = model.namedParameters().keys

        assertEquals(namesBefore, namesAfter)
        assertTrue(namesAfter.contains("block.0.attention.wQ.weight"))
        assertTrue(namesAfter.contains("block.0.attention.wV.weight"))
    }
}
