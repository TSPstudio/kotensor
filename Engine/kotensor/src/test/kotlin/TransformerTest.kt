package io.kotensor.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class TransformerTest {

    @Test
    fun testTransformerBlockForwardBackward() {
        val dimension = 16
        val numHeads = 2
        val tokens = 4
        val block = TransformerBlock(dimension, numHeads)

        val inputData = FloatArray(tokens * dimension) { i -> (i % 5).toFloat() * 0.1f }
        val input = Tensor(inputData, intArrayOf(tokens, dimension))

        val output = block(input)
        assertEquals(tokens * dimension, output.data.size)
        assertTrue(output.data.all { !it.isNaN() && !it.isInfinite() })

        val loss = output.data.sum()
        for (i in output.grad.indices) {
            output.grad[i] = 1.0f
        }
        output.backward()

        assertTrue(input.grad.any { it != 0.0f })
    }

    @Test
    fun testCrossEntropyLoss() {
        val count = 2
        val vocabSize = 3
        val logitsData = floatArrayOf(
            2.0f, 1.0f, 0.1f,
            0.5f, 2.5f, 0.3f
        )
        val logits = Tensor(logitsData, intArrayOf(count, vocabSize))
        val targets = intArrayOf(0, 1)

        val loss = crossEntropyLoss(logits, targets)
        assertTrue(loss.data[0] > 0.0f)

        loss.backward()
        assertTrue(logits.grad.all { !it.isNaN() })
        assertTrue(logits.grad[0] < 0.0f)
        assertTrue(logits.grad[4] < 0.0f)
    }

    @Test
    fun testTransformerEndToEndTraining() {
        val text = "hello hello hello "
        val tokenizer = SimpleTokenizer(text)
        val vocabSize = tokenizer.vocabularySize
        val dimension = 16
        val numHeads = 2
        val numLayers = 1

        val model = Transformer(vocabSize, dimension, numLayers, numHeads)
        val optimizer = AdamW(model.parameters(), learningRate = 0.05f)

        val encoded = tokenizer.encode(text)
        val inputTokens = encoded.sliceArray(0 until encoded.size - 1)
        val targetTokens = encoded.sliceArray(1 until encoded.size)

        var initialLoss = 0.0f
        var finalLoss = 0.0f

        for (epoch in 0 until 40) {
            val logits = model.forward(inputTokens)
            val loss = crossEntropyLoss(logits, targetTokens)

            if (epoch == 0) initialLoss = loss.data[0]
            if (epoch == 39) finalLoss = loss.data[0]

            loss.backward()
            optimizer.step()
            optimizer.zeroGrad()
        }

        assertTrue(finalLoss < initialLoss)

        val generated = model.generate("h", tokenizer, 5)
        assertEquals(6, generated.length)
    }
}

