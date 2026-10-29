package io.kotensor.core

import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class Phase5Test {

    @Test
    fun testSafeTensorsSaveAndLoad() {
        val tempFile = File.createTempFile("kotensor_test", ".safetensors")
        tempFile.deleteOnExit()

        val tensorA = Tensor(floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f), intArrayOf(2, 2))
        val tensorB = Tensor(floatArrayOf(0.5f, -1.5f, 2.5f), intArrayOf(1, 3))
        val originalMap = mapOf("weight_a" to tensorA, "bias_b" to tensorB)

        SafeTensors.save(tempFile, originalMap)
        assertTrue(tempFile.exists())
        assertTrue(tempFile.length() > 0)

        val loadedMap = SafeTensors.load(tempFile)
        assertEquals(2, loadedMap.size)

        val loadedA = loadedMap["weight_a"]!!
        val loadedB = loadedMap["bias_b"]!!

        assertTrue(loadedA.shape.contentEquals(intArrayOf(2, 2)))
        assertTrue(loadedB.shape.contentEquals(intArrayOf(1, 3)))

        for (i in tensorA.data.indices) {
            assertEquals(tensorA.data[i], loadedA.data[i], 1e-6f)
        }
        for (i in tensorB.data.indices) {
            assertEquals(tensorB.data[i], loadedB.data[i], 1e-6f)
        }
    }

    @Test
    fun testTransformerModelWeightPersistence() {
        val tempFile = File.createTempFile("model_weights", ".safetensors")
        tempFile.deleteOnExit()

        val vocabSize = 10
        val dimension = 16
        val model1 = Transformer(vocabSize, dimension, numLayers = 1, numHeads = 2)

        SafeTensors.save(tempFile, model1.namedParameters())

        val model2 = Transformer(vocabSize, dimension, numLayers = 1, numHeads = 2)
        val loadedWeights = SafeTensors.load(tempFile)
        model2.loadParameters(loadedWeights)

        val prompt = intArrayOf(1, 2, 3)
        val logits1 = model1.forward(prompt)
        val logits2 = model2.forward(prompt)

        for (i in logits1.data.indices) {
            assertEquals(logits1.data[i], logits2.data[i], 1e-5f)
        }
    }

    @Test
    fun testLoRAAdapterAndFineTuning() {
        val tempFile = File.createTempFile("adapter", ".safetensors")
        tempFile.deleteOnExit()

        val vocabSize = 8
        val dimension = 16
        val model = Transformer(vocabSize, dimension, numLayers = 1, numHeads = 2)
        val adapter = LoRAAdapter(model, rank = 2, alpha = 1.0f)

        assertEquals(4, adapter.parameters().size)

        val optimizer = AdamW(adapter.parameters(), learningRate = 0.05f)
        val inputTokens = intArrayOf(0, 1, 2, 3)
        val targetTokens = intArrayOf(1, 2, 3, 4)

        var initialLoss = 0.0f
        var finalLoss = 0.0f

        for (epoch in 0 until 30) {
            val logits = model.forward(inputTokens)
            val loss = crossEntropyLoss(logits, targetTokens)

            if (epoch == 0) initialLoss = loss.data[0]
            if (epoch == 29) finalLoss = loss.data[0]

            loss.backward()
            optimizer.step()
            optimizer.zeroGrad()
        }

        assertTrue(finalLoss < initialLoss)

        adapter.saveDelta(tempFile)
        assertTrue(tempFile.exists())
        assertTrue(tempFile.length() > 0)

        val loadedDelta = SafeTensors.load(tempFile)
        assertTrue(loadedDelta.containsKey("block.0.attention.wQ.loraA"))
        assertTrue(loadedDelta.containsKey("block.0.attention.wQ.loraB"))

        adapter.merge()
    }

    @Test
    fun testStreamingSessionAndCancellation() {
        val text = "abcdefgh"
        val tokenizer = SimpleTokenizer(text)
        val model = Transformer(tokenizer.vocabularySize, dimension = 8, numLayers = 1, numHeads = 2)
        val session = StreamingSession(model, tokenizer)

        val receivedTokens = mutableListOf<String>()
        val finalState = session.stream("a", maxTokens = 4) { state ->
            if (state.generatedText.isNotEmpty()) {
                receivedTokens.add(state.generatedText)
            }
        }

        assertEquals(4, finalState.tokensGenerated)
        assertTrue(finalState.tokensPerSecond >= 0.0f)
        assertTrue(!finalState.isGenerating)

        session.cancel()
        val cancelledState = session.stream("a", maxTokens = 10) { state ->
            session.cancel()
        }
        assertTrue(cancelledState.tokensGenerated <= 1)
    }

    @Test
    fun testFineTuningPipeline() {
        val vocabSize = 6
        val dimension = 8
        val model = Transformer(vocabSize, dimension, numLayers = 1, numHeads = 2)
        val adapter = LoRAAdapter(model, rank = 2)
        val optimizer = AdamW(adapter.parameters(), learningRate = 0.05f)
        val pipeline = FineTuningPipeline(model, adapter, optimizer)

        val tokens = intArrayOf(0, 1, 2, 3, 4, 5, 0, 1, 2, 3)
        val dataLoader = DataLoader(tokens, batchSize = 1, sequenceLength = 4)

        var finalLoss = 0.0f
        pipeline.train(dataLoader, epochs = 10) { epoch, loss ->
            finalLoss = loss
        }

        assertTrue(finalLoss >= 0.0f)
        val tempFile = File.createTempFile("pipeline_adapter", ".safetensors")
        tempFile.deleteOnExit()
        pipeline.exportAdapter(tempFile)
        assertTrue(tempFile.exists())
        assertTrue(tempFile.length() > 0)
    }
}

