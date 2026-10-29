package io.kotensor.sample

import io.kotensor.core.AdamW
import io.kotensor.core.BPETokenizer
import io.kotensor.core.DataLoader
import io.kotensor.core.HuggingFaceModelLoader
import io.kotensor.core.Quantization
import io.kotensor.core.SimpleTokenizer
import io.kotensor.core.Tokenizer
import io.kotensor.core.Transformer
import io.kotensor.core.crossEntropyLoss
import java.io.File

class LoadedModel(
    val model: Transformer,
    val tokenizer: Tokenizer,
    val description: String,
    val stopTokens: Set<Int>
)

object ModelSource {
    const val DEMO_STEPS = 160
    val IMPORT_QUANTIZATION: Quantization = Quantization.INT8

    val DEMO_TEXT = """
        the engine runs on the phone. the model learns on the phone too.
        kotensor keeps every weight in plain kotlin arrays and trains them with autograd.
        a tiny model like this one learns the shape of its text in a few seconds.
        the cache remembers every token so the next one costs almost nothing.
        quantize the weights to eight or four bits and the model shrinks in place.
        attach an adapter, fine tune on new text, and the words change their course.
        the engine runs on the phone. the words flow one token at a time.
        every layer is a few loops over float arrays, and every loop has a backward pass.
        small models learn fast; large models stream fast; both fit in the same jar.
    """.trimIndent().replace("\n", " ") + " "

    fun fromFiles(directory: File, quantization: Quantization?, onProgress: (String) -> Unit): LoadedModel? {
        if (!directory.isDirectory) return null
        val quantized = File(directory, "kotensor-quantized.safetensors")
        val tokenizerFile = File(directory, "tokenizer.json")
        if (quantized.isFile && tokenizerFile.isFile) {
            onProgress("Loading ${quantized.name}")
            val model = Transformer.loadQuantized(quantized)
            val tokenizer = BPETokenizer.fromFile(tokenizerFile)
            return LoadedModel(model, tokenizer, "${quantized.name} (${model.quantization})", setOfNotNull(tokenizer.endOfTextId))
        }
        if (File(directory, "config.json").isFile && tokenizerFile.isFile) {
            val model = HuggingFaceModelLoader.load(directory, quantization, onProgress)
            val tokenizer = HuggingFaceModelLoader.loadTokenizer(directory)
            val config = HuggingFaceModelLoader.loadConfig(directory)
            val label = "${config.modelType} ${config.numHiddenLayers}×${config.hiddenSize}" + (quantization?.let { " ($it)" } ?: " (fp32)")
            return LoadedModel(model, tokenizer, label, setOfNotNull(tokenizer.endOfTextId))
        }
        return null
    }

    fun trainDemo(onProgress: (Int, Int, Float) -> Unit): LoadedModel {
        val tokenizer = SimpleTokenizer(DEMO_TEXT)
        val model = Transformer(
            vocabularySize = tokenizer.vocabularySize,
            dimension = 48,
            numLayers = 2,
            numHeads = 4,
            hiddenDimension = 128
        )
        val optimizer = AdamW(model.parameters(), learningRate = 3e-3f)
        val loader = DataLoader(tokenizer.encode(DEMO_TEXT), batchSize = 4, sequenceLength = 32)
        for (step in 1..DEMO_STEPS) {
            val (inputs, targets) = loader.nextBatch()
            val loss = crossEntropyLoss(model.forward(inputs), targets)
            loss.backward()
            optimizer.step()
            optimizer.zeroGrad()
            onProgress(step, DEMO_STEPS, loss.data[0])
        }
        return LoadedModel(model, tokenizer, "demo char model trained on device (${model.parameterCount()} params)", emptySet())
    }
}
