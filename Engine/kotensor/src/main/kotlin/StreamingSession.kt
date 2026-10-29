package io.kotensor.core

data class GenerationState(
    val prompt: String = "",
    val generatedText: String = "",
    val isGenerating: Boolean = false,
    val tokensGenerated: Int = 0,
    val tokensPerSecond: Float = 0.0f
)

class StreamingSession(
    val model: Transformer,
    val tokenizer: Tokenizer
) {
    @Volatile
    private var isCancelled = false

    fun cancel() {
        isCancelled = true
    }

    fun stream(
        prompt: String,
        maxTokens: Int,
        onUpdate: (GenerationState) -> Unit
    ): GenerationState {
        isCancelled = false
        val startTime = System.currentTimeMillis()
        var currentText = ""
        var count = 0

        onUpdate(GenerationState(prompt = prompt, generatedText = "", isGenerating = true, tokensGenerated = 0, tokensPerSecond = 0.0f))

        val encoded = tokenizer.encode(prompt)
        for (token in model.generateStream(encoded, maxTokens)) {
            if (isCancelled) break

            val char = tokenizer.decode(intArrayOf(token))
            currentText += char
            count++

            val elapsedSec = (System.currentTimeMillis() - startTime) / 1000.0f
            val tps = if (elapsedSec > 0.001f) count.toFloat() / elapsedSec else 0.0f

            onUpdate(
                GenerationState(
                    prompt = prompt,
                    generatedText = currentText,
                    isGenerating = true,
                    tokensGenerated = count,
                    tokensPerSecond = tps
                )
            )
        }

        val totalSec = (System.currentTimeMillis() - startTime) / 1000.0f
        val finalTps = if (totalSec > 0.001f) count.toFloat() / totalSec else 0.0f
        val finalState = GenerationState(
            prompt = prompt,
            generatedText = currentText,
            isGenerating = false,
            tokensGenerated = count,
            tokensPerSecond = finalTps
        )
        onUpdate(finalState)
        return finalState
    }
}
