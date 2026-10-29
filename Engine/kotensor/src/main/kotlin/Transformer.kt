package io.kotensor.core

class Transformer(
    val vocabularySize: Int,
    val dimension: Int,
    val numLayers: Int,
    val numHeads: Int,
    val hiddenDimension: Int = 4 * dimension
) : Module() {
    val embedding = Embedding(vocabularySize, dimension)
    val blocks = List(numLayers) {
        TransformerBlock(dimension, numHeads, hiddenDimension)
    }
    val finalNorm = RMSNorm(dimension)
    val lmHead = Linear(dimension, vocabularySize)

    /**
     * Token IDs as a tensor. A 2-D input is read as [batch, sequenceLength] (the shape
     * DataLoader produces) and each row is an independent sequence; any other shape is
     * treated as one sequence. Returns logits shaped [batch × sequenceLength, vocabularySize].
     */
    override fun forward(input: Tensor): Tensor {
        val sequenceLength = if (input.shape.size == 2) input.shape[1] else input.data.size
        return forward(input, sequenceLength)
    }

    /** Token IDs as a tensor, split into independent sequences of [sequenceLength] tokens. */
    fun forward(input: Tensor, sequenceLength: Int): Tensor {
        return forwardEmbedded(embedding(input), sequenceLength)
    }

    /** A single sequence of token IDs. Returns logits shaped [indices.size, vocabularySize]. */
    fun forward(indices: IntArray): Tensor {
        return forwardEmbedded(embedding.forward(indices), indices.size)
    }

    private fun forwardEmbedded(embedded: Tensor, sequenceLength: Int): Tensor {
        var x = embedded
        for (block in blocks) {
            x = block.forward(x, sequenceLength)
        }
        val normed = finalNorm(x)
        return lmHead(normed)
    }

    override fun parameters(): List<Tensor> {
        val params = mutableListOf<Tensor>()
        params.addAll(embedding.parameters())
        for (block in blocks) {
            params.addAll(block.parameters())
        }
        params.addAll(finalNorm.parameters())
        params.addAll(lmHead.parameters())
        return params
    }

    fun generate(prompt: IntArray, maxTokens: Int): IntArray {
        require(prompt.isNotEmpty()) { "Prompt cannot be empty" }
        var tokens = prompt.copyOf()
        for (step in 0 until maxTokens) {
            val logits = forward(tokens)
            val count = tokens.size
            val lastTokenOffset = (count - 1) * vocabularySize
            var bestIndex = 0
            var maxLogit = -1e9f
            for (v in 0 until vocabularySize) {
                val logit = logits.data[lastTokenOffset + v]
                if (logit > maxLogit) {
                    maxLogit = logit
                    bestIndex = v
                }
            }
            tokens = tokens + bestIndex
        }
        return tokens
    }

    fun generate(prompt: String, tokenizer: Tokenizer, maxTokens: Int): String {
        val encodedPrompt = tokenizer.encode(prompt)
        val generatedTokens = generate(encodedPrompt, maxTokens)
        return tokenizer.decode(generatedTokens)
    }

    fun generateStream(prompt: IntArray, maxTokens: Int): Sequence<Int> = sequence {
        require(prompt.isNotEmpty()) { "Prompt cannot be empty" }
        var tokens = prompt.copyOf()
        for (step in 0 until maxTokens) {
            val logits = forward(tokens)
            val count = tokens.size
            val lastTokenOffset = (count - 1) * vocabularySize
            var bestIndex = 0
            var maxLogit = -1e9f
            for (v in 0 until vocabularySize) {
                val logit = logits.data[lastTokenOffset + v]
                if (logit > maxLogit) {
                    maxLogit = logit
                    bestIndex = v
                }
            }
            tokens = tokens + bestIndex
            yield(bestIndex)
        }
    }

    fun generateStream(prompt: String, tokenizer: Tokenizer, maxTokens: Int, onToken: (String) -> Unit) {
        val encodedPrompt = tokenizer.encode(prompt)
        for (token in generateStream(encodedPrompt, maxTokens)) {
            val tokenString = tokenizer.decode(intArrayOf(token))
            onToken(tokenString)
        }
    }

    fun namedParameters(): Map<String, Tensor> {
        val result = mutableMapOf<String, Tensor>()
        result["embedding.weight"] = embedding.weight

        for (l in blocks.indices) {
            val b = blocks[l]
            result["block.$l.norm1.weight"] = b.norm1.weight
            val q = baseLinearOf(b.attention.wQ)
            if (q != null) {
                result["block.$l.attention.wQ.weight"] = q.weight
                result["block.$l.attention.wQ.bias"] = q.bias
            }
            val k = baseLinearOf(b.attention.wK)
            if (k != null) {
                result["block.$l.attention.wK.weight"] = k.weight
                result["block.$l.attention.wK.bias"] = k.bias
            }
            val v = baseLinearOf(b.attention.wV)
            if (v != null) {
                result["block.$l.attention.wV.weight"] = v.weight
                result["block.$l.attention.wV.bias"] = v.bias
            }
            val o = baseLinearOf(b.attention.wOut)
            if (o != null) {
                result["block.$l.attention.wOut.weight"] = o.weight
                result["block.$l.attention.wOut.bias"] = o.bias
            }
            result["block.$l.norm2.weight"] = b.norm2.weight
            result["block.$l.ffn.wGate.weight"] = b.ffn.wGate.weight
            result["block.$l.ffn.wGate.bias"] = b.ffn.wGate.bias
            result["block.$l.ffn.wUp.weight"] = b.ffn.wUp.weight
            result["block.$l.ffn.wUp.bias"] = b.ffn.wUp.bias
            result["block.$l.ffn.wDown.weight"] = b.ffn.wDown.weight
            result["block.$l.ffn.wDown.bias"] = b.ffn.wDown.bias
        }

        result["finalNorm.weight"] = finalNorm.weight
        result["lmHead.weight"] = lmHead.weight
        result["lmHead.bias"] = lmHead.bias
        return result
    }

    /** The underlying Linear of a projection, looking through a LoRA wrapper (whose A/B matrices are saved separately). */
    private fun baseLinearOf(module: Module): Linear? = when (module) {
        is Linear -> module
        is LoRALinear -> module.baseLinear
        else -> null
    }

    fun loadParameters(weights: Map<String, Tensor>) {
        val current = namedParameters()
        for ((name, tensor) in current) {
            val source = weights[name] ?: continue
            require(tensor.data.size == source.data.size)
            for (i in tensor.data.indices) {
                tensor.data[i] = source.data[i]
            }
        }
    }
}
