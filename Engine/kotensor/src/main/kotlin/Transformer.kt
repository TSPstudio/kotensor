package io.kotensor.core

class Transformer(
    val vocabularySize: Int,
    val dimension: Int,
    val numLayers: Int,
    val numHeads: Int,
    val hiddenDimension: Int = 4 * dimension,
    val numKeyValueHeads: Int = numHeads,
    val normEps: Float = 1e-6f,
    val ropeTheta: Float = 10000.0f,
    val ropeScaling: RoPEScaling? = null,
    quantization: Quantization? = null
) : Module() {
    val headDim = dimension / numHeads
    var embedding: TokenEmbedding = if (quantization == null) Embedding(vocabularySize, dimension) else QuantizedEmbedding(vocabularySize, dimension, quantization)
    val blocks = List(numLayers) {
        TransformerBlock(dimension, numHeads, hiddenDimension, numKeyValueHeads, normEps, ropeTheta, ropeScaling, quantization)
    }
    val finalNorm = RMSNorm(dimension, normEps)
    var lmHead: Module = projection(dimension, vocabularySize, quantization)

    override fun forward(input: Tensor): Tensor {
        val sequenceLength = if (input.shape.size == 2) input.shape[1] else input.data.size
        return forward(input, sequenceLength)
    }

    fun forward(input: Tensor, sequenceLength: Int): Tensor {
        return forwardEmbedded(embedding(input), sequenceLength)
    }

    fun forward(indices: IntArray): Tensor {
        return forwardEmbedded(embedding.forward(indices), indices.size)
    }

    private fun forwardEmbedded(embedded: Tensor, sequenceLength: Int): Tensor {
        return lmHead(encode(embedded, sequenceLength))
    }

    private fun encode(embedded: Tensor, sequenceLength: Int): Tensor {
        var x = embedded
        for (block in blocks) {
            x = block.forward(x, sequenceLength)
        }
        return finalNorm(x)
    }

    fun hiddenStates(input: Tensor): Tensor {
        val sequenceLength = if (input.shape.size == 2) input.shape[1] else input.data.size
        return encode(embedding(input), sequenceLength)
    }

    fun hiddenStates(indices: IntArray): Tensor {
        return encode(embedding.forward(indices), indices.size)
    }

    @JvmOverloads
    fun loss(input: Tensor, targets: Tensor, ignoreIndex: Int = IGNORE_INDEX): Tensor {
        return lossFromHidden(hiddenStates(input), targetIndices(targets), ignoreIndex)
    }

    @JvmOverloads
    fun loss(indices: IntArray, targets: IntArray, ignoreIndex: Int = IGNORE_INDEX): Tensor {
        return lossFromHidden(hiddenStates(indices), targets, ignoreIndex)
    }

    private fun lossFromHidden(hidden: Tensor, targets: IntArray, ignoreIndex: Int): Tensor {
        val head = lmHead
        return if (head is Linear || head is QuantizedLinear) {
            fusedCrossEntropyLoss(hidden, head, targets, ignoreIndex)
        } else {
            crossEntropyLoss(head(hidden), targets, ignoreIndex)
        }
    }

    fun newCache(maxSequenceLength: Int): KVCache {
        return KVCache(numLayers, numKeyValueHeads, maxSequenceLength, headDim)
    }

    fun forward(indices: IntArray, cache: KVCache): Tensor {
        require(indices.isNotEmpty()) { "Input cannot be empty" }
        require(cache.numLayers == numLayers) { "Cache built for ${cache.numLayers} layers, model has $numLayers" }
        require(cache.remaining >= indices.size) {
            "KV cache full: ${cache.length} + ${indices.size} positions exceed maxSequenceLength ${cache.maxSequenceLength}"
        }
        var x = embedding.forward(indices)
        for (l in blocks.indices) {
            x = blocks[l].forward(x, cache, l)
        }
        cache.advance(indices.size)
        val normed = finalNorm(x)
        return lmHead(normed)
    }

    fun newInferenceBuffers(maxSequenceLength: Int): InferenceBuffers {
        return InferenceBuffers(dimension, hiddenDimension, vocabularySize, maxSequenceLength, numKeyValueHeads * headDim)
    }

    fun decode(token: Int, cache: KVCache, buffers: InferenceBuffers): Tensor {
        require(cache.numLayers == numLayers) { "Cache built for ${cache.numLayers} layers, model has $numLayers" }
        require(cache.remaining >= 1) { "KV cache full: maxSequenceLength ${cache.maxSequenceLength}" }
        require(buffers.dimension == dimension && buffers.hiddenDimension == hiddenDimension && buffers.vocabularySize == vocabularySize && buffers.kvDimension == numKeyValueHeads * headDim) {
            "InferenceBuffers built for another model configuration"
        }
        val x = embedding.lookupInto(token, buffers.x)
        for (l in blocks.indices) {
            blocks[l].decodeInto(x, cache, l, buffers)
        }
        cache.advance(1)
        finalNorm.forwardInto(x, buffers.normed)
        return lmHead.forwardInto(buffers.normed, buffers.logits)
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

    fun generate(prompt: IntArray, maxTokens: Int, stopTokens: Set<Int> = emptySet()): IntArray {
        require(prompt.isNotEmpty()) { "Prompt cannot be empty" }
        val tokens = IntArray(prompt.size + maxTokens)
        prompt.copyInto(tokens)
        var count = prompt.size
        for (token in generateStream(prompt, maxTokens, stopTokens)) {
            tokens[count] = token
            count++
        }
        return tokens.copyOf(count)
    }

    fun generate(prompt: String, tokenizer: Tokenizer, maxTokens: Int, stopTokens: Set<Int> = emptySet()): String {
        val encodedPrompt = tokenizer.encode(prompt)
        val generatedTokens = generate(encodedPrompt, maxTokens, stopTokens)
        return tokenizer.decode(generatedTokens)
    }

    fun generateStream(prompt: IntArray, maxTokens: Int, stopTokens: Set<Int> = emptySet()): Sequence<Int> = sequence {
        require(prompt.isNotEmpty()) { "Prompt cannot be empty" }
        if (maxTokens <= 0) return@sequence
        val maxSequenceLength = prompt.size + maxTokens
        val cache = newCache(maxSequenceLength)
        val buffers = newInferenceBuffers(maxSequenceLength)
        var logits = forward(prompt, cache)
        for (step in 0 until maxTokens) {
            val bestIndex = argmaxOfLastToken(logits)
            if (bestIndex in stopTokens) break
            yield(bestIndex)
            if (step == maxTokens - 1) break
            logits = decode(bestIndex, cache, buffers)
        }
    }

    fun generateStream(prompt: String, tokenizer: Tokenizer, maxTokens: Int, stopTokens: Set<Int>, onToken: (String) -> Unit) {
        val encodedPrompt = tokenizer.encode(prompt)
        val decoder = IncrementalDecoder(tokenizer)
        for (token in generateStream(encodedPrompt, maxTokens, stopTokens)) {
            val piece = decoder.push(token)
            if (piece.isNotEmpty()) {
                onToken(piece)
            }
        }
        val rest = decoder.flush()
        if (rest.isNotEmpty()) {
            onToken(rest)
        }
    }

    fun generateStream(prompt: String, tokenizer: Tokenizer, maxTokens: Int, onToken: (String) -> Unit) {
        generateStream(prompt, tokenizer, maxTokens, emptySet(), onToken)
    }

    private fun argmaxOfLastToken(logits: Tensor): Int {
        val count = logits.data.size / vocabularySize
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
        return bestIndex
    }

    fun namedProjections(): Map<String, Module> {
        val result = LinkedHashMap<String, Module>()
        for (l in blocks.indices) {
            val b = blocks[l]
            result["block.$l.attention.wQ"] = b.attention.wQ
            result["block.$l.attention.wK"] = b.attention.wK
            result["block.$l.attention.wV"] = b.attention.wV
            result["block.$l.attention.wOut"] = b.attention.wOut
            result["block.$l.ffn.wGate"] = b.ffn.wGate
            result["block.$l.ffn.wUp"] = b.ffn.wUp
            result["block.$l.ffn.wDown"] = b.ffn.wDown
        }
        result["lmHead"] = lmHead
        return result
    }

    private fun replaceProjection(name: String, module: Module) {
        if (name == "lmHead") {
            lmHead = module
            return
        }
        val parts = name.split('.')
        val block = blocks[parts[1].toInt()]
        when (parts[2] + "." + parts[3]) {
            "attention.wQ" -> block.attention.wQ = module
            "attention.wK" -> block.attention.wK = module
            "attention.wV" -> block.attention.wV = module
            "attention.wOut" -> block.attention.wOut = module
            "ffn.wGate" -> block.ffn.wGate = module
            "ffn.wUp" -> block.ffn.wUp = module
            "ffn.wDown" -> block.ffn.wDown = module
            else -> throw IllegalArgumentException("Unknown projection $name")
        }
    }

    fun namedQuantizedLayers(): Map<String, QuantizedLinear> {
        val result = LinkedHashMap<String, QuantizedLinear>()
        for ((name, module) in namedProjections()) {
            val base = when (module) {
                is QuantizedLinear -> module
                is LoRALinear -> module.baseLinear as? QuantizedLinear
                else -> null
            }
            if (base != null) result[name] = base
        }
        return result
    }

    fun namedParameters(): Map<String, Tensor> {
        val result = mutableMapOf<String, Tensor>()
        (embedding as? Embedding)?.let { table -> result["embedding.weight"] = table.weight }
        for (l in blocks.indices) {
            result["block.$l.norm1.weight"] = blocks[l].norm1.weight
            result["block.$l.norm2.weight"] = blocks[l].norm2.weight
        }
        result["finalNorm.weight"] = finalNorm.weight
        for ((name, module) in namedProjections()) {
            when (val base = baseProjectionOf(module)) {
                is Linear -> {
                    result["$name.weight"] = base.weight
                    result["$name.bias"] = base.bias
                }
                is QuantizedLinear -> result["$name.bias"] = base.bias
                else -> {}
            }
        }
        return result
    }

    private fun baseProjectionOf(module: Module): Projection? = when (module) {
        is Projection -> module
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

    fun namedQuantizedTensors(): Map<String, QuantizedTensor> {
        val result = LinkedHashMap<String, QuantizedTensor>()
        (embedding as? QuantizedEmbedding)?.let { table -> result["embedding"] = table.weight }
        for ((name, layer) in namedQuantizedLayers()) result[name] = layer.weight
        return result
    }

    fun quantize(quantization: Quantization): Transformer {
        (embedding as? Embedding)?.let { table -> embedding = QuantizedEmbedding.fromEmbedding(table, quantization) }
        for ((name, module) in namedProjections()) {
            when (module) {
                is Linear -> replaceProjection(name, QuantizedLinear.fromLinear(module, quantization))
                is QuantizedLinear -> {}
                is LoRALinear -> throw IllegalStateException("Merge or remove LoRA adapters before quantizing ($name)")
                else -> throw IllegalStateException("Cannot quantize $name of type ${module::class.simpleName}")
            }
        }
        return this
    }

    val quantization: Quantization?
        get() = namedQuantizedLayers().values.firstOrNull()?.quantization

    fun parameterCount(): Long {
        var count = 0L
        for (p in namedParameters().values) count += p.data.size
        for (tensor in namedQuantizedTensors().values) count += tensor.rows.toLong() * tensor.columns
        return count
    }

    fun weightBytes(): Long {
        var bytes = 0L
        for (p in namedParameters().values) bytes += p.data.size.toLong() * 4L
        for (tensor in namedQuantizedTensors().values) bytes += tensor.sizeInBytes
        return bytes
    }

    fun saveQuantized(file: java.io.File) {
        val entries = ArrayList<SafeTensorsEntry>()
        val metadata = LinkedHashMap<String, String>()
        metadata["kotensor.vocabularySize"] = vocabularySize.toString()
        metadata["kotensor.dimension"] = dimension.toString()
        metadata["kotensor.numLayers"] = numLayers.toString()
        metadata["kotensor.numHeads"] = numHeads.toString()
        metadata["kotensor.hiddenDimension"] = hiddenDimension.toString()
        metadata["kotensor.numKeyValueHeads"] = numKeyValueHeads.toString()
        metadata["kotensor.normEps"] = normEps.toString()
        metadata["kotensor.ropeTheta"] = ropeTheta.toString()
        ropeScaling?.let { scaling ->
            metadata["kotensor.ropeScaling"] = "${scaling.factor},${scaling.lowFrequencyFactor},${scaling.highFrequencyFactor},${scaling.originalMaxPositionEmbeddings}"
        }
        val q = quantization ?: throw IllegalStateException("No quantized layers to save; call quantize() first")
        metadata["kotensor.quantization.bits"] = q.bits.toString()
        metadata["kotensor.quantization.groupSize"] = q.groupSize.toString()
        for ((name, tensor) in namedParameters()) {
            entries.add(SafeTensorsEntry.of(name, tensor))
        }
        for ((name, weight) in namedQuantizedTensors()) {
            require(weight.bits == q.bits && weight.groupSize == q.groupSize) { "$name uses int${weight.bits}/g${weight.groupSize}, file uses $q" }
            entries.add(SafeTensorsEntry("$name.qweight", if (q.bits == 8) "I8" else "U8", intArrayOf(weight.rows, weight.bytesPerRow), weight.qdata))
            entries.add(SafeTensorsEntry.of("$name.scales", Tensor(weight.scales, intArrayOf(weight.groups, weight.columns))))
            weight.zeros?.let { zeros -> entries.add(SafeTensorsEntry("$name.zeros", "U8", intArrayOf(weight.groups, weight.columns), zeros)) }
        }
        SafeTensors.saveEntries(file, entries, metadata)
    }

    companion object {
        fun loadQuantized(file: java.io.File): Transformer {
            SafeTensorsFile(file).use { reader ->
                val meta = reader.metadata
                fun value(key: String): String = meta[key] ?: throw IllegalArgumentException("$file has no $key metadata; was it written by saveQuantized?")
                val scaling = meta["kotensor.ropeScaling"]?.split(',')?.let { parts ->
                    RoPEScaling(parts[0].toFloat(), parts[1].toFloat(), parts[2].toFloat(), parts[3].toInt())
                }
                val quantization = Quantization(value("kotensor.quantization.bits").toInt(), value("kotensor.quantization.groupSize").toInt())
                val model = Transformer(
                    vocabularySize = value("kotensor.vocabularySize").toInt(),
                    dimension = value("kotensor.dimension").toInt(),
                    numLayers = value("kotensor.numLayers").toInt(),
                    numHeads = value("kotensor.numHeads").toInt(),
                    hiddenDimension = value("kotensor.hiddenDimension").toInt(),
                    numKeyValueHeads = value("kotensor.numKeyValueHeads").toInt(),
                    normEps = value("kotensor.normEps").toFloat(),
                    ropeTheta = value("kotensor.ropeTheta").toFloat(),
                    ropeScaling = scaling,
                    quantization = quantization
                )
                for ((name, tensor) in model.namedParameters()) {
                    require(name in reader) { "$file is missing $name" }
                    reader.readInto(name, tensor.data)
                }
                for ((name, weight) in model.namedQuantizedTensors()) {
                    reader.readRaw("$name.qweight", weight.qdata)
                    reader.readInto("$name.scales", weight.scales)
                    weight.zeros?.let { zeros -> reader.readRaw("$name.zeros", zeros) }
                }
                return model
            }
        }
    }
}

