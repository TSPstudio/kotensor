package io.kotensor.core

import java.io.File

class HuggingFaceConfig(
    val modelType: String,
    val hiddenSize: Int,
    val numHiddenLayers: Int,
    val numAttentionHeads: Int,
    val numKeyValueHeads: Int,
    val intermediateSize: Int,
    val rmsNormEps: Float,
    val ropeTheta: Float,
    val vocabSize: Int,
    val maxPositionEmbeddings: Int,
    val tieWordEmbeddings: Boolean,
    val ropeScaling: RoPEScaling? = null
) {
    val headDim: Int
        get() = hiddenSize / numAttentionHeads

    init {
        require(modelType in SUPPORTED_MODEL_TYPES) { "Unsupported model_type '$modelType'; supported: $SUPPORTED_MODEL_TYPES" }
        require(hiddenSize > 0 && numHiddenLayers > 0 && numAttentionHeads > 0 && intermediateSize > 0 && vocabSize > 0) {
            "config.json has a non-positive model dimension"
        }
        require(hiddenSize % numAttentionHeads == 0) { "hidden_size $hiddenSize is not divisible by num_attention_heads $numAttentionHeads" }
        require(numAttentionHeads % numKeyValueHeads == 0) { "num_attention_heads $numAttentionHeads is not a multiple of num_key_value_heads $numKeyValueHeads" }
    }

    companion object {
        val SUPPORTED_MODEL_TYPES = setOf("llama", "qwen2", "mistral")

        fun fromFile(file: File): HuggingFaceConfig = fromJson(file.readText(Charsets.UTF_8))

        fun fromJson(json: String): HuggingFaceConfig {
            val root = JsonParser.parse(json) as? Map<*, *> ?: throw IllegalArgumentException("config.json must be a JSON object")
            val modelType = root["model_type"] as? String ?: "llama"
            val hiddenSize = int(root, "hidden_size")
            val numAttentionHeads = int(root, "num_attention_heads")
            val headDim = (root["head_dim"] as? Number)?.toInt()
            require(headDim == null || headDim * numAttentionHeads == hiddenSize) {
                "head_dim $headDim × num_attention_heads $numAttentionHeads must equal hidden_size $hiddenSize"
            }
            return HuggingFaceConfig(
                modelType = modelType,
                hiddenSize = hiddenSize,
                numHiddenLayers = int(root, "num_hidden_layers"),
                numAttentionHeads = numAttentionHeads,
                numKeyValueHeads = (root["num_key_value_heads"] as? Number)?.toInt() ?: numAttentionHeads,
                intermediateSize = int(root, "intermediate_size"),
                rmsNormEps = (root["rms_norm_eps"] as? Number)?.toFloat() ?: 1e-6f,
                ropeTheta = (root["rope_theta"] as? Number)?.toFloat() ?: 10000.0f,
                vocabSize = int(root, "vocab_size"),
                maxPositionEmbeddings = (root["max_position_embeddings"] as? Number)?.toInt() ?: 2048,
                tieWordEmbeddings = root["tie_word_embeddings"] == true,
                ropeScaling = ropeScalingOf(root["rope_scaling"])
            )
        }

        private fun int(root: Map<*, *>, key: String): Int {
            return (root[key] as? Number)?.toInt() ?: throw IllegalArgumentException("config.json has no numeric \"$key\"")
        }

        private fun ropeScalingOf(node: Any?): RoPEScaling? {
            val scaling = node as? Map<*, *> ?: return null
            val type = (scaling["rope_type"] ?: scaling["type"]) as? String ?: "default"
            return when (type) {
                "default" -> null
                "llama3" -> RoPEScaling(
                    factor = (scaling["factor"] as? Number)?.toFloat() ?: 8.0f,
                    lowFrequencyFactor = (scaling["low_freq_factor"] as? Number)?.toFloat() ?: 1.0f,
                    highFrequencyFactor = (scaling["high_freq_factor"] as? Number)?.toFloat() ?: 4.0f,
                    originalMaxPositionEmbeddings = (scaling["original_max_position_embeddings"] as? Number)?.toInt() ?: 8192
                )
                else -> throw IllegalArgumentException("Unsupported rope_scaling type '$type'; only 'llama3' and default are supported")
            }
        }
    }
}

class ImportReport(
    val loaded: List<String>,
    val ignored: List<String>,
    val missing: List<String>,
    val tiedLmHead: Boolean
)

object HuggingFaceModelLoader {

    val weightMapping: List<Pair<Regex, String>> = listOf(
        Regex("""^model\.embed_tokens\.weight$""") to "embedding.weight",
        Regex("""^model\.norm\.weight$""") to "finalNorm.weight",
        Regex("""^lm_head\.weight$""") to "lmHead.weight",
        Regex("""^model\.layers\.(\d+)\.input_layernorm\.weight$""") to "block.$1.norm1.weight",
        Regex("""^model\.layers\.(\d+)\.post_attention_layernorm\.weight$""") to "block.$1.norm2.weight",
        Regex("""^model\.layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$""") to "block.$1.attention.wQ.$2",
        Regex("""^model\.layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$""") to "block.$1.attention.wK.$2",
        Regex("""^model\.layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$""") to "block.$1.attention.wV.$2",
        Regex("""^model\.layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$""") to "block.$1.attention.wOut.$2",
        Regex("""^model\.layers\.(\d+)\.mlp\.gate_proj\.(weight|bias)$""") to "block.$1.ffn.wGate.$2",
        Regex("""^model\.layers\.(\d+)\.mlp\.up_proj\.(weight|bias)$""") to "block.$1.ffn.wUp.$2",
        Regex("""^model\.layers\.(\d+)\.mlp\.down_proj\.(weight|bias)$""") to "block.$1.ffn.wDown.$2"
    )

    private val transposedWeights = Regex("""^(block\.\d+\.(attention\.w(Q|K|V|Out)|ffn\.w(Gate|Up|Down))|lmHead)\.weight$""")

    fun kotensorName(huggingFaceName: String): String? {
        for ((pattern, replacement) in weightMapping) {
            val match = pattern.find(huggingFaceName) ?: continue
            var name = replacement
            for (group in 1 until match.groupValues.size) {
                name = name.replace("\$$group", match.groupValues[group])
            }
            return name
        }
        return null
    }

    fun isTransposedOnImport(kotensorName: String): Boolean = transposedWeights.matches(kotensorName)

    fun loadConfig(directory: File): HuggingFaceConfig = HuggingFaceConfig.fromFile(File(directory, "config.json"))

    fun loadTokenizer(directory: File): BPETokenizer = BPETokenizer.fromFile(File(directory, "tokenizer.json"))

    fun newModel(config: HuggingFaceConfig, quantization: Quantization? = null): Transformer {
        return Transformer(
            vocabularySize = config.vocabSize,
            dimension = config.hiddenSize,
            numLayers = config.numHiddenLayers,
            numHeads = config.numAttentionHeads,
            hiddenDimension = config.intermediateSize,
            numKeyValueHeads = config.numKeyValueHeads,
            normEps = config.rmsNormEps,
            ropeTheta = config.ropeTheta,
            ropeScaling = config.ropeScaling,
            quantization = quantization
        )
    }

    fun weightFiles(directory: File): List<File> {
        val index = File(directory, "model.safetensors.index.json")
        if (index.isFile) {
            val root = JsonParser.parse(index.readText(Charsets.UTF_8)) as? Map<*, *>
                ?: throw IllegalArgumentException("$index is not a JSON object")
            val weightMap = root["weight_map"] as? Map<*, *> ?: throw IllegalArgumentException("$index has no weight_map")
            val shards = LinkedHashSet<String>()
            for ((_, shard) in weightMap) shards.add(shard as String)
            return shards.map { shard -> File(directory, shard) }
        }
        val single = File(directory, "model.safetensors")
        require(single.isFile) { "No model.safetensors or model.safetensors.index.json in $directory" }
        return listOf(single)
    }

    fun load(directory: File, quantization: Quantization? = null, onProgress: (String) -> Unit = {}): Transformer {
        val config = loadConfig(directory)
        val model = newModel(config, quantization)
        loadWeights(model, config, weightFiles(directory), onProgress)
        return model
    }

    fun loadWeights(model: Transformer, config: HuggingFaceConfig, files: List<File>, onProgress: (String) -> Unit = {}): ImportReport {
        val targets = model.namedParameters()
        val quantizedLayers = model.namedQuantizedLayers()
        val loaded = LinkedHashSet<String>()
        val ignored = mutableListOf<String>()
        var embeddingSource: Pair<File, String>? = null

        for (file in files) {
            SafeTensorsFile(file).use { reader ->
                for (huggingFaceName in reader.names) {
                    val name = kotensorName(huggingFaceName)
                    if (name == null) {
                        ignored.add(huggingFaceName)
                        continue
                    }
                    val quantizedLayer = if (name.endsWith(".weight")) quantizedLayers[name.removeSuffix(".weight")] else null
                    val quantizedEmbedding = if (name == "embedding.weight") model.embedding as? QuantizedEmbedding else null
                    val target = targets[name]
                    when {
                        quantizedLayer != null -> importQuantized(reader, huggingFaceName, name, quantizedLayer)
                        quantizedEmbedding != null -> {
                            val table = FloatArray(model.vocabularySize * model.dimension)
                            reader.readInto(huggingFaceName, table)
                            quantizedEmbedding.weight = QuantizedTensor.quantize(table, model.vocabularySize, model.dimension, quantizedEmbedding.quantization)
                            embeddingSource = file to huggingFaceName
                        }
                        target != null -> {
                            importTensor(reader, huggingFaceName, name, target)
                            if (name == "embedding.weight") embeddingSource = file to huggingFaceName
                        }
                        else -> {
                            ignored.add(huggingFaceName)
                            continue
                        }
                    }
                    loaded.add(name)
                    onProgress(name)
                }
            }
        }

        var tiedLmHead = false
        if ("lmHead.weight" !in loaded && "embedding.weight" in loaded) {
            require(config.tieWordEmbeddings) { "lm_head.weight is missing and tie_word_embeddings is false" }
            val (sourceFile, sourceName) = embeddingSource!!
            val table = when (val embedding = model.embedding) {
                is Embedding -> embedding.weight.data
                else -> FloatArray(model.vocabularySize * model.dimension).also { buffer ->
                    SafeTensorsFile(sourceFile).use { reader -> reader.readInto(sourceName, buffer) }
                }
            }
            val transposed = when (val head = model.lmHead) {
                is Linear -> head.weight.data
                is QuantizedLinear -> FloatArray(model.dimension * model.vocabularySize)
                else -> throw IllegalStateException("lmHead is ${head::class.simpleName}")
            }
            MatMul.transpose(table, model.vocabularySize, model.dimension, transposed)
            (model.lmHead as? QuantizedLinear)?.let { head ->
                head.weight = QuantizedTensor.quantize(transposed, model.dimension, model.vocabularySize, head.quantization)
            }
            loaded.add("lmHead.weight")
            tiedLmHead = true
            onProgress("lmHead.weight (tied to embedding.weight)")
        }

        val expected = targets.keys + quantizedLayers.keys.map { name -> "$name.weight" } +
            (if (model.embedding is QuantizedEmbedding) listOf("embedding.weight") else emptyList())
        val missing = expected.filter { name -> name !in loaded }
        val missingWeights = missing.filter { name -> !name.endsWith(".bias") }
        require(missingWeights.isEmpty()) { "Checkpoint is missing ${missingWeights.size} weights, e.g. ${missingWeights.take(5)}" }
        return ImportReport(loaded.toList(), ignored, missing, tiedLmHead)
    }

    private fun readTransposed(reader: SafeTensorsFile, huggingFaceName: String, inSize: Int, outSize: Int): FloatArray {
        val entry = reader.entry(huggingFaceName)
        require(entry.shape.size == 2 && entry.shape[0] == outSize && entry.shape[1] == inSize) {
            "$huggingFaceName has shape ${entry.shape.contentToString()}, expected [$outSize, $inSize]"
        }
        val rowMajorOutIn = FloatArray(inSize * outSize)
        reader.readInto(huggingFaceName, rowMajorOutIn)
        val transposed = FloatArray(inSize * outSize)
        MatMul.transpose(rowMajorOutIn, outSize, inSize, transposed)
        return transposed
    }

    private fun importQuantized(reader: SafeTensorsFile, huggingFaceName: String, name: String, layer: QuantizedLinear) {
        require(isTransposedOnImport(name)) { "$name is not a projection weight" }
        val transposed = readTransposed(reader, huggingFaceName, layer.inSize, layer.outSize)
        layer.weight = QuantizedTensor.quantize(transposed, layer.inSize, layer.outSize, layer.quantization)
    }

    private fun importTensor(reader: SafeTensorsFile, huggingFaceName: String, name: String, target: Tensor) {
        val entry = reader.entry(huggingFaceName)
        require(entry.elementCount == target.data.size) {
            "$huggingFaceName has shape ${entry.shape.contentToString()}, Kotensor $name expects ${target.shape.contentToString()}"
        }
        if (isTransposedOnImport(name)) {
            val inSize = target.shape[0]
            val outSize = target.shape[1]
            require(entry.shape.size == 2 && entry.shape[0] == outSize && entry.shape[1] == inSize) {
                "$huggingFaceName has shape ${entry.shape.contentToString()}, expected [$outSize, $inSize]"
            }
            val rowMajorOutIn = FloatArray(target.data.size)
            reader.readInto(huggingFaceName, rowMajorOutIn)
            MatMul.transpose(rowMajorOutIn, outSize, inSize, target.data)
        } else {
            reader.readInto(huggingFaceName, target.data)
        }
    }
}

