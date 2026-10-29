package io.kotensor.core

import java.io.File
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull
import kotlin.test.assertTrue

class HuggingFaceImportTest {

    private fun fixture(name: String): File {
        val url = HuggingFaceImportTest::class.java.getResource("/hf/$name/expected.json")
            ?: throw IllegalStateException("missing fixture $name")
        return File(url.toURI()).parentFile
    }

    private fun expected(directory: File): Pair<IntArray, List<FloatArray>> {
        val root = JsonParser.parse(File(directory, "expected.json").readText()) as Map<*, *>
        val ids = (root["input_ids"] as List<*>).map { v -> (v as Number).toInt() }.toIntArray()
        val logits = (root["logits"] as List<*>).map { row -> (row as List<*>).map { v -> (v as Number).toFloat() }.toFloatArray() }
        return ids to logits
    }

    private fun assertLogitsMatch(model: Transformer, ids: IntArray, logits: List<FloatArray>, label: String) {
        val actual = model.forward(ids)
        for (t in ids.indices) {
            for (v in 0 until model.vocabularySize) {
                assertEquals(logits[t][v], actual.data[t * model.vocabularySize + v], 2e-4f, "$label position $t vocab $v")
            }
        }
    }

    @Test
    fun testLlamaStyleCheckpointMatchesNumpyReference() {
        val directory = fixture("llama_tiny")
        val config = HuggingFaceModelLoader.loadConfig(directory)
        assertEquals("llama", config.modelType)
        assertEquals(2, config.numKeyValueHeads)
        assertEquals(8, config.headDim)
        assertEquals(1e-5f, config.rmsNormEps)
        assertNull(config.ropeScaling)

        val model = HuggingFaceModelLoader.newModel(config)
        val report = HuggingFaceModelLoader.loadWeights(model, config, HuggingFaceModelLoader.weightFiles(directory))
        assertTrue(!report.tiedLmHead)
        assertTrue(report.ignored.isEmpty())
        assertTrue(report.missing.all { name -> name.endsWith(".bias") })
        assertEquals(model.namedParameters().size - report.missing.size, report.loaded.size)

        val (ids, logits) = expected(directory)
        assertLogitsMatch(model, ids, logits, "llama")

        val cache = model.newCache(ids.size)
        val buffers = model.newInferenceBuffers(ids.size)
        for (t in ids.indices) {
            val step = model.decode(ids[t], cache, buffers)
            for (v in 0 until model.vocabularySize) {
                assertEquals(logits[t][v], step.data[v], 2e-4f, "decode position $t vocab $v")
            }
        }
    }

    @Test
    fun testShardedQwenStyleCheckpointWithBiasesAndTiedEmbeddings() {
        val directory = fixture("qwen_tiny_sharded")
        val files = HuggingFaceModelLoader.weightFiles(directory)
        assertEquals(2, files.size)
        assertTrue(files.all { f -> f.isFile })

        val progress = mutableListOf<String>()
        val model = HuggingFaceModelLoader.load(directory) { name -> progress.add(name) }
        assertEquals(1, model.numKeyValueHeads)
        assertTrue(progress.last().startsWith("lmHead.weight"))
        assertTrue((model.blocks[0].attention.wQ as Linear).bias.data.any { b -> b != 0.0f })
        assertTrue((model.blocks[0].attention.wOut as Linear).bias.data.all { b -> b == 0.0f })

        val config = HuggingFaceModelLoader.loadConfig(directory)
        val report = HuggingFaceModelLoader.loadWeights(HuggingFaceModelLoader.newModel(config), config, files)
        assertTrue(report.tiedLmHead)
        assertContentEquals(listOf("model.layers.0.self_attn.rotary_emb.inv_freq"), report.ignored)

        val (ids, logits) = expected(directory)
        assertLogitsMatch(model, ids, logits, "qwen")

        val tokenizer = HuggingFaceModelLoader.loadTokenizer(directory)
        assertEquals(420, tokenizer.vocabularySize)
        val generated = model.generate(ids.sliceArray(0 until 3), maxTokens = 4)
        assertEquals(7, generated.size)
    }

    @Test
    fun testLlama3RopeScalingCheckpointMatchesNumpyReference() {
        val directory = fixture("llama3_tiny")
        val config = HuggingFaceModelLoader.loadConfig(directory)
        assertEquals(500000.0f, config.ropeTheta)
        assertEquals(8.0f, config.ropeScaling!!.factor)
        val model = HuggingFaceModelLoader.load(directory)
        val (ids, logits) = expected(directory)
        assertLogitsMatch(model, ids, logits, "llama3")
    }

    @Test
    fun testWeightNameMapping() {
        assertEquals("embedding.weight", HuggingFaceModelLoader.kotensorName("model.embed_tokens.weight"))
        assertEquals("block.12.attention.wQ.weight", HuggingFaceModelLoader.kotensorName("model.layers.12.self_attn.q_proj.weight"))
        assertEquals("block.3.attention.wK.bias", HuggingFaceModelLoader.kotensorName("model.layers.3.self_attn.k_proj.bias"))
        assertEquals("block.0.ffn.wGate.weight", HuggingFaceModelLoader.kotensorName("model.layers.0.mlp.gate_proj.weight"))
        assertEquals("block.7.norm2.weight", HuggingFaceModelLoader.kotensorName("model.layers.7.post_attention_layernorm.weight"))
        assertEquals("finalNorm.weight", HuggingFaceModelLoader.kotensorName("model.norm.weight"))
        assertEquals("lmHead.weight", HuggingFaceModelLoader.kotensorName("lm_head.weight"))
        assertNull(HuggingFaceModelLoader.kotensorName("model.layers.0.self_attn.rotary_emb.inv_freq"))
        assertNull(HuggingFaceModelLoader.kotensorName("transformer.h.0.attn.c_attn.weight"))

        assertTrue(HuggingFaceModelLoader.isTransposedOnImport("block.1.attention.wOut.weight"))
        assertTrue(HuggingFaceModelLoader.isTransposedOnImport("lmHead.weight"))
        assertTrue(!HuggingFaceModelLoader.isTransposedOnImport("block.1.attention.wOut.bias"))
        assertTrue(!HuggingFaceModelLoader.isTransposedOnImport("embedding.weight"))
        assertTrue(!HuggingFaceModelLoader.isTransposedOnImport("block.1.norm1.weight"))
    }

    @Test
    fun testConfigValidation() {
        assertFailsWith<IllegalArgumentException> {
            HuggingFaceConfig.fromJson("""{"model_type": "gemma", "hidden_size": 32, "num_hidden_layers": 1, "num_attention_heads": 4, "intermediate_size": 64, "vocab_size": 10}""")
        }
        assertFailsWith<IllegalArgumentException> {
            HuggingFaceConfig.fromJson("""{"model_type": "llama", "hidden_size": 32, "num_hidden_layers": 1, "num_attention_heads": 4, "num_key_value_heads": 3, "intermediate_size": 64, "vocab_size": 10}""")
        }
        assertFailsWith<IllegalArgumentException> {
            HuggingFaceConfig.fromJson("""{"model_type": "llama", "hidden_size": 32, "num_hidden_layers": 1, "num_attention_heads": 4, "intermediate_size": 64, "vocab_size": 10, "rope_scaling": {"rope_type": "yarn", "factor": 4.0}}""")
        }
        val minimal = HuggingFaceConfig.fromJson("""{"hidden_size": 32, "num_hidden_layers": 1, "num_attention_heads": 4, "intermediate_size": 64, "vocab_size": 10}""")
        assertEquals("llama", minimal.modelType)
        assertEquals(4, minimal.numKeyValueHeads)
        assertEquals(10000.0f, minimal.ropeTheta)
        assertEquals(1e-6f, minimal.rmsNormEps)
        assertTrue(!minimal.tieWordEmbeddings)
    }

    @Test
    fun testMissingWeightsAreReported() {
        val directory = fixture("llama_tiny")
        val config = HuggingFaceModelLoader.loadConfig(directory)
        val bigger = HuggingFaceConfig(
            modelType = config.modelType, hiddenSize = config.hiddenSize, numHiddenLayers = config.numHiddenLayers + 1,
            numAttentionHeads = config.numAttentionHeads, numKeyValueHeads = config.numKeyValueHeads,
            intermediateSize = config.intermediateSize, rmsNormEps = config.rmsNormEps, ropeTheta = config.ropeTheta,
            vocabSize = config.vocabSize, maxPositionEmbeddings = config.maxPositionEmbeddings, tieWordEmbeddings = false
        )
        val error = assertFailsWith<IllegalArgumentException> {
            HuggingFaceModelLoader.loadWeights(HuggingFaceModelLoader.newModel(bigger), bigger, HuggingFaceModelLoader.weightFiles(directory))
        }
        assertTrue(error.message!!.contains("block.2"))

        val wrongWidth = HuggingFaceConfig(
            modelType = config.modelType, hiddenSize = config.hiddenSize, numHiddenLayers = config.numHiddenLayers,
            numAttentionHeads = config.numAttentionHeads, numKeyValueHeads = config.numKeyValueHeads,
            intermediateSize = config.intermediateSize * 2, rmsNormEps = config.rmsNormEps, ropeTheta = config.ropeTheta,
            vocabSize = config.vocabSize, maxPositionEmbeddings = config.maxPositionEmbeddings, tieWordEmbeddings = false
        )
        assertFailsWith<IllegalArgumentException> {
            HuggingFaceModelLoader.loadWeights(HuggingFaceModelLoader.newModel(wrongWidth), wrongWidth, HuggingFaceModelLoader.weightFiles(directory))
        }
    }

    @Test
    fun testSafeTensorsReadsHalfBrainFloatAndFloat() {
        val directory = fixture("dtypes")
        val expected = JsonParser.parse(File(directory, "expected.json").readText()) as Map<*, *>
        SafeTensorsFile(File(directory, "dtypes.safetensors")).use { reader ->
            assertEquals(listOf("halves", "bfloats", "floats"), reader.names)
            assertEquals("F16", reader.entry("halves").dtype)
            assertEquals("BF16", reader.entry("bfloats").dtype)
            assertEquals("pt", reader.metadata["format"])
            assertTrue(reader.entry("floats").shape.contentEquals(intArrayOf(2, 2)))
            for (name in reader.names) {
                val values = (expected[name] as List<*>).map { v -> (v as Number).toFloat() }
                val tensor = reader.load(name)
                assertEquals(values.size, tensor.data.size)
                for (i in values.indices) {
                    assertEquals(values[i], tensor.data[i], "$name[$i]")
                }
            }
            val halves = reader.load("halves")
            assertEquals(Float.POSITIVE_INFINITY, SafeTensorsFile.halfToFloat(0x7C00))
            assertTrue(SafeTensorsFile.halfToFloat(0x7E00).isNaN())
            assertEquals(-0.0f, halves.data[7])
            assertEquals(5.960464477539063e-08f, halves.data[4])
        }

        val roundTrip = File.createTempFile("kotensor_f32", ".safetensors")
        roundTrip.deleteOnExit()
        val original = mapOf("a" to Tensor(floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f), intArrayOf(2, 3)))
        SafeTensors.save(roundTrip, original)
        val loaded = SafeTensors.load(roundTrip)
        assertContentEquals(original.getValue("a").data, loaded.getValue("a").data)
        assertTrue(loaded.getValue("a").shape.contentEquals(intArrayOf(2, 3)))
    }
}

