package io.kotensor.core

import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class BPETokenizerTest {

    private fun resource(name: String): String {
        val stream = BPETokenizerTest::class.java.getResourceAsStream("/$name")
            ?: throw IllegalStateException("missing test resource $name")
        return stream.use { s -> s.readBytes().toString(Charsets.UTF_8) }
    }

    private fun loadTokenizer(style: String): BPETokenizer = BPETokenizer.fromJson(resource("tokenizer_$style.json"))

    private fun ints(value: Any?): IntArray = (value as List<*>).map { v -> (v as Number).toInt() }.toIntArray()

    @Test
    fun testMatchesHuggingFaceReference() {
        val reference = JsonParser.parse(resource("reference_encodings.json")) as Map<*, *>
        for (style in listOf("gpt2style", "qwenstyle", "llama3style")) {
            val tokenizer = loadTokenizer(style)
            val expected = reference[style] as Map<*, *>

            assertEquals((expected["vocab_size"] as Number).toInt(), tokenizer.vocabularySize, "$style vocabularySize")
            assertEquals((expected["endoftext"] as Number).toInt(), tokenizer.endOfTextId, "$style endOfTextId")
            assertEquals((expected["im_start"] as Number).toInt(), tokenizer.imStartId, "$style imStartId")
            assertEquals((expected["im_end"] as Number).toInt(), tokenizer.imEndId, "$style imEndId")

            val cases = expected["cases"] as List<*>
            assertTrue(cases.size >= 10)
            for (entry in cases) {
                val case = entry as Map<*, *>
                val text = case["text"] as String
                val ids = ints(case["ids"])
                assertContentEquals(ids, tokenizer.encode(text), "$style encode(${text.replace("\n", "\\n")})")
                assertContentEquals(ints(case["ids_no_special"]), tokenizer.encode(text, addSpecialTokens = false), "$style encode without template")
                assertEquals(case["decoded"] as String, tokenizer.decode(ids), "$style decode")
                assertEquals(case["decoded_skip"] as String, tokenizer.decode(ids, skipSpecialTokens = true), "$style decode skipping specials")
            }
        }
    }

    @Test
    fun testByteLevelCoverageRoundTripsAnyText() {
        val tokenizer = loadTokenizer("gpt2style")
        val texts = listOf(
            "plain ascii",
            "Automatización e Industria 4.0 en el ITT2",
            "日本語のテキスト and عربى and русский",
            "emoji 🚀🔥🧠 and flags 🇲🇽",
            "control chars \u0001\u0007 and \t tabs \r\n newlines",
            "symbols ©®™ ¡¬ ®ÿ Ā ă",
            "x".repeat(300),
            " non-breaking spaces　here"
        )
        for (text in texts) {
            val ids = tokenizer.encode(text)
            for (id in ids) {
                assertTrue(id in 0 until tokenizer.vocabularySize, "id $id out of range")
            }
            assertEquals(text, tokenizer.decode(ids), "round trip of ${text.take(20)}")
        }
    }

    @Test
    fun testSpecialTokensEncodeAsSingleIds() {
        val tokenizer = loadTokenizer("qwenstyle")
        val imStart = assertNotNull(tokenizer.imStartId)
        val imEnd = assertNotNull(tokenizer.imEndId)
        val endOfText = assertNotNull(tokenizer.endOfTextId)

        val ids = tokenizer.encode("<|im_start|>user\nhi<|im_end|><|endoftext|>")
        assertEquals(imStart, ids.first())
        assertEquals(endOfText, ids.last())
        assertEquals(imEnd, ids[ids.size - 2])
        assertEquals(1, ids.count { it == imStart })

        assertTrue(tokenizer.isSpecial(imStart))
        assertTrue(!tokenizer.isSpecial(tokenizer.encode("a").first()))
        assertEquals(imStart, tokenizer.tokenId(BPETokenizer.IM_START))
        assertEquals(BPETokenizer.IM_END, tokenizer.token(imEnd))
        assertEquals("user\nhi", tokenizer.decode(ids, skipSpecialTokens = true))
        assertEquals("<|im_start|>user\nhi<|im_end|><|endoftext|>", tokenizer.decode(ids))
    }

    @Test
    fun testTemplateTokensAreOptional() {
        val tokenizer = loadTokenizer("llama3style")
        val bos = assertNotNull(tokenizer.endOfTextId)
        assertContentEquals(intArrayOf(bos), tokenizer.prefixTokens)
        assertEquals(0, tokenizer.suffixTokens.size)

        val withTemplate = tokenizer.encode("Hello world")
        val plain = tokenizer.encode("Hello world", addSpecialTokens = false)
        assertEquals(bos, withTemplate.first())
        assertContentEquals(plain, withTemplate.sliceArray(1 until withTemplate.size))
        assertEquals("Hello world", tokenizer.decode(withTemplate, skipSpecialTokens = true))
    }

    @Test
    fun testConstructorFromVocabAndMerges() {
        val bytes = mutableMapOf<String, Int>()
        for (c in 0x21..0x7E) bytes[c.toChar().toString()] = bytes.size
        bytes["Ġ"] = bytes.size
        val vocab = bytes + mapOf("he" to bytes.size, "ll" to bytes.size + 1, "hell" to bytes.size + 2, "hello" to bytes.size + 3, "Ġw" to bytes.size + 4)
        val merges = listOf("h" to "e", "l" to "l", "he" to "ll", "hell" to "o", "Ġ" to "w")
        val tokenizer = BPETokenizer(vocab, merges, specialTokens = mapOf("<|endoftext|>" to vocab.size))

        assertEquals(vocab.size + 1, tokenizer.vocabularySize)
        assertContentEquals(intArrayOf(vocab.getValue("hello")), tokenizer.encode("hello"))
        assertContentEquals(intArrayOf(vocab.getValue("hell")), tokenizer.encode("hell"))
        val helloWorld = tokenizer.encode("hello world")
        assertEquals(vocab.getValue("hello"), helloWorld[0])
        assertEquals(vocab.getValue("Ġw"), helloWorld[1])
        assertEquals("hello world", tokenizer.decode(helloWorld))
        assertEquals(vocab.size, tokenizer.endOfTextId)
        assertContentEquals(intArrayOf(vocab.size), tokenizer.encode("<|endoftext|>"))

        assertFailsWith<IllegalStateException> { tokenizer.encode("é") }
    }

    @Test
    fun testIncrementalDecoderWaitsForCompleteCharacters() {
        val tokenizer = loadTokenizer("gpt2style")
        val text = "ok 🚀 日本"
        val ids = tokenizer.encode(text)
        assertTrue(ids.size > text.length, "multi-byte characters should span several byte tokens")

        val decoder = IncrementalDecoder(tokenizer)
        val pieces = mutableListOf<String>()
        for (id in ids) {
            val piece = decoder.push(id)
            if (piece.isNotEmpty()) pieces.add(piece)
        }
        pieces.add(decoder.flush())

        assertEquals(text, pieces.joinToString(""))
        assertTrue(pieces.none { piece -> piece.contains('�') }, "no partial characters were emitted: $pieces")

        val charLevel = IncrementalDecoder(SimpleTokenizer("abc"))
        assertEquals("a", charLevel.push(0))
        assertEquals("c", charLevel.push(2))
        assertEquals("", charLevel.flush())
    }

    @Test
    fun testPreTokenizerPatternCompilation() {
        val pattern = BPETokenizer.compileUnicodePattern(BPETokenizer.GPT2_PRE_TOKENIZER_PATTERN)
        assertTrue(pattern.flags() and java.util.regex.Pattern.UNICODE_CHARACTER_CLASS != 0)
        assertFailsWith<java.util.regex.PatternSyntaxException> { BPETokenizer.compileUnicodePattern("(unclosed") }
    }

    @Test
    fun testJsonParser() {
        val parsed = JsonParser.parse(
            """{"a": [1, 2.5, -3e2, true, false, null], "s": "q\"\\\/\né🚀", "o": {"k": {}}, "e": []}"""
        ) as Map<*, *>
        assertEquals(listOf<Any?>(1L, 2.5, -300.0, true, false, null), parsed["a"])
        assertEquals("q\"\\/\né🚀", parsed["s"])
        assertEquals(emptyMap<String, Any?>(), (parsed["o"] as Map<*, *>)["k"])
        assertEquals(emptyList<Any?>(), parsed["e"])
        assertEquals(emptyMap<String, Any?>(), JsonParser.member(parsed["o"], "k"))
        assertEquals(null, JsonParser.member("not an object", "k"))

        assertFailsWith<IllegalArgumentException> { JsonParser.parse("{\"a\": }") }
        assertFailsWith<IllegalArgumentException> { JsonParser.parse("[1, 2") }
        assertFailsWith<IllegalArgumentException> { JsonParser.parse("{} extra") }
    }
}

