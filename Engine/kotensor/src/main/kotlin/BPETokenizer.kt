package io.kotensor.core

import java.io.ByteArrayOutputStream
import java.io.File
import java.text.Normalizer
import java.util.concurrent.ConcurrentHashMap
import java.util.regex.Pattern

class BPETokenizer(
    vocab: Map<String, Int>,
    merges: List<Pair<String, String>>,
    specialTokens: Map<String, Int> = emptyMap(),
    addedTokens: Map<String, Int> = emptyMap(),
    val preTokenizerPattern: String = GPT2_PRE_TOKENIZER_PATTERN,
    val addPrefixSpace: Boolean = false,
    val normalization: String? = null,
    val ignoreMerges: Boolean = false,
    val unkToken: String? = null,
    prefixTokens: IntArray = IntArray(0),
    suffixTokens: IntArray = IntArray(0)
) : Tokenizer {

    val vocab: Map<String, Int>

    val specialTokens: Map<String, Int>

    val addedTokens: Map<String, Int>

    val prefixTokens: IntArray = prefixTokens.copyOf()
    val suffixTokens: IntArray = suffixTokens.copyOf()

    override val vocabularySize: Int

    private val idToToken: Array<String?>
    private val addedIds: BooleanArray
    private val specialIds: BooleanArray
    private val mergeRanks: HashMap<String, Int>
    private val byteToUnicode = CharArray(256)
    private val unicodeToByte = IntArray(UNICODE_TABLE_SIZE) { -1 }
    private val pattern: Pattern = compileUnicodePattern(preTokenizerPattern)
    private val addedTokenPattern: Pattern?
    private val normalizerForm: Normalizer.Form?
    private val unkId: Int?
    private val wordCache = ConcurrentHashMap<String, IntArray>()

    init {
        require(vocab.isNotEmpty()) { "Vocabulary cannot be empty" }
        val fullVocab = HashMap<String, Int>(vocab.size * 2)
        fullVocab.putAll(vocab)
        fullVocab.putAll(addedTokens)
        fullVocab.putAll(specialTokens)
        this.vocab = fullVocab
        this.specialTokens = HashMap(specialTokens)
        val allAdded = HashMap<String, Int>(addedTokens)
        allAdded.putAll(specialTokens)
        this.addedTokens = allAdded

        var maxId = -1
        for (id in fullVocab.values) {
            require(id >= 0) { "Token id $id must not be negative" }
            if (id > maxId) maxId = id
        }
        vocabularySize = maxId + 1

        idToToken = arrayOfNulls(vocabularySize)
        for ((token, id) in vocab) {
            idToToken[id] = token
        }
        addedIds = BooleanArray(vocabularySize)
        specialIds = BooleanArray(vocabularySize)
        for ((token, id) in allAdded) {
            idToToken[id] = token
            addedIds[id] = true
        }
        for ((_, id) in specialTokens) {
            specialIds[id] = true
        }

        mergeRanks = HashMap(merges.size * 2)
        for (i in merges.indices) {
            val (left, right) = merges[i]
            mergeRanks.putIfAbsent(left + " " + right, i)
        }

        buildByteTable()

        addedTokenPattern = if (allAdded.isEmpty()) {
            null
        } else {
            val alternatives = allAdded.keys.sortedByDescending { it.length }.joinToString("|") { Pattern.quote(it) }
            Pattern.compile(alternatives)
        }

        normalizerForm = when (normalization) {
            null -> null
            "NFC" -> Normalizer.Form.NFC
            "NFD" -> Normalizer.Form.NFD
            "NFKC" -> Normalizer.Form.NFKC
            "NFKD" -> Normalizer.Form.NFKD
            else -> throw IllegalArgumentException("Unsupported normalization '$normalization'")
        }

        unkId = if (unkToken != null) fullVocab[unkToken] else null
        for (id in this.prefixTokens + this.suffixTokens) {
            require(id in 0 until vocabularySize) { "Template token id $id out of vocabulary bounds $vocabularySize" }
        }
    }

    val endOfTextId: Int?
        get() = vocab[END_OF_TEXT]

    val imStartId: Int?
        get() = vocab[IM_START]

    val imEndId: Int?
        get() = vocab[IM_END]

    fun tokenId(token: String): Int? = vocab[token]

    fun token(id: Int): String? = if (id in 0 until vocabularySize) idToToken[id] else null

    fun isSpecial(id: Int): Boolean = id in 0 until vocabularySize && specialIds[id]

    override fun encode(text: String): IntArray = encode(text, addSpecialTokens = true)

    fun encode(text: String, addSpecialTokens: Boolean): IntArray {
        val ids = ArrayList<Int>(text.length / 3 + 8)
        if (addSpecialTokens) {
            for (id in prefixTokens) ids.add(id)
        }
        val splitter = addedTokenPattern
        if (splitter == null) {
            encodeText(text, ids)
        } else {
            val matcher = splitter.matcher(text)
            var last = 0
            while (matcher.find()) {
                encodeText(text.substring(last, matcher.start()), ids)
                ids.add(addedTokens.getValue(matcher.group()))
                last = matcher.end()
            }
            encodeText(text.substring(last), ids)
        }
        if (addSpecialTokens) {
            for (id in suffixTokens) ids.add(id)
        }
        return ids.toIntArray()
    }

    private fun encodeText(chunk: String, out: MutableList<Int>) {
        if (chunk.isEmpty()) return
        var text = chunk
        if (normalizerForm != null) {
            text = Normalizer.normalize(text, normalizerForm)
        }
        if (addPrefixSpace && !text.startsWith(' ')) {
            text = " $text"
        }
        val matcher = pattern.matcher(text)
        var last = 0
        while (matcher.find()) {
            if (matcher.start() > last) {
                encodeWord(text.substring(last, matcher.start()), out)
            }
            if (matcher.end() > matcher.start()) {
                encodeWord(matcher.group(), out)
            }
            last = matcher.end()
        }
        if (last < text.length) {
            encodeWord(text.substring(last), out)
        }
    }

    private fun encodeWord(word: String, out: MutableList<Int>) {
        val cached = wordCache[word]
        val ids = cached ?: bpe(word).also { wordCache[word] = it }
        for (id in ids) out.add(id)
    }

    private fun bpe(word: String): IntArray {
        val bytes = word.toByteArray(Charsets.UTF_8)
        val mapped = CharArray(bytes.size) { i -> byteToUnicode[bytes[i].toInt() and 0xFF] }
        val wholeWord = String(mapped)

        if (ignoreMerges) {
            val whole = vocab[wholeWord]
            if (whole != null) return intArrayOf(whole)
        }

        var symbols = ArrayList<String>(mapped.size)
        for (c in mapped) symbols.add(c.toString())

        while (symbols.size > 1) {
            var bestRank = Int.MAX_VALUE
            var bestIndex = -1
            for (i in 0 until symbols.size - 1) {
                val rank = mergeRanks[symbols[i] + " " + symbols[i + 1]] ?: continue
                if (rank < bestRank) {
                    bestRank = rank
                    bestIndex = i
                }
            }
            if (bestIndex < 0) break

            val left = symbols[bestIndex]
            val right = symbols[bestIndex + 1]
            val mergedPair = left + right
            val merged = ArrayList<String>(symbols.size)
            var i = 0
            while (i < symbols.size) {
                if (i < symbols.size - 1 && symbols[i] == left && symbols[i + 1] == right) {
                    merged.add(mergedPair)
                    i += 2
                } else {
                    merged.add(symbols[i])
                    i++
                }
            }
            symbols = merged
        }

        val ids = IntArray(symbols.size)
        for (i in symbols.indices) {
            ids[i] = vocab[symbols[i]] ?: unkId
                ?: throw IllegalStateException("Token '${symbols[i]}' is not in the vocabulary and no unk_token is set")
        }
        return ids
    }

    override fun decode(tokens: IntArray): String = decode(tokens, skipSpecialTokens = false)

    fun decode(tokens: IntArray, skipSpecialTokens: Boolean): String {
        val sb = StringBuilder()
        val bytes = ByteArrayOutputStream()
        for (id in tokens) {
            if (id !in 0 until vocabularySize) continue
            val token = idToToken[id] ?: continue
            if (addedIds[id]) {
                flushBytes(bytes, sb)
                if (!(skipSpecialTokens && specialIds[id])) {
                    sb.append(token)
                }
            } else {
                for (c in token) {
                    val b = if (c.code < UNICODE_TABLE_SIZE) unicodeToByte[c.code] else -1
                    if (b >= 0) {
                        bytes.write(b)
                    } else {
                        val raw = c.toString().toByteArray(Charsets.UTF_8)
                        bytes.write(raw, 0, raw.size)
                    }
                }
            }
        }
        flushBytes(bytes, sb)
        return sb.toString()
    }

    private fun flushBytes(bytes: ByteArrayOutputStream, sb: StringBuilder) {
        if (bytes.size() == 0) return
        sb.append(String(bytes.toByteArray(), Charsets.UTF_8))
        bytes.reset()
    }

    private fun buildByteTable() {
        var next = 256
        for (b in 0 until 256) {
            val printable = (b in '!'.code..'~'.code) || (b in '¡'.code..'¬'.code) || (b in '®'.code..'ÿ'.code)
            val c = if (printable) b.toChar() else (next++).toChar()
            byteToUnicode[b] = c
            unicodeToByte[c.code] = b
        }
    }

    companion object {
        const val END_OF_TEXT = "<|endoftext|>"
        const val IM_START = "<|im_start|>"
        const val IM_END = "<|im_end|>"

        const val GPT2_PRE_TOKENIZER_PATTERN = "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"

        private const val UNICODE_TABLE_SIZE = 256 + 68

        fun compileUnicodePattern(regex: String): Pattern {
            return try {
                Pattern.compile(regex, Pattern.UNICODE_CHARACTER_CLASS)
            } catch (e: IllegalArgumentException) {
                if (e is java.util.regex.PatternSyntaxException) throw e
                Pattern.compile(regex)
            }
        }

        fun fromFile(file: File): BPETokenizer = fromJson(file.readText(Charsets.UTF_8))

        fun fromJson(json: String): BPETokenizer {
            val root = JsonParser.parse(json) as? Map<*, *>
                ?: throw IllegalArgumentException("tokenizer.json must be a JSON object")
            val model = root["model"] as? Map<*, *>
                ?: throw IllegalArgumentException("tokenizer.json has no \"model\" object")
            val modelType = model["type"]
            require(modelType == null || modelType == "BPE") { "Unsupported tokenizer model type '$modelType'; only BPE is supported" }

            val vocabJson = model["vocab"] as? Map<*, *>
                ?: throw IllegalArgumentException("tokenizer.json model has no \"vocab\"")
            val vocab = HashMap<String, Int>(vocabJson.size * 2)
            for ((token, id) in vocabJson) {
                vocab[token as String] = (id as Number).toInt()
            }

            val mergesJson = model["merges"] as? List<*> ?: emptyList<Any?>()
            val merges = ArrayList<Pair<String, String>>(mergesJson.size)
            for (entry in mergesJson) {
                when (entry) {
                    is String -> {
                        val space = entry.indexOf(' ')
                        require(space > 0 && space < entry.length - 1) { "Malformed merge '$entry'" }
                        merges.add(entry.substring(0, space) to entry.substring(space + 1))
                    }
                    is List<*> -> {
                        require(entry.size == 2) { "Malformed merge $entry" }
                        merges.add(entry[0] as String to entry[1] as String)
                    }
                    else -> throw IllegalArgumentException("Malformed merge $entry")
                }
            }

            val specialTokens = LinkedHashMap<String, Int>()
            val addedTokens = LinkedHashMap<String, Int>()
            val addedJson = root["added_tokens"] as? List<*> ?: emptyList<Any?>()
            for (entry in addedJson) {
                val added = entry as? Map<*, *> ?: continue
                val content = added["content"] as? String ?: continue
                val id = (added["id"] as? Number)?.toInt() ?: continue
                if (added["special"] == true) specialTokens[content] = id else addedTokens[content] = id
            }

            var pattern: String? = null
            var addPrefixSpace = false
            forEachPreTokenizer(root["pre_tokenizer"]) { pre ->
                when (pre["type"]) {
                    "Split" -> {
                        val spec = pre["pattern"] as? Map<*, *>
                        val regex = spec?.get("Regex") as? String
                        val literal = spec?.get("String") as? String
                        if (pattern == null) {
                            if (regex != null) pattern = regex else if (literal != null) pattern = Pattern.quote(literal)
                        }
                    }
                    "ByteLevel" -> {
                        if (pre["add_prefix_space"] == true) addPrefixSpace = true
                        if (pattern == null && pre["use_regex"] != false) pattern = GPT2_PRE_TOKENIZER_PATTERN
                    }
                }
            }

            val normalization = normalizationOf(root["normalizer"])
            val template = templateTokensOf(root["post_processor"], vocab, specialTokens, addedTokens)

            return BPETokenizer(
                vocab = vocab,
                merges = merges,
                specialTokens = specialTokens,
                addedTokens = addedTokens,
                preTokenizerPattern = pattern ?: GPT2_PRE_TOKENIZER_PATTERN,
                addPrefixSpace = addPrefixSpace,
                normalization = normalization,
                ignoreMerges = model["ignore_merges"] == true,
                unkToken = model["unk_token"] as? String,
                prefixTokens = template.first,
                suffixTokens = template.second
            )
        }

        private fun forEachPreTokenizer(node: Any?, action: (Map<*, *>) -> Unit) {
            val pre = node as? Map<*, *> ?: return
            if (pre["type"] == "Sequence") {
                val children = pre["pretokenizers"] as? List<*> ?: return
                for (child in children) forEachPreTokenizer(child, action)
            } else {
                action(pre)
            }
        }

        private fun normalizationOf(node: Any?): String? {
            val normalizer = node as? Map<*, *> ?: return null
            return when (val type = normalizer["type"]) {
                "NFC", "NFD", "NFKC", "NFKD" -> type as String
                "Sequence" -> {
                    val children = normalizer["normalizers"] as? List<*> ?: return null
                    children.firstNotNullOfOrNull { child -> normalizationOf(child) }
                }
                else -> null
            }
        }

        private fun templateTokensOf(
            node: Any?,
            vocab: Map<String, Int>,
            specialTokens: Map<String, Int>,
            addedTokens: Map<String, Int>
        ): Pair<IntArray, IntArray> {
            val empty = IntArray(0) to IntArray(0)
            val processor = node as? Map<*, *> ?: return empty
            val processors: List<Map<*, *>> = if (processor["type"] == "Sequence") {
                (processor["processors"] as? List<*>)?.filterIsInstance<Map<*, *>>() ?: emptyList()
            } else {
                listOf(processor)
            }
            val template = processors.firstOrNull { p -> p["type"] == "TemplateProcessing" } ?: return empty
            val single = template["single"] as? List<*> ?: return empty
            val declared = template["special_tokens"] as? Map<*, *>

            val prefix = ArrayList<Int>()
            val suffix = ArrayList<Int>()
            var seenSequence = false
            for (piece in single) {
                val item = piece as? Map<*, *> ?: continue
                val special = item["SpecialToken"] as? Map<*, *>
                if (special != null) {
                    val name = special["id"] as? String ?: continue
                    val declaredIds = (declared?.get(name) as? Map<*, *>)?.get("ids") as? List<*>
                    val id = (declaredIds?.firstOrNull() as? Number)?.toInt()
                        ?: specialTokens[name] ?: addedTokens[name] ?: vocab[name]
                        ?: throw IllegalArgumentException("Template token '$name' is not in the vocabulary")
                    if (seenSequence) suffix.add(id) else prefix.add(id)
                } else if (item["Sequence"] != null) {
                    seenSequence = true
                }
            }
            return prefix.toIntArray() to suffix.toIntArray()
        }
    }
}

