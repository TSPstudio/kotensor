package io.kotensor.core

interface Tokenizer {
    val vocabularySize: Int
    fun encode(text: String): IntArray
    fun decode(tokens: IntArray): String
}

class IncrementalDecoder(val tokenizer: Tokenizer) {
    private val pending = mutableListOf<Int>()

    fun push(token: Int): String {
        pending.add(token)
        val text = tokenizer.decode(pending.toIntArray())
        if (text.endsWith(REPLACEMENT_CHAR) && pending.size < MAX_PENDING) {
            return ""
        }
        pending.clear()
        return text
    }

    fun flush(): String {
        if (pending.isEmpty()) return ""
        val text = tokenizer.decode(pending.toIntArray())
        pending.clear()
        return text
    }

    private companion object {
        const val REPLACEMENT_CHAR = '�'

        const val MAX_PENDING = 4
    }
}

class SimpleTokenizer(
    val textBody: String
) : Tokenizer {
    private val charToId = mutableMapOf<Char, Int>()
    private val idToChar = mutableMapOf<Int, Char>()

    init {
        val uniqueChars = textBody.toSet().toList()
        for (i in uniqueChars.indices) {
            val c = uniqueChars[i]
            charToId[c] = i
            idToChar[i] = c
        }
    }

    override val vocabularySize: Int
        get() = charToId.size

    override fun encode(text: String): IntArray {
        val result = IntArray(text.length)
        for (i in text.indices) {
            result[i] = charToId[text[i]] ?: 0
        }
        return result
    }

    override fun decode(tokens: IntArray): String {
        val sb = StringBuilder()
        for (token in tokens) {
            sb.append(idToChar[token] ?: "")
        }
        return sb.toString()
    }
}

