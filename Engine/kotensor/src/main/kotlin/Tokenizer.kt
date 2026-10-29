package io.kotensor.core

interface Tokenizer {
    val vocabularySize: Int
    fun encode(text: String): IntArray
    fun decode(tokens: IntArray): String
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
