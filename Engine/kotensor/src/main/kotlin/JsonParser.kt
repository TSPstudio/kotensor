package io.kotensor.core

class JsonParser private constructor(private val text: String) {
    private var index = 0

    companion object {
        fun parse(text: String): Any? {
            val parser = JsonParser(text)
            parser.skipWhitespace()
            val value = parser.readValue()
            parser.skipWhitespace()
            require(parser.index == text.length) { "Unexpected trailing content at ${parser.index}" }
            return value
        }

        fun member(value: Any?, key: String): Any? {
            return if (value is Map<*, *>) value[key] else null
        }
    }

    private fun readValue(): Any? {
        require(index < text.length) { "Unexpected end of JSON" }
        return when (val c = text[index]) {
            '{' -> readObject()
            '[' -> readArray()
            '"' -> readString()
            't' -> readLiteral("true", true)
            'f' -> readLiteral("false", false)
            'n' -> readLiteral("null", null)
            else -> {
                if (c == '-' || c in '0'..'9') readNumber() else throw IllegalArgumentException("Unexpected character '$c' at $index")
            }
        }
    }

    private fun readObject(): Map<String, Any?> {
        index++
        val result = LinkedHashMap<String, Any?>()
        skipWhitespace()
        if (peek() == '}') {
            index++
            return result
        }
        while (true) {
            skipWhitespace()
            require(peek() == '"') { "Expected a string key at $index" }
            val key = readString()
            skipWhitespace()
            require(peek() == ':') { "Expected ':' at $index" }
            index++
            skipWhitespace()
            result[key] = readValue()
            skipWhitespace()
            when (peek()) {
                ',' -> index++
                '}' -> {
                    index++
                    return result
                }
                else -> throw IllegalArgumentException("Expected ',' or '}' at $index")
            }
        }
    }

    private fun readArray(): List<Any?> {
        index++
        val result = ArrayList<Any?>()
        skipWhitespace()
        if (peek() == ']') {
            index++
            return result
        }
        while (true) {
            skipWhitespace()
            result.add(readValue())
            skipWhitespace()
            when (peek()) {
                ',' -> index++
                ']' -> {
                    index++
                    return result
                }
                else -> throw IllegalArgumentException("Expected ',' or ']' at $index")
            }
        }
    }

    private fun readString(): String {
        index++
        val sb = StringBuilder()
        while (true) {
            require(index < text.length) { "Unterminated string" }
            val c = text[index++]
            when (c) {
                '"' -> return sb.toString()
                '\\' -> {
                    require(index < text.length) { "Unterminated escape" }
                    when (val e = text[index++]) {
                        '"' -> sb.append('"')
                        '\\' -> sb.append('\\')
                        '/' -> sb.append('/')
                        'b' -> sb.append('\b')
                        'f' -> sb.append('\u000C')
                        'n' -> sb.append('\n')
                        'r' -> sb.append('\r')
                        't' -> sb.append('\t')
                        'u' -> {
                            require(index + 4 <= text.length) { "Truncated \\u escape" }
                            sb.append(text.substring(index, index + 4).toInt(16).toChar())
                            index += 4
                        }
                        else -> throw IllegalArgumentException("Invalid escape '\\$e' at ${index - 1}")
                    }
                }
                else -> sb.append(c)
            }
        }
    }

    private fun readNumber(): Any {
        val start = index
        if (peek() == '-') index++
        while (index < text.length && text[index] in '0'..'9') index++
        var integral = true
        if (index < text.length && text[index] == '.') {
            integral = false
            index++
            while (index < text.length && text[index] in '0'..'9') index++
        }
        if (index < text.length && (text[index] == 'e' || text[index] == 'E')) {
            integral = false
            index++
            if (index < text.length && (text[index] == '+' || text[index] == '-')) index++
            while (index < text.length && text[index] in '0'..'9') index++
        }
        val literal = text.substring(start, index)
        if (integral) {
            val asLong = literal.toLongOrNull()
            if (asLong != null) return asLong
        }
        return literal.toDouble()
    }

    private fun readLiteral(word: String, value: Any?): Any? {
        require(text.startsWith(word, index)) { "Unexpected literal at $index" }
        index += word.length
        return value
    }

    private fun peek(): Char {
        require(index < text.length) { "Unexpected end of JSON" }
        return text[index]
    }

    private fun skipWhitespace() {
        while (index < text.length) {
            val c = text[index]
            if (c == ' ' || c == '\n' || c == '\r' || c == '\t') index++ else break
        }
    }
}

