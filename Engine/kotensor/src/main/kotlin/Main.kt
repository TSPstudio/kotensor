package io.kotensor.core

fun main() {
    val text = "Hola Max"
    val vocabulary = text.toSet().toList()
    val vocabularySize = vocabulary.size

    fun oneHot(c: Char): Tensor {
        val data = FloatArray(vocabularySize)
        data[vocabulary.indexOf(c)] = 1.0f
        return Tensor(data, intArrayOf(1, vocabularySize))
    }

    val dataset = mutableListOf<Pair<Tensor, Tensor>>()
    for (i in 0 until text.length - 1) {
        dataset.add(oneHot(text[i]) to oneHot(text[i + 1]))
    }

    val neurons = 32
    val l1 = Linear(vocabularySize, neurons)
    val l2 = Linear(neurons, vocabularySize)
    val optimizer = SGD(l1.parameters() + l2.parameters(), learningRate = 0.15f)

    fun generate(seed: Char, length: Int): String {
        var current = seed
        var result = "$seed"
        for (step in 0 until length - 1) {
            val input = oneHot(current)
            val pred = l2(l1(input).relu())

            var bestIdx = 0
            var bestScore = pred.data[0]
            for (i in 1 until pred.data.size) {
                if (pred.data[i] > bestScore) {
                    bestScore = pred.data[i]
                    bestIdx = i
                }
            }
            current = vocabulary[bestIdx]
            result += current
        }
        return result
    }

    println("...")

    for (epoch in 1..1000) {
        var totalLoss = 0.0f
        for ((input, target) in dataset) {
            val hidden = l1(input).relu()
            val prediction = l2(hidden)

            val loss = mseLoss(prediction, target)
            totalLoss += loss.data[0]

            loss.backward()
            optimizer.step()
            optimizer.zeroGrad()
        }

        if (epoch % 10 == 0 || epoch == 1) {
            val sample = generate('H', text.length)
            val avgLoss = totalLoss / dataset.size
            println("loop $epoch \n perdida: $avgLoss \n output: \"$sample\"")
        }
    }

    println("\n")
    println(generate('H', text.length))
}
