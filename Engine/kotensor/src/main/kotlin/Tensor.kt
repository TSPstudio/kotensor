package io.kotensor.core

import kotlin.math.pow

class Tensor (
    val data: FloatArray,
    val shape: IntArray,
    val parents: List<Tensor> = emptyList(),
    val backwardOp: () -> Unit = {}
){
    var requiresGrad: Boolean = inheritsGrad(parents)

    private var gradient: FloatArray? = null

    val grad: FloatArray
        get() = gradient ?: FloatArray(data.size).also { gradient = it }

    val hasGrad: Boolean
        get() = gradient != null

    init {
        var expectedSize = 1
        for(dim in shape){
            expectedSize *= dim
        }
        require(data.size == expectedSize){
            "${data.size} != $expectedSize"
        }
    }

    operator fun plus(other: Tensor): Tensor {
        if (this.shape.contentEquals(other.shape)) {
            val outData = FloatArray(data.size) { i -> data[i] + other.data[i] }
            lateinit var out: Tensor
            out = Tensor(outData, shape, listOf(this, other)) {
                val outGrad = out.grad
                if (this.requiresGrad) {
                    val g = this.grad
                    for (i in data.indices) g[i] += outGrad[i]
                }
                if (other.requiresGrad) {
                    val g = other.grad
                    for (i in data.indices) g[i] += outGrad[i]
                }
            }
            return out
        }

        if (this.shape.size == 2 && other.shape.size == 2 && this.shape[1] == other.shape[1]) {
            if (other.shape[0] == 1) {
                val rows = this.shape[0]
                val cols = this.shape[1]
                val outData = FloatArray(data.size)
                for (r in 0 until rows) {
                    val rowOffset = r * cols
                    for (c in 0 until cols) {
                        outData[rowOffset + c] = this.data[rowOffset + c] + other.data[c]
                    }
                }
                lateinit var out: Tensor
                out = Tensor(outData, this.shape.copyOf(), listOf(this, other)) {
                    accumulateBroadcast(out.grad, this, other, rows, cols)
                }
                return out
            } else if (this.shape[0] == 1) {
                val rows = other.shape[0]
                val cols = other.shape[1]
                val outData = FloatArray(other.data.size)
                for (r in 0 until rows) {
                    val rowOffset = r * cols
                    for (c in 0 until cols) {
                        outData[rowOffset + c] = this.data[c] + other.data[rowOffset + c]
                    }
                }
                lateinit var out: Tensor
                out = Tensor(outData, other.shape.copyOf(), listOf(this, other)) {
                    accumulateBroadcast(out.grad, other, this, rows, cols)
                }
                return out
            }
        }

        throw IllegalArgumentException("Shape mismatch: ${this.shape.contentToString()} and ${other.shape.contentToString()}")
    }

    operator fun minus(other: Tensor):Tensor{
        require(this.shape.contentEquals(other.shape))
        val outData = FloatArray(data.size) { i -> data[i] - other.data[i] }
        lateinit var out: Tensor
        out = Tensor(outData,shape,listOf(this, other)){
            val outGrad = out.grad
            if (this.requiresGrad) {
                val g = this.grad
                for (i in data.indices) g[i] += outGrad[i]
            }
            if (other.requiresGrad) {
                val g = other.grad
                for (i in data.indices) g[i] += -(outGrad[i])
            }
        }
        return out
    }

    operator fun times(other: Tensor):Tensor{
        require(this.shape.contentEquals(other.shape))
        val outData = FloatArray(data.size) { i -> data[i] * other.data[i] }
        lateinit var out: Tensor
        out = Tensor(outData,shape,listOf(this, other)){
            val outGrad = out.grad
            if (this.requiresGrad) {
                val g = this.grad
                for (i in data.indices) g[i] += outGrad[i] * other.data[i]
            }
            if (other.requiresGrad) {
                val g = other.grad
                for (i in data.indices) g[i] += outGrad[i] * this.data[i]
            }
        }
        return out
    }

    operator fun times(scalar: Float): Tensor {
        val outData = FloatArray(data.size) { i -> data[i] * scalar }
        lateinit var out: Tensor
        out = Tensor(outData, shape.copyOf(), listOf(this)) {
            if (this.requiresGrad) {
                val outGrad = out.grad
                val g = this.grad
                for (i in data.indices) g[i] += outGrad[i] * scalar
            }
        }
        return out
    }

    infix fun matmul(other: Tensor): Tensor {
        require(this.shape.size == 2 && other.shape.size == 2) { "not matrices" }
        val m = this.shape[0]
        val k = this.shape[1]
        val otherk = other.shape[0]
        val n = other.shape[1]
        require(k == otherk) { "$k != $otherk" }
        val outShape = intArrayOf(m, n)
        val outData = FloatArray(m * n)
        MatMul.multiply(this.data, other.data, outData, m, k, n, accumulate = false)

        lateinit var out: Tensor
        out = Tensor(outData, outShape, listOf(this, other)) {
            if (this.requiresGrad) {
                MatMul.multiplyTransposedB(out.grad, other.data, this.grad, m, n, k)
            }
            if (other.requiresGrad) {
                MatMul.multiplyTransposedA(this.data, out.grad, other.grad, k, m, n)
            }
        }
        return out
    }

    fun matmulDequant(weight: QuantizedTensor): Tensor {
        require(this.shape.size == 2) { "not a matrix" }
        val m = this.shape[0]
        val k = this.shape[1]
        require(k == weight.rows) { "$k != ${weight.rows}" }
        val n = weight.columns
        val outData = FloatArray(m * n)
        MatMul.multiplyQuantized(this.data, weight, outData, m, accumulate = false)

        lateinit var out: Tensor
        out = Tensor(outData, intArrayOf(m, n), listOf(this)) {
            if (this.requiresGrad) {
                MatMul.multiplyByQuantizedTransposed(out.grad, weight, this.grad, m)
            }
        }
        return out
    }

    fun matmulDequantInto(weight: QuantizedTensor, out: Tensor, accumulate: Boolean = false, scale: Float = 1.0f): Tensor {
        require(this.shape.size == 2 && out.shape.size == 2) { "not matrices" }
        val m = this.shape[0]
        val k = this.shape[1]
        require(k == weight.rows) { "$k != ${weight.rows}" }
        require(out.shape[0] == m && out.shape[1] == weight.columns) { "out shape ${out.shape.contentToString()} != [$m, ${weight.columns}]" }
        MatMul.multiplyQuantized(this.data, weight, out.data, m, accumulate, scale)
        return out
    }

    fun matmulInto(other: Tensor, out: Tensor, accumulate: Boolean = false, scale: Float = 1.0f): Tensor {
        require(this.shape.size == 2 && other.shape.size == 2 && out.shape.size == 2) { "not matrices" }
        val m = this.shape[0]
        val k = this.shape[1]
        val n = other.shape[1]
        require(k == other.shape[0]) { "$k != ${other.shape[0]}" }
        require(out.shape[0] == m && out.shape[1] == n) { "out shape ${out.shape.contentToString()} != [$m, $n]" }
        MatMul.multiply(this.data, other.data, out.data, m, k, n, accumulate, scale)
        return out
    }

    fun addInto(other: Tensor, out: Tensor): Tensor {
        require(out.data.size == data.size) { "out has ${out.data.size} elements, this has ${data.size}" }
        if (other.data.size == data.size) {
            for (i in data.indices) {
                out.data[i] = data[i] + other.data[i]
            }
            return out
        }
        require(data.size % other.data.size == 0) { "Shape mismatch: ${shape.contentToString()} and ${other.shape.contentToString()}" }
        val cols = other.data.size
        val rows = data.size / cols
        for (r in 0 until rows) {
            val rowOffset = r * cols
            for (c in 0 until cols) {
                out.data[rowOffset + c] = data[rowOffset + c] + other.data[c]
            }
        }
        return out
    }

    fun addInPlace(other: Tensor): Tensor = addInto(other, this)

    fun mulInPlace(other: Tensor): Tensor {
        require(other.data.size == data.size) { "Shape mismatch: ${shape.contentToString()} and ${other.shape.contentToString()}" }
        for (i in data.indices) {
            data[i] *= other.data[i]
        }
        return this
    }

    fun siluInPlace(): Tensor {
        for (i in data.indices) {
            val x = data[i]
            data[i] = x / (1.0f + kotlin.math.exp(-x))
        }
        return this
    }

    fun copyFrom(source: Tensor): Tensor {
        require(source.data.size == data.size) { "Size mismatch: ${source.data.size} != ${data.size}" }
        System.arraycopy(source.data, 0, data, 0, data.size)
        return this
    }

    fun reshape(newShape: IntArray): Tensor {
        val outData = data.copyOf()
        lateinit var out: Tensor
        out = Tensor(outData, newShape, listOf(this)){
            if (this.requiresGrad) {
                val outGrad = out.grad
                val g = this.grad
                for (i in data.indices) g[i] += outGrad[i]
            }
        }
        return out
    }

    fun transpose(): Tensor{
        require(this.shape.size == 2)
        val m = this.shape[0]
        val n = this.shape[1]
        val outShape = intArrayOf(n, m)
        val outData = FloatArray(data.size)

        for(i in 0 until m){
            for(j in 0 until n){
                outData[j * m + i] = this.data[i * n + j]
            }
        }

        lateinit var out: Tensor
        out = Tensor(outData, outShape, listOf(this)){
            if (this.requiresGrad) {
                val outGrad = out.grad
                val g = this.grad
                for(i in 0 until m){
                    for(j in 0 until n){
                        g[i * n + j] += outGrad[j * m + i]
                    }
                }
            }
        }
        return out
    }

    fun backward(){
        if (!requiresGrad) return
        val topology = mutableListOf<Tensor>()
        val visited = mutableSetOf<Tensor>()

        fun buildTopology(v: Tensor){
            if(visited.add(v)){
                for (parent in v.parents) {
                    if (parent.requiresGrad) buildTopology(parent)
                }
                topology.add(v)
            }
        }

        buildTopology(this)
        this.grad.fill(1.0f)
        for(node in topology.asReversed()){
            node.backwardOp()
        }
    }

    fun relu(): Tensor{
        val outData = FloatArray(data.size){i -> if(data[i] > 0f) data[i] else 0f}
        lateinit var out: Tensor
        out = Tensor(outData, shape, listOf(this)){
            if (this.requiresGrad) {
                val outGrad = out.grad
                val g = this.grad
                for(i in data.indices){
                    if(data[i] > 0f){
                        g[i] += outGrad[i]
                    }
                }
            }
        }
        return out
    }

    fun silu(): Tensor{
        val outData = FloatArray(data.size)
        val sigmoids = FloatArray(data.size)
        for(i in data.indices){
            val x = data[i]
            val s = 1.0f / (1.0f + kotlin.math.exp(-x))
            sigmoids[i] = s
            outData[i] = x * s
        }
        lateinit var out: Tensor
        out = Tensor(outData, shape, listOf(this)){
            if (this.requiresGrad) {
                val outGrad = out.grad
                val g = this.grad
                for(i in data.indices){
                    val s = sigmoids[i]
                    val rate = s * (1.0f + data[i] * (1.0f - s))
                    g[i] += outGrad[i] * rate
                }
            }
        }
        return out
    }

    fun gelu(): Tensor{
        val constant = 0.79788456f
        val outData = FloatArray(data.size)
        val tanhValues = FloatArray(data.size)
        for(i in data.indices){
            val x = data[i]
            val inner = constant * (x + 0.044715f * x * x * x)
            val t = kotlin.math.tanh(inner)
            tanhValues[i] = t
            outData[i] = 0.5f * x * (1.0f + t)
        }
        lateinit var out: Tensor
        out = Tensor(outData, shape, listOf(this)){
            if (this.requiresGrad) {
                val outGrad = out.grad
                val g = this.grad
                for(i in data.indices){
                    val x = data[i]
                    val t = tanhValues[i]
                    val sech2 = 1.0f - t * t
                    val dInner = constant * (1.0f + 3.0f * 0.044715f * x * x)
                    val rate = 0.5f * (1.0f + t) + 0.5f * x * sech2 * dInner
                    g[i] += outGrad[i] * rate
                }
            }
        }
        return out
    }
}

private fun inheritsGrad(parents: List<Tensor>): Boolean {
    if (parents.isEmpty()) return true
    for (i in parents.indices) {
        if (parents[i].requiresGrad) return true
    }
    return false
}

private fun accumulateBroadcast(outGrad: FloatArray, full: Tensor, row: Tensor, rows: Int, cols: Int) {
    if (full.requiresGrad) {
        val g = full.grad
        for (i in 0 until rows * cols) g[i] += outGrad[i]
    }
    if (row.requiresGrad) {
        val g = row.grad
        for (r in 0 until rows) {
            val rowOffset = r * cols
            for (c in 0 until cols) {
                g[c] += outGrad[rowOffset + c]
            }
        }
    }
}
