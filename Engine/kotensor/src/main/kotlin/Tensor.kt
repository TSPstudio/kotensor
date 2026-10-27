package io.kotensor.core

class Tensor (
    val data: FloatArray,
    val shape: IntArray,
    val parents: List<Tensor> = emptyList(),
    val backwardOp: () -> Unit = {}
){
    val grad = FloatArray(data.size)

    init {
        var expectedSize = 1
        for(dim in shape){
            expectedSize *= dim
        }
        require(data.size == expectedSize){
            "${data.size} != $expectedSize"
        }
    }

    operator fun plus(other: Tensor): Tensor{
        require(this.shape.contentEquals(other.shape))
        val outData = FloatArray(data.size) { i -> data[i] + other.data[i] }
        lateinit var out: Tensor
        out = Tensor(outData,shape,listOf(this, other)){
            for(i in data.indices){
                this.grad[i] += out.grad[i]
                other.grad[i] += out.grad[i]
            }
        }
        return out
    }

    operator fun minus(other: Tensor):Tensor{
        require(this.shape.contentEquals(other.shape))
        val outData = FloatArray(data.size) { i -> data[i] - other.data[i] }
        lateinit var out: Tensor
        out = Tensor(outData,shape,listOf(this, other)){
            for(i in data.indices){
                this.grad[i] += out.grad[i]
                other.grad[i] += -(out.grad[i])
            }
        }
        return out
    }

    operator fun times(other: Tensor):Tensor{
        require(this.shape.contentEquals(other.shape))
        val outData = FloatArray(data.size) { i -> data[i] * other.data[i] }
        lateinit var out: Tensor
        out = Tensor(outData,shape,listOf(this, other)){
            for(i in data.indices){
                this.grad[i] += out.grad[i] * other.data[i]
                other.grad[i] += out.grad[i] * this.data[i]
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

            for (i in 0 until m) {
                for (j in 0 until n) {
                    var sum = 0.0f
                    for (l in 0 until k) {
                        sum += this.data[i * k + l] * other.data[l * n + j]
                    }
                    outData[i * n + j] = sum
                }
            }

            lateinit var out: Tensor
            out = Tensor(outData, outShape, listOf(this, other)) {
                for (i in 0 until m) {
                    for (j in 0 until n) {
                        val outGrad = out.grad[i * n + j]
                        for (l in 0 until k) {
                            this.grad[i * k + l] += outGrad * other.data[l * n + j]
                            other.grad[l * n + j] += outGrad * this.data[i * k + l]
                        }
                    }
                }
            }
            return out
    }

    fun reshape(newShape: IntArray): Tensor {
        val outData = data.copyOf()
        lateinit var out: Tensor
        out = Tensor(outData, newShape, listOf(this)){
            for(i in data.indices){
                this.grad[i] += out.grad[i]
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
            for(i in 0 until m){
                for(j in 0 until n){
                    this.grad[i * n + j] += out.grad[j * m + i]
                }
            }
        }
        return out  
    }

    fun backward(){
        val topology = mutableListOf<Tensor>()
        val visited = mutableSetOf<Tensor>()

        fun buildTopology(v: Tensor){
            if(visited.add(v)){
                v.parents.forEach {v -> buildTopology(v)}
                topology.add(v)
            }
        }

        buildTopology(this)
        this.grad.fill(1.0f)
        for(node in topology.asReversed()){
            node.backwardOp()
        }
    }
}