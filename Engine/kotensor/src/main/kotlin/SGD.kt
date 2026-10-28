package io.kotensor.core

class SGD(
    val parameters: List<Tensor>,
    val learningRate: Float = 0.01f
) {
    fun step(){
        for(tensor in parameters){
            for(i in tensor.data.indices){
                tensor.data[i] -= learningRate * tensor.grad[i]
            }
        }
    }

    fun zeroGrad(){
        for(tensor in parameters){
            tensor.grad.fill(0.0f)
        }
    }
}