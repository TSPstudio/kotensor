package io.kotensor.core

import kotlin.math.pow
import kotlin.math.sqrt

class AdamW(
    val parameters: List<Tensor>,
    val learningRate: Float = 0.0001f,
    val beta1: Float = 0.9f,
    val beta2: Float = 0.999f,
    val epsilon: Float = 1e-8f,
    val weightDecay: Float = 0.01f
){
    var stepCount: Int = 0
    val momentum = parameters.map { tensor -> FloatArray(tensor.data.size) }
    val variance = parameters.map { tensor -> FloatArray(tensor.data.size) }

    fun step(){
        stepCount++
        val correction1 = 1.0f - beta1.pow(stepCount)
        val correction2 = 1.0f - beta2.pow(stepCount)

        for(p in parameters.indices){
            val tensor = parameters[p]
            val m = momentum[p]
            val v = variance[p]

            for(i in tensor.data.indices){
                tensor.data[i] -= learningRate * weightDecay * tensor.data[i]

                val g = tensor.grad[i]
                m[i] = beta1 * m[i] + (1.0f - beta1) * g
                v[i] = beta2 * v[i] + (1.0f - beta2) * g * g

                val mHat = m[i] / correction1
                val vHat = v[i] / correction2

                tensor.data[i] -= learningRate * mHat / (sqrt(vHat) + epsilon)
            }
        }
    }

    fun zeroGrad(){
        for(tensor in parameters){
            tensor.grad.fill(0.0f)
        }
    }
}

