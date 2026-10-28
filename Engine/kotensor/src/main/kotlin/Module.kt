package io.kotensor.core

abstract class Module {
    abstract fun forward(input: Tensor): Tensor
    open fun parameters(): List<Tensor> = emptyList()
    operator fun invoke(input: Tensor): Tensor = forward(input)
}