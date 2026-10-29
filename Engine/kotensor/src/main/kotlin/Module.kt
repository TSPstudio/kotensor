package io.kotensor.core

abstract class Module {
    abstract fun forward(input: Tensor): Tensor
    open fun parameters(): List<Tensor> = emptyList()
    open fun forwardInto(input: Tensor, out: Tensor): Tensor = out.copyFrom(forward(input))
    operator fun invoke(input: Tensor): Tensor = forward(input)
}

