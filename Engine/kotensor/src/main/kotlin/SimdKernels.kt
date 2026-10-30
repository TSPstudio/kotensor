package io.kotensor.core

import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.VectorOperators
import jdk.incubator.vector.VectorSpecies

@JvmField
internal val FLOAT_SPECIES: VectorSpecies<Float> = FloatVector.SPECIES_PREFERRED

internal class SimdKernels : VectorKernels {

    override fun axpy(y: FloatArray, x: FloatArray, a: Float, len: Int) {
        val species = FLOAT_SPECIES
        val step = species.length()
        val bound = species.loopBound(len)
        val va = FloatVector.broadcast(species, a)
        var j = 0
        while (j < bound) {
            FloatVector.fromArray(species, x, j).fma(va, FloatVector.fromArray(species, y, j)).intoArray(y, j)
            j += step
        }
        while (j < len) {
            y[j] += a * x[j]
            j++
        }
    }

    override fun axpy4(
        y0: FloatArray, y1: FloatArray, y2: FloatArray, y3: FloatArray, x: FloatArray,
        a0: Float, a1: Float, a2: Float, a3: Float, len: Int
    ) {
        val species = FLOAT_SPECIES
        val step = species.length()
        val bound = species.loopBound(len)
        val va0 = FloatVector.broadcast(species, a0)
        val va1 = FloatVector.broadcast(species, a1)
        val va2 = FloatVector.broadcast(species, a2)
        val va3 = FloatVector.broadcast(species, a3)
        var j = 0
        while (j < bound) {
            val xv = FloatVector.fromArray(species, x, j)
            xv.fma(va0, FloatVector.fromArray(species, y0, j)).intoArray(y0, j)
            xv.fma(va1, FloatVector.fromArray(species, y1, j)).intoArray(y1, j)
            xv.fma(va2, FloatVector.fromArray(species, y2, j)).intoArray(y2, j)
            xv.fma(va3, FloatVector.fromArray(species, y3, j)).intoArray(y3, j)
            j += step
        }
        while (j < len) {
            val xv = x[j]
            y0[j] += a0 * xv
            y1[j] += a1 * xv
            y2[j] += a2 * xv
            y3[j] += a3 * xv
            j++
        }
    }

    override fun dot(a: FloatArray, aOffset: Int, b: FloatArray, bOffset: Int, len: Int): Float {
        val species = FLOAT_SPECIES
        val step = species.length()
        val bound = species.loopBound(len)
        var acc0 = FloatVector.zero(species)
        var acc1 = FloatVector.zero(species)
        var j = 0
        while (j + step < bound) {
            acc0 = FloatVector.fromArray(species, a, aOffset + j).fma(FloatVector.fromArray(species, b, bOffset + j), acc0)
            acc1 = FloatVector.fromArray(species, a, aOffset + j + step).fma(FloatVector.fromArray(species, b, bOffset + j + step), acc1)
            j += 2 * step
        }
        while (j < bound) {
            acc0 = FloatVector.fromArray(species, a, aOffset + j).fma(FloatVector.fromArray(species, b, bOffset + j), acc0)
            j += step
        }
        var sum = acc0.add(acc1).reduceLanes(VectorOperators.ADD)
        while (j < len) {
            sum += a[aOffset + j] * b[bOffset + j]
            j++
        }
        return sum
    }

    override fun dot4(a: FloatArray, aOffset: Int, b: FloatArray, b0: Int, b1: Int, b2: Int, b3: Int, len: Int, out: FloatArray) {
        val species = FLOAT_SPECIES
        val step = species.length()
        val bound = species.loopBound(len)
        var acc0 = FloatVector.zero(species)
        var acc1 = FloatVector.zero(species)
        var acc2 = FloatVector.zero(species)
        var acc3 = FloatVector.zero(species)
        var j = 0
        while (j < bound) {
            val av = FloatVector.fromArray(species, a, aOffset + j)
            acc0 = av.fma(FloatVector.fromArray(species, b, b0 + j), acc0)
            acc1 = av.fma(FloatVector.fromArray(species, b, b1 + j), acc1)
            acc2 = av.fma(FloatVector.fromArray(species, b, b2 + j), acc2)
            acc3 = av.fma(FloatVector.fromArray(species, b, b3 + j), acc3)
            j += step
        }
        var s0 = acc0.reduceLanes(VectorOperators.ADD)
        var s1 = acc1.reduceLanes(VectorOperators.ADD)
        var s2 = acc2.reduceLanes(VectorOperators.ADD)
        var s3 = acc3.reduceLanes(VectorOperators.ADD)
        while (j < len) {
            val av = a[aOffset + j]
            s0 += av * b[b0 + j]
            s1 += av * b[b1 + j]
            s2 += av * b[b2 + j]
            s3 += av * b[b3 + j]
            j++
        }
        out[0] = s0
        out[1] = s1
        out[2] = s2
        out[3] = s3
    }
}
