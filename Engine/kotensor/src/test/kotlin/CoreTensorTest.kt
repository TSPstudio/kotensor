package io.kotensor.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class CoreTensorTest {

    @Test
    fun testTransposeAndReshape() {
        val data = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f)
        val original = Tensor(data, intArrayOf(2, 3))

        val transposed = original.transpose()
        assertTrue(transposed.shape.contentEquals(intArrayOf(3, 2)))
        assertEquals(1f, transposed.data[0])
        assertEquals(4f, transposed.data[1])
        assertEquals(2f, transposed.data[2])
        assertEquals(5f, transposed.data[3])
        assertEquals(3f, transposed.data[4])
        assertEquals(6f, transposed.data[5])

        transposed.backward()
        assertEquals(1f, original.grad[0])
        assertEquals(1f, original.grad[5])

        val reshaped = original.reshape(intArrayOf(6, 1))
        assertTrue(reshaped.shape.contentEquals(intArrayOf(6, 1)))
        assertEquals(6, reshaped.data.size)
    }

    @Test
    fun testActivations() {
        val inputData = floatArrayOf(-2f, -1f, 0f, 1f, 2f)
        val input = Tensor(inputData, intArrayOf(1, 5))

        val reluOut = input.relu()
        assertEquals(0f, reluOut.data[0])
        assertEquals(0f, reluOut.data[1])
        assertEquals(0f, reluOut.data[2])
        assertEquals(1f, reluOut.data[3])
        assertEquals(2f, reluOut.data[4])

        reluOut.backward()
        assertEquals(0f, input.grad[0])
        assertEquals(1f, input.grad[3])
        assertEquals(1f, input.grad[4])

        val siluInput = Tensor(floatArrayOf(1f, -1f), intArrayOf(1, 2))
        val siluOut = siluInput.silu()
        assertTrue(siluOut.data[0] > 0f)
        assertTrue(siluOut.data[1] < 0f)

        siluOut.backward()
        assertTrue(siluInput.grad.all { !it.isNaN() })

        val geluInput = Tensor(floatArrayOf(1f, -1f), intArrayOf(1, 2))
        val geluOut = geluInput.gelu()
        assertTrue(geluOut.data[0] > 0f)
        assertTrue(geluOut.data[1] < 0f)

        geluOut.backward()
        assertTrue(geluInput.grad.all { !it.isNaN() })
    }

    @Test
    fun testTensorArithmeticAndBroadcasting() {
        val a = Tensor(floatArrayOf(1f, 2f, 3f, 4f), intArrayOf(2, 2))
        val b = Tensor(floatArrayOf(10f, 20f), intArrayOf(1, 2))

        val broadcastSum = a + b
        assertEquals(11f, broadcastSum.data[0])
        assertEquals(22f, broadcastSum.data[1])
        assertEquals(13f, broadcastSum.data[2])
        assertEquals(24f, broadcastSum.data[3])

        broadcastSum.backward()
        assertEquals(2f, b.grad[0])
        assertEquals(2f, b.grad[1])

        val diff = a - Tensor(floatArrayOf(1f, 1f, 1f, 1f), intArrayOf(2, 2))
        assertEquals(0f, diff.data[0])
        assertEquals(3f, diff.data[3])

        val scaled = a * 2.5f
        assertEquals(2.5f, scaled.data[0])
        assertEquals(10f, scaled.data[3])
    }

    @Test
    fun testLinearAndSGDConvergence() {
        val linear = Linear(1, 1)
        val optimizer = SGD(linear.parameters(), learningRate = 0.1f)

        val x = Tensor(floatArrayOf(2f), intArrayOf(1, 1))
        val target = Tensor(floatArrayOf(10f), intArrayOf(1, 1))

        var initialLoss = 0f
        var finalLoss = 0f

        for (epoch in 0 until 50) {
            val pred = linear(x)
            val loss = mseLoss(pred, target)

            if (epoch == 0) initialLoss = loss.data[0]
            if (epoch == 49) finalLoss = loss.data[0]

            loss.backward()
            optimizer.step()
            optimizer.zeroGrad()
        }

        assertTrue(finalLoss < initialLoss)
        assertTrue(finalLoss < 0.1f)
    }
}

