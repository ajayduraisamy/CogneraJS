// src/core/autodiff.ts
// CogneraJS — Autodiff Engine
// Manages the computation tape and gradient propagation

import { Tensor } from './tensor'

/** Type definition for gradient function */
export type GradFn = (grad: Float32Array) => void

/** Global tape to store computation graph nodes */
const tape: Tensor[] = []

/** Push new operation result (Tensor) to the tape */
export function tapePush(t: Tensor): void {
    tape.push(t)
}

/**
 * Run backward pass starting from a given node
 * Propagates gradients through the computation graph
 */
export function backward(node: Tensor, grad?: Float32Array): void {
    // Initialize gradient if not provided
    if (!grad) grad = new Float32Array(node.data.length).fill(1)

    // Initialize node gradient storage
    if (!node.grad) node.grad = new Float32Array(node.data.length)
    for (let i = 0; i < grad.length; i++) node.grad[i] += grad[i]

    // Iterate backwards through the tape
    for (let i = tape.length - 1; i >= 0; i--) {
        const n = tape[i]
        if (n._gradFn && n.grad) {
            n._gradFn(n.grad)
        }
    }

    // Optional: clear tape after backprop to avoid double backprop
    tape.length = 0
}
