// src/core/tensor.ts
// CogneraJS — Core Tensor class
// TypeScript implementation for CogneraJS ML Framework
// Provides: Tensor creation, arithmetic ops, matmul, relu, autodiff integration

import { GradFn, tapePush, backward } from './autodiff'

// Type alias for tensor shape (dimensions)
export type Shape = number[]

export class Tensor {
    data: Float32Array
    shape: Shape
    grad: Float32Array | null = null
    _gradFn?: GradFn | null = null
    _parents: Tensor[] = []

    constructor(data: Float32Array, shape: Shape) {
        this.data = data
        this.shape = shape
    }

    /** Create Tensor from array (1D or 2D only for now) */
    static fromArray(arr: number[] | number[][]): Tensor {
        if (Array.isArray(arr[0])) {
            const rows = arr as number[][]
            const r = rows.length
            const c = (rows[0] as number[]).length
            const flat = new Float32Array(r * c)
            for (let i = 0; i < r; i++) {
                for (let j = 0; j < c; j++) {
                    flat[i * c + j] = rows[i][j]
                }
            }
            return new Tensor(flat, [r, c])
        } else {
            const flat = new Float32Array(arr as number[])
            return new Tensor(flat, [flat.length])
        }
    }

    /** Compare shapes for equality */
    shapeEquals(s: Shape): boolean {
        if (s.length !== this.shape.length) return false
        for (let i = 0; i < s.length; i++) if (s[i] !== this.shape[i]) return false
        return true
    }

    /** Elementwise addition */
    add(b: Tensor): Tensor {
        if (this.data.length !== b.data.length) throw new Error('shape mismatch in add')
        const out = new Float32Array(this.data.length)
        for (let i = 0; i < out.length; i++) out[i] = this.data[i] + b.data[i]

        const res = new Tensor(out, this.shape)
        res._parents = [this, b]
        res._gradFn = (g: Float32Array) => {
            this._accumulateGrad(g)
            b._accumulateGrad(g)
        }

        tapePush(res)
        return res
    }

    /** Elementwise multiplication */
    mul(b: Tensor): Tensor {
        if (this.data.length !== b.data.length) throw new Error('shape mismatch in mul')
        const out = new Float32Array(this.data.length)
        for (let i = 0; i < out.length; i++) out[i] = this.data[i] * b.data[i]

        const res = new Tensor(out, this.shape)
        res._parents = [this, b]
        res._gradFn = (g: Float32Array) => {
            const ga = new Float32Array(g.length)
            const gb = new Float32Array(g.length)
            for (let i = 0; i < g.length; i++) {
                ga[i] = g[i] * b.data[i]
                gb[i] = g[i] * this.data[i]
            }
            this._accumulateGrad(ga)
            b._accumulateGrad(gb)
        }

        tapePush(res)
        return res
    }

    /** Matrix multiplication (2D only) */
    matmul(b: Tensor): Tensor {
        const [m, k] = this.shape
        const [k2, n] = b.shape
        if (k !== k2) throw new Error('matmul shape mismatch')

        const out = new Float32Array(m * n)
        for (let i = 0; i < m; i++) {
            for (let j = 0; j < n; j++) {
                let s = 0
                for (let t = 0; t < k; t++) s += this.data[i * k + t] * b.data[t * n + j]
                out[i * n + j] = s
            }
        }

        const res = new Tensor(out, [m, n])
        res._parents = [this, b]
        res._gradFn = (g: Float32Array) => {
            const dA = new Float32Array(m * k)
            const dB = new Float32Array(k * n)

            // dA = g @ B^T
            for (let i = 0; i < m; i++) {
                for (let t = 0; t < k; t++) {
                    let s = 0
                    for (let j = 0; j < n; j++) s += g[i * n + j] * b.data[t * n + j]
                    dA[i * k + t] = s
                }
            }

            // dB = A^T @ g
            for (let t = 0; t < k; t++) {
                for (let j = 0; j < n; j++) {
                    let s = 0
                    for (let i = 0; i < m; i++) s += this.data[i * k + t] * g[i * n + j]
                    dB[t * n + j] = s
                }
            }

            this._accumulateGrad(dA)
            b._accumulateGrad(dB)
        }

        tapePush(res)
        return res
    }

    /** ReLU activation */
    relu(): Tensor {
        const out = new Float32Array(this.data.length)
        for (let i = 0; i < out.length; i++) out[i] = Math.max(0, this.data[i])

        const res = new Tensor(out, this.shape)
        res._parents = [this]
        res._gradFn = (g: Float32Array) => {
            const dg = new Float32Array(g.length)
            for (let i = 0; i < g.length; i++) dg[i] = this.data[i] > 0 ? g[i] : 0
            this._accumulateGrad(dg)
        }

        tapePush(res)
        return res
    }

    /** Internal: accumulate gradient values */
    private _accumulateGrad(gradArr: Float32Array): void {
        if (!this.grad) this.grad = new Float32Array(gradArr.length)
        for (let i = 0; i < gradArr.length; i++) this.grad[i] += gradArr[i]
    }

    /** Trigger backpropagation through computation graph */
    backward(grad?: Float32Array): void {
        backward(this, grad)
    }

    /** Utility to print tensor contents (for debugging) */
    toString(): string {
        return `Tensor(shape=[${this.shape.join(',')}], data=[${Array.from(this.data)
            .slice(0, 10)
            .map(x => x.toFixed(3))
            .join(', ')}${this.data.length > 10 ? '...' : ''}])`
    }
}
