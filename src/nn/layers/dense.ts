import { Tensor } from '../../core/tensor'

export class Dense {
    inFeatures: number
    outFeatures: number
    weight: Tensor
    bias: Tensor

    constructor(inF: number, outF: number) {
        this.inFeatures = inF
        this.outFeatures = outF
        // random init
        const w = new Float32Array(inF * outF)
        for (let i = 0; i < w.length; i++) w[i] = (Math.random() - 0.5) * 0.1
        this.weight = new Tensor(w, [inF, outF])
        this.bias = new Tensor(new Float32Array(outF), [outF])
    }

    forward(x: Tensor) {
        // naive dense: x [batch, in] matmul weight [in,out] -> [batch,out] then add bias
        return x.matmul(this.weight) // then add bias (implement broadcasting if needed)
    }
}
