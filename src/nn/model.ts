import { Dense } from './layers/dense'
import { Tensor } from '../core/tensor'

export class Model {
    layers: any[]
    constructor(layers: any[]) {
        this.layers = layers
    }
    predict(x: Tensor) {
        let out = x
        for (const L of this.layers) out = L.forward(out)
        return out
    }
}
