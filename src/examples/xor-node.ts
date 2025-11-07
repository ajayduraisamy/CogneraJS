import { Tensor } from '../core/tensor'
import { Model } from '../nn/model'
import { Dense } from '../nn/layers/dense'
import { backward } from '../core/autodiff'

// training XOR (toy) with naive steps (no optimizer implemented here)
const X = Tensor.fromArray([[0, 0], [0, 1], [1, 0], [1, 1]]) // shape [4,2]
const Y = Tensor.fromArray([[0], [1], [1], [0]]) // shape [4,1]

const l1 = new Dense(2, 4)
const l2 = new Dense(4, 1)
const model = new Model([l1, l2])

// Very naive training loop (for demo only)
for (let epoch = 0; epoch < 200; epoch++) {
    const preds = model.predict(X) // Tensor
    // compute MSE (preds - Y)^2 mean
    const diff = preds.add(new Tensor(Y.data.map(v => -v) as Float32Array, Y.shape)) // naive; replace with proper ops
    // For the minimal demo we'll skip doing real gradient descent here
    if (epoch % 50 === 0) console.log('epoch', epoch)
}

console.log('Done (this demo is scaffold only).')
