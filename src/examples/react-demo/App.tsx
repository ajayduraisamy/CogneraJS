import React from 'react'
import { Tensor } from '../../core/tensor'

export default function App() {
    const a = Tensor.fromArray([[1, 2]])
    const b = Tensor.fromArray([[3, 4]])
    const c = a.add(b)
    return (
        <div style={{ padding: 20 }}>
            <h1>CogneraJS Demo</h1>
            <p>Tensor a: {Array.from(a.data).join(', ')}</p>
            <p>Tensor b: {Array.from(b.data).join(', ')}</p>
            <p>Sum: {Array.from(c.data).join(', ')}</p>
            <p>Open dev server with <code>npm run dev</code></p>
        </div>
    )
}
