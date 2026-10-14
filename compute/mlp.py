"""Forward pass of a small multilayer perceptron (MLP), in plain numpy.

This is the single source of truth for the numbers shown in the web app and the Manim scene.
The TypeScript port in web/src/experiments/mlp-forward/model.ts must agree with it
(checked against assets/mlp/golden.json, tolerance documented in export.py).

Conventions (same in TypeScript):
  - Layers are numbered 0..L. Layer 0 is the input, layer L is the output.
  - weights[l-1][i][j] connects neuron j of layer l-1 to neuron i of layer l.
  - biases[l-1][i] belongs to neuron i of layer l.
  - pre_activations[l-1] is z of layer l;  activations[l] is a of layer l  (activations[0] = input).
  - Hidden layers use the chosen activation function. The output layer is softmax over its z values.
"""
from __future__ import annotations

import numpy as np

ACTIVATIONS = {
    "tanh": np.tanh,
    "relu": lambda z: np.maximum(z, 0.0),
    "sigmoid": lambda z: 1.0 / (1.0 + np.exp(-z)),
}


def softmax(z: np.ndarray) -> np.ndarray:
    # Subtracting the max does not change the result but avoids overflow in exp.
    e = np.exp(z - np.max(z))
    return e / e.sum()


def forward(weights, biases, x, activation: str) -> dict:
    """Run the network. Returns pre_activations, activations and per-neuron contributions."""
    act = ACTIVATIONS[activation]
    a = np.asarray(x, dtype=float)
    pre_activations: list[list[float]] = []
    activations: list[list[float]] = [a.tolist()]
    contributions: list[list[list[float]]] = []  # contributions[l-1][i][j] = w_ij * a_j

    n_layers = len(weights)
    for l, (w, b) in enumerate(zip(weights, biases), start=1):
        w = np.asarray(w, dtype=float)
        b = np.asarray(b, dtype=float)
        contributions.append((w * a[None, :]).tolist())
        z = w @ a + b
        a = softmax(z) if l == n_layers else act(z)
        pre_activations.append(z.tolist())
        activations.append(a.tolist())

    return {
        "pre_activations": pre_activations,
        "activations": activations,
        "contributions": contributions,
    }
