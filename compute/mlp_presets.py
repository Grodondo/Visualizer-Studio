"""Networks used by the 'MLP forward pass' experiment.

Both are SCHEMATIC: the weights are not learned from data (training arrives with experiment B).
The numbers computed *from* them by compute.mlp.forward are exact.
"""
from __future__ import annotations

import numpy as np

# Random preset: one fixed seed so the page and the video always show the same network.
RANDOM_SEED = 7


def _r(a) -> list:
    # 2 decimals keeps displayed weights tidy; JSON round-trips floats exactly, so TS and Python agree.
    return np.round(np.asarray(a, dtype=float), 2).tolist()


def xor_preset() -> dict:
    """Inputs in [0, 1]. Hand-built so every hidden unit has a nameable job (valid for tanh).

    Layer 1 detects simple logic of (x1, x2); layer 2 combines them; the output says whether
    the two inputs differ (XOR). Weights are large (+-4) so tanh behaves like a switch near 0/1 inputs.
    """
    weights = [
        # layer 1: OR, AND, NAND, NOR of (x1, x2)
        [[4, 4], [4, 4], [-4, -4], [-4, -4]],
        # layer 2: "exactly one" = OR and NAND ; "both" = AND ; "neither" = NOR
        [[1.5, 0, 1.5, 0], [0, 2, 0, 0], [0, 0, 0, 2]],
        # output: different / same
        [[3, -1.5, -1.5], [-3, 1.5, 1.5]],
    ]
    biases = [[-2, -6, 6, 2], [-1.5, 0, 0], [0, 0]]
    return {
        "id": "xor",
        "title": "XOR (hand-designed)",
        "description": "Says 'different' when exactly one of the two inputs is high. "
        "Weights were chosen by hand so each hidden unit has a nameable role; nothing was trained.",
        "activation": "tanh",
        "layer_sizes": [2, 4, 3, 2],
        "weights": weights,
        "biases": biases,
        "input_labels": ["x1", "x2"],
        "neuron_labels": [None, ["OR", "AND", "NAND", "NOR"], ["exactly one", "both", "neither"], ["different", "same"]],
        "input_range": [0.0, 1.0],
        "default_input": [1.0, 0.0],
    }


def random_preset() -> dict:
    rng = np.random.default_rng(RANDOM_SEED)
    sizes = [3, 5, 4, 3]
    weights = [_r(rng.normal(0.0, 1.0, size=(n_out, n_in))) for n_in, n_out in zip(sizes[:-1], sizes[1:])]
    biases = [_r(rng.normal(0.0, 0.3, size=n_out)) for n_out in sizes[1:]]
    return {
        "id": "random",
        "title": "Random weights (seed 7)",
        "description": "Untrained random weights: the outputs mean nothing. "
        "Use it to see how any weights turn inputs into activations.",
        "activation": "tanh",
        "layer_sizes": sizes,
        "weights": weights,
        "biases": biases,
        "input_labels": ["in 1", "in 2", "in 3"],
        "neuron_labels": [None, None, None, ["out A", "out B", "out C"]],
        "input_range": [-1.0, 1.0],
        "default_input": [0.8, -0.4, 0.5],
    }


def all_presets() -> list[dict]:
    return [xor_preset(), random_preset()]
