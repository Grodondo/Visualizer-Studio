"""Loader for assets/digits/digits_trace.json (written by `uv run python -m compute.digits`).

The digit video reads this file and nothing else: the weights, the example digits, every number of their forward
passes, and the training history. Display rounding reuses studio.trace (format_value, probability_typst).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

INK_MAX = 16  # pixel values in the dataset run from 0 (no ink) to 16 (full ink)


@dataclass(frozen=True)
class DigitExample:
    role: str
    test_index: int
    label: int
    pixels: tuple  # 64 raw integers 0..16, row by row
    z1: tuple  # hidden pre-activations (16)
    a1: tuple  # hidden activations (16)
    z2: tuple  # output scores (10)
    probabilities: tuple  # softmax of z2 (10)
    predicted: int

    @property
    def ink(self) -> tuple:
        """Pixels scaled to 0..1, the numbers the network actually receives."""
        return tuple(v / INK_MAX for v in self.pixels)


@dataclass(frozen=True)
class NeuronFocus:
    neuron: int
    z: float
    values: dict  # what sigmoid, tanh and relu would each output for this z


@dataclass(frozen=True)
class DigitsTrace:
    dataset: dict
    layer_sizes: tuple
    activation: str
    w1: tuple  # 16 x 64
    b1: tuple
    w2: tuple  # 10 x 16
    b2: tuple
    focus: NeuronFocus
    quiet: NeuronFocus
    showcase: DigitExample
    mistake: DigitExample
    history: tuple  # one dict per epoch: epoch, train_loss, test_accuracy
    test_accuracy: float
    test_errors: int
    curves: dict  # x, sigmoid, tanh, relu

    @classmethod
    def load(cls, path) -> "DigitsTrace":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        examples = {e["role"]: _example(e) for e in d["examples"]}
        return cls(
            dataset=d["dataset"],
            layer_sizes=tuple(d["layer_sizes"]),
            activation=d["activation"],
            w1=_tuple(d["weights"][0]),
            b1=_tuple(d["biases"][0]),
            w2=_tuple(d["weights"][1]),
            b2=_tuple(d["biases"][1]),
            focus=NeuronFocus(d["focus"]["neuron"], d["focus"]["z"], d["focus"]["values"]),
            quiet=NeuronFocus(d["quiet"]["neuron"], d["quiet"]["z"], d["quiet"]["values"]),
            showcase=examples["showcase"],
            mistake=examples["mistake"],
            history=tuple(d["training"]["history"]),
            test_accuracy=d["training"]["test_accuracy"],
            test_errors=d["training"]["test_errors"],
            curves=d["activation_curves"],
        )

    def weight_image(self, neuron: int) -> tuple:
        """The 64 weights of one hidden neuron, in the same pixel order as the digit."""
        return self.w1[neuron]

    def products(self, neuron: int, example: DigitExample) -> tuple:
        """weight x pixel for every pixel: what each pixel adds to the neuron's sum."""
        return tuple(w * x for w, x in zip(self.w1[neuron], example.ink))

    def weighted_sum(self, neuron: int, example: DigitExample) -> float:
        return sum(self.products(neuron, example))

    def signal(self, example: DigitExample, output: int, hidden: int) -> float:
        """What hidden neuron `hidden` sends to output `output`: weight x activation."""
        return self.w2[output][hidden] * example.a1[hidden]


def _tuple(x):
    return tuple(_tuple(v) for v in x) if isinstance(x, list) else float(x)


def _example(e: dict) -> DigitExample:
    return DigitExample(
        role=e["role"],
        test_index=e["test_index"],
        label=e["label"],
        pixels=tuple(int(v) for v in e["pixels"]),
        z1=_tuple(e["z1"]),
        a1=_tuple(e["a1"]),
        z2=_tuple(e["z2"]),
        probabilities=_tuple(e["probabilities"]),
        predicted=e["predicted"],
    )
