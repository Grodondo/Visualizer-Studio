"""The only bridge between the numerical model and the scenes.

A scene never calls the model. It loads assets/mlp/forward_trace.json (written by
`uv run python -m compute.export` from compute.mlp.forward) and shows what is in it.
This module also decides how those numbers are *rounded for display*.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DECIMALS = 2  # every ordinary value on screen is shown to this many decimals ("values rounded" footer)
MIN_PROB_DECIMALS = 2
MAX_PROB_DECIMALS = 8


@dataclass(frozen=True)
class ForwardTrace:
    preset: str
    title: str
    activation: str
    input: tuple[float, ...]
    layer_sizes: tuple[int, ...]
    input_labels: tuple[str, ...]
    neuron_labels: tuple[tuple[str, ...] | None, ...]
    weights: tuple  # weights[l-1][to][from]
    biases: tuple  # biases[l-1][to]
    pre_activations: tuple  # z of layer l is pre_activations[l-1]
    activations: tuple  # a of layer l is activations[l]; activations[0] is the input
    contributions: tuple  # contributions[l-1][to][from] = weight * incoming activation
    prediction_index: int
    prediction_label: str
    prediction_probability: float

    @classmethod
    def load(cls, path: str | Path) -> "ForwardTrace":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            preset=d["preset"],
            title=d["title"],
            activation=d["activation"],
            input=tuple(float(v) for v in d["input"]),
            layer_sizes=tuple(d["layer_sizes"]),
            input_labels=tuple(d["input_labels"]),
            neuron_labels=tuple(tuple(x) if x else None for x in d["neuron_labels"]),
            weights=_floats(d["weights"]),
            biases=_floats(d["biases"]),
            pre_activations=_floats(d["pre_activations"]),
            activations=_floats(d["activations"]),
            contributions=_floats(d["contributions"]),
            prediction_index=d["prediction"]["index"],
            prediction_label=d["prediction"]["label"],
            prediction_probability=float(d["prediction"]["probability"]),
        )

    @property
    def last_layer(self) -> int:
        return len(self.layer_sizes) - 1

    def neuron_label(self, layer: int, index: int) -> str | None:
        labels = self.input_labels if layer == 0 else self.neuron_labels[layer]
        return labels[index] if labels else None


def _floats(x):
    return tuple(_floats(v) for v in x) if isinstance(x, list) else float(x)


def clean_zero(value: float, decimals: int = DEFAULT_DECIMALS) -> float:
    """Values that round to zero are shown as 0, never as -0."""
    return 0.0 if round(value, decimals) == 0 else value


def format_value(value: float, decimals: int = DEFAULT_DECIMALS) -> str:
    return f"{clean_zero(value, decimals):.{decimals}f}"


def probability_decimals(p: float) -> int:
    """Decimals needed so that the first significant digit of the distance from 0 or 1 is visible.

    0.43 -> 2 (never fewer), 0.99998 -> 5, 0.0000168 -> 5. Rounding 0.9999832 to two decimals would
    print 1.00 and hide that the model is not literally certain.
    """
    distance = min(p, 1.0 - p)
    if distance <= 0:
        return MAX_PROB_DECIMALS
    return max(MIN_PROB_DECIMALS, min(MAX_PROB_DECIMALS, -math.floor(math.log10(distance))))


def format_probability(p: float) -> str:
    return f"{p:.{probability_decimals(p)}f}"


def probability_typst(p: float) -> str:
    """Typst math for a probability, always marked as rounded: 'p approx 0.99998'.

    Beyond the display limit it says so ('p > 0.99999999') instead of printing an exact-looking 1.00000000.
    """
    d = probability_decimals(p)
    shown = f"{p:.{d}f}"
    if p in (0.0, 1.0):
        return f"p = {int(p)}"
    if float(shown) in (0.0, 1.0):
        return f"p > {1 - 10 ** -d:.{d}f}" if p > 0.5 else f"p < {10 ** -d:.{d}f}"
    return f"p approx {shown}"


def typst_label(name: str) -> str:
    """'x1' -> 'x_1' so Typst draws a subscript."""
    return re.sub(r"^([A-Za-z]+)(\d+)$", r"\1_\2", name)
