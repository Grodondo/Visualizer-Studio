"""Equations for the arithmetic of one neuron, built from a ForwardTrace.

    z = w1*x1 + w2*x2 + b          symbolic: which quantity is which
    z = 4.00 * 0.80 + 4.00 * 0.30 - 2.00        the real numbers from the trace
    z = 3.20 + 1.20 - 2.00         each weight times its input
    z = 2.40                       the pre-activation

Every number is formatted by studio.trace.format_value and colored by its sign (blue +, orange -).
Each part of an equation is a named group (w1, x1, ..., b, p1, ..., z) so it can be picked out
with `eq.select("w1")` and animated on its own.
"""
from __future__ import annotations

from dataclasses import dataclass

from manim import ReplacementTransform, MathTypst, Scene, VGroup

from .theme import TEXT, sign_color
from .trace import ForwardTrace, format_value

EQUATION_FONT_SIZE = 36


@dataclass
class NeuronEquations:
    symbolic: MathTypst
    substituted: MathTypst
    products: MathTypst
    result: MathTypst
    n_inputs: int
    term_labels: list[str]  # group names in symbolic and substituted: w1, x1, w2, x2, ..., b
    product_labels: list[str]  # group names in products: p1, p2, ..., b


def typst_number(value: float) -> str:
    """Typst for a number with a leading minus kept inside its group, e.g. '- 2.00'."""
    text = format_value(value)
    return f"- {text[1:]}" if text.startswith("-") else text


def neuron_equations(trace: ForwardTrace, layer: int, index: int, font_size: float = EQUATION_FONT_SIZE) -> NeuronEquations:
    weights = trace.weights[layer - 1][index]
    incoming = trace.activations[layer - 1]
    products = trace.contributions[layer - 1][index]
    bias = trace.biases[layer - 1][index]
    z = trace.pre_activations[layer - 1][index]
    n = len(weights)
    letter = "x" if layer == 1 else "a"  # inputs are called x, later layers' outputs are called a

    def group(text, label):
        """Typst source for a named part: {{ text : label }} (MathTypst turns it into select(label))."""
        return "{{ " + text + " : " + label + " }}"

    w_names = [f"w{k + 1}" for k in range(n)]
    in_names = [f"{letter}{k + 1}" for k in range(n)]
    p_names = [f"p{k + 1}" for k in range(n)]

    # A negative bias reads "- 2.00" and a positive one "+ 2.00"; symbolic form is always "+ b".
    if bias < 0:
        bias_number = " " + group(typst_number(bias), "b")
    else:
        bias_number = " + " + group(typst_number(bias), "b")
    bias_symbol = " + " + group("b", "b")

    symbolic = "z = " + " + ".join(f"{group(f'w_{k + 1}', w_names[k])} {group(f'{letter}_{k + 1}', in_names[k])}" for k in range(n)) + bias_symbol
    substituted = "z = " + " + ".join(f"{group(typst_number(weights[k]), w_names[k])} dot {group(typst_number(incoming[k]), in_names[k])}" for k in range(n)) + bias_number
    summed = "z = " + " + ".join(group(typst_number(products[k]), p_names[k]) for k in range(n)) + bias_number
    result = "z = " + group(typst_number(z), "z")

    def build(code, values):
        eq = MathTypst(code, font_size=font_size, color=TEXT)
        for label, value in values.items():
            eq.select(label).set_color(sign_color(value))  # number color = its sign
        return eq

    term_values = {**dict(zip(w_names, weights)), **dict(zip(in_names, incoming)), "b": bias}
    return NeuronEquations(
        symbolic=build(symbolic, term_values),
        substituted=build(substituted, term_values),
        products=build(summed, {**dict(zip(p_names, products)), "b": bias}),
        result=build(result, {"z": z}),
        n_inputs=n,
        term_labels=[name for pair in zip(w_names, in_names) for name in pair] + ["b"],
        product_labels=p_names + ["b"],
    )


def rest_of(eq: MathTypst, labels: list[str]) -> VGroup:
    """The glyphs of an equation that are not in any named group (symbols like z, =, +)."""
    grouped = [m for label in labels for m in eq.select(label).submobjects]
    return VGroup(*[m for m in eq.submobjects if not any(m is g for g in grouped)])


def morph(scene: Scene, pairs: list[tuple[VGroup, VGroup]], run_time: float = 1.0) -> None:
    """Turn each source group into its target group.

    ReplacementTransform (not Transform) matters: afterwards the scene holds the target's glyphs, so the
    next step can select() from the target again. Plain Transform leaves stray glyphs when a group
    changes its number of glyphs (for example `b` becoming `- 2.00`).
    """
    scene.play(*[ReplacementTransform(src, dst) for src, dst in pairs], run_time=run_time)
