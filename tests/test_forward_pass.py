"""Beats 5 and 6: the forward pass and the result. Every number shown must come from the trace."""
import dataclasses

import pytest
from manim import ManimColor

from compute.export import ASSETS
from mlp_forward import answer_caption, recap_rows
from studio.equations import typst_number
from studio.network import NetworkDiagram
from studio.theme import NEG, POS
from studio.trace import ForwardTrace, probability_typst

TRACE_PATH = ASSETS / "forward_trace.json"


@pytest.fixture(scope="module")
def trace() -> ForwardTrace:
    return ForwardTrace.load(TRACE_PATH)


@pytest.fixture(scope="module")
def diagram(trace) -> NetworkDiagram:
    return NetworkDiagram(trace)


def test_signal_lines_are_colored_by_the_sign_of_what_they_carry(trace, diagram):
    # The signal on an edge is weight x incoming activation, which can differ in sign from the weight alone.
    for layer in range(1, trace.last_layer + 1):
        for index in range(trace.layer_sizes[layer]):
            lines = diagram.signal_lines(layer, index)
            carried = [trace.contributions[layer - 1][index][src] for src, w in enumerate(trace.weights[layer - 1][index]) if w != 0]
            assert len(lines) == len(carried)
            for line, value in zip(lines, carried):
                assert line.get_stroke_color().to_hex() == ManimColor(POS if value >= 0 else NEG).to_hex()


def test_a_negative_weight_can_carry_a_positive_signal(trace, diagram):
    # "different" <- hidden unit "both": weight is negative, its activation is negative, so the signal is positive.
    edge = diagram.edges[(3, 0, 1)]
    assert edge.weight < 0 and trace.contributions[2][0][1] > 0
    line = diagram.signal_lines(3, 0)[1]
    assert line.get_stroke_color().to_hex() == ManimColor(POS).to_hex()
    assert edge.get_stroke_color().to_hex() == ManimColor(NEG).to_hex()


def test_pre_activation_labels_show_the_traced_z(trace, diagram):
    for layer in range(1, trace.last_layer + 1):
        for index in range(trace.layer_sizes[layer]):
            label = diagram.pre_activation_label(layer, index)
            assert typst_number(trace.pre_activations[layer - 1][index]) in label.typst_code


def test_hidden_z_sits_above_its_node_and_output_z_where_the_probability_appears(trace, diagram):
    hidden = diagram.pre_activation_label(1, 1)
    assert hidden.get_center()[1] > diagram.nodes[1][1].ring.get_center()[1]
    out = diagram.pre_activation_label(trace.last_layer, 0)
    prob = diagram.output_labels[0].prob
    assert out.get_left()[0] == pytest.approx(prob.get_left()[0]) and out.get_center()[1] == pytest.approx(prob.get_center()[1])


def test_answer_caption_uses_the_traced_class_and_probability(trace):
    caption = answer_caption(trace)
    words, math = caption[0], caption[1]
    assert trace.prediction_label in words.original_text
    assert probability_typst(trace.prediction_probability) in math.typst_code
    assert "0.99998" in math.typst_code and "approx" in math.typst_code  # rounded, and marked as rounded


def test_recap_names_the_activation_function_from_the_trace(trace):
    assert recap_rows(trace)[1][0] == "a = tanh(z)"
    relu = dataclasses.replace(trace, activation="relu")
    assert recap_rows(relu)[1][0] == "a = relu(z)"
