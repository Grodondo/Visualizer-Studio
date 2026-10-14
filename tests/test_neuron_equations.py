"""The neuron explanation: every number in the equations and labels must come from the trace."""
import pytest
from manim import ManimColor

from compute.export import ASSETS
from studio.equations import neuron_equations, typst_number
from studio.network import NetworkDiagram
from studio.theme import NEG, POS
from studio.trace import ForwardTrace, format_value

TRACE_PATH = ASSETS / "forward_trace.json"


@pytest.fixture(scope="module")
def trace() -> ForwardTrace:
    return ForwardTrace.load(TRACE_PATH)


def test_typst_number_keeps_the_minus_inside_the_group():
    assert typst_number(-2.0) == "- 2.00"
    assert typst_number(2.4000000000000004) == "2.40"
    assert typst_number(-0.001) == "0.00"  # never "-0.00"


def test_or_neuron_equations_use_the_traced_numbers(trace):
    eq = neuron_equations(trace, 1, 0)
    # 4.00 * 0.80 + 4.00 * 0.30 - 2.00 = 2.40, all read from the trace
    assert eq.term_labels == ["w1", "x1", "w2", "x2", "b"]
    for expected in ("4.00", "0.80", "0.30", "- 2.00"):
        assert expected in eq.substituted.typst_code
    assert "3.20" in eq.products.typst_code and "1.20" in eq.products.typst_code and "- 2.00" in eq.products.typst_code
    assert "2.40" in eq.result.typst_code
    assert eq.symbolic.typst_code.count("manimgrp(") == len(eq.term_labels)  # one named group per term


def test_displayed_parts_add_up_to_the_displayed_result(trace):
    # Rounded numbers on screen should still agree with each other for this neuron.
    c = trace.contributions[0][0]
    shown = float(format_value(c[0])) + float(format_value(c[1])) + float(format_value(trace.biases[0][0]))
    assert shown == pytest.approx(float(format_value(trace.pre_activations[0][0])))


def test_term_colors_follow_the_sign_of_each_number(trace):
    eq = neuron_equations(trace, 1, 0)
    values = {"w1": 4.0, "x1": 0.8, "w2": 4.0, "x2": 0.3, "b": -2.0}
    for label, value in values.items():
        want = ManimColor(POS if value >= 0 else NEG).to_hex()
        for glyph in eq.symbolic.select(label):
            assert glyph.get_fill_color().to_hex() == want, label


def test_equations_work_for_any_neuron_and_input_count(trace):
    # Hidden layer 2 neurons have 4 inputs named a1..a4 instead of x1, x2.
    eq = neuron_equations(trace, 2, 0)
    assert eq.n_inputs == 4
    assert eq.term_labels[:2] == ["w1", "a1"] and eq.term_labels[-1] == "b"
    assert eq.product_labels == ["p1", "p2", "p3", "p4", "b"]


def test_weight_labels_show_the_edge_weights(trace):
    diagram = NetworkDiagram(trace)
    labels = diagram.weight_labels(1, 0)
    edges = diagram.edges_into(1, 0)
    assert [l.get_value() for l in labels] == [e.weight for e in edges] == [4.0, 4.0]
    for label in labels:
        assert label.get_color().to_hex() == ManimColor(POS).to_hex()


def test_node_reveal_uses_the_traced_activation(trace):
    diagram = NetworkDiagram(trace)
    node = diagram.nodes[1][0]
    value = node.prepare_value()
    assert value.get_value() == trace.activations[1][0]
    assert value.get_fill_opacity() == 1
    assert value.get_color().to_hex() == ManimColor(POS).to_hex()
