"""Display rules and the diagram: what appears on screen must be exactly what the trace says."""
import pytest
from manim import ManimColor, config

from compute.export import ASSETS
from mlp_forward import caption, encoding_key, footer
from studio.network import EDGE_WIDTH_MAX, EDGE_WIDTH_MIN, TINT_MAX, NetworkDiagram
from studio.theme import NEG, POS
from studio.trace import DEFAULT_DECIMALS, ForwardTrace, format_value, probability_decimals, probability_typst, typst_label

TRACE_PATH = ASSETS / "forward_trace.json"


# --- formatting -----------------------------------------------------------------------------

@pytest.mark.parametrize(
    "p, expected",
    [
        (0.9999831959684379, "p approx 0.99998"),  # the video's actual output probability
        (1.680403156203298e-05, "p approx 0.00002"),
        (0.43, "p approx 0.43"),
        (0.5, "p approx 0.50"),
        (1 - 1e-12, "p > 0.99999999"),  # beyond the display limit: never an exact-looking 1.00000000
        (1e-12, "p < 0.00000001"),
        (1.0, "p = 1"),
        (0.0, "p = 0"),
    ],
)
def test_probability_is_always_marked_as_rounded(p, expected):
    assert probability_typst(p) == expected


def test_probability_never_prints_as_a_plain_one():
    assert probability_decimals(0.9999831959684379) == 5
    assert "1.00" not in probability_typst(0.9999831959684379)


@pytest.mark.parametrize("value, expected", [(-0.9216685544064712, "-0.92"), (0.8, "0.80"), (-0.001, "0.00"), (2.4000000000000004, "2.40")])
def test_ordinary_values_use_fixed_decimals_without_negative_zero(value, expected):
    assert format_value(value) == expected


def test_typst_subscript_label():
    assert typst_label("x1") == "x_1"
    assert typst_label("OR") == "OR"


# --- on-screen explanations ---------------------------------------------------------------------

def texts(group) -> list[str]:
    # Text.text drops spaces; original_text is the string the viewer reads.
    return [m.original_text for m in group.get_family() if hasattr(m, "original_text")]


def test_footer_states_the_mixed_rounding_rule():
    lines = texts(footer())
    assert any("probabilities show additional precision" in t for t in lines)
    assert not any("2 decimals" in t for t in lines)  # the old, inaccurate wording


def test_key_explains_omitted_zero_weight_edges(trace, diagram):
    assert "no line = weight 0" in texts(encoding_key())
    # ...and the diagram really omits exactly the zero weights, nothing else.
    zero = sum(1 for l in range(1, trace.last_layer + 1) for row in trace.weights[l - 1] for w in row if w == 0)
    total = sum(len(row) for l in range(1, trace.last_layer + 1) for row in trace.weights[l - 1])
    assert zero > 0 and len(diagram.edges) == total - zero


def test_text_rows_stay_inside_the_frame_and_clear_the_network(trace, diagram):
    frame_bottom = -config.frame_height / 2
    rows = [caption("Layers of simple units, joined by weighted connections"), encoding_key(), footer()]
    for row in rows:
        assert row.get_bottom()[1] >= frame_bottom + 0.1  # a clipped footer line went unnoticed once
        assert row.get_left()[0] >= -config.frame_width / 2 and row.get_right()[0] <= config.frame_width / 2
    lowest_node = min(n.ring.get_bottom()[1] for layer in diagram.nodes for n in layer)
    assert rows[0].get_top()[1] < lowest_node


# --- the diagram is a faithful picture of the trace -------------------------------------------

@pytest.fixture(scope="module")
def trace() -> ForwardTrace:
    return ForwardTrace.load(TRACE_PATH)


@pytest.fixture(scope="module")
def diagram(trace) -> NetworkDiagram:
    return NetworkDiagram(trace).show_all_activations()


def test_trace_loader_matches_the_file(trace):
    assert trace.input == (0.8, 0.3)
    assert trace.layer_sizes == (2, 4, 3, 2)
    assert trace.prediction_label == "different"


def test_node_numbers_and_colors_come_from_the_trace(trace, diagram):
    for layer in range(trace.last_layer):  # the output layer shows probabilities beside it, not a number inside
        for i, node in enumerate(diagram.nodes[layer]):
            a = trace.activations[layer][i]
            assert node.value.get_value() == a  # the exact trace value; rounding happens only in display
            assert node.value.num_decimal_places == DEFAULT_DECIMALS
            expected = ManimColor(POS if a >= 0 else NEG).to_hex()
            assert node.ring.get_stroke_color().to_hex() == expected
            assert node.value.get_color().to_hex() == expected
            assert node.ring.get_fill_opacity() == pytest.approx(TINT_MAX * min(1.0, abs(a)))


def test_output_nodes_show_the_traced_probabilities(trace, diagram):
    for i, label in enumerate(diagram.output_labels):
        p = trace.activations[trace.last_layer][i]
        assert label.text.text == trace.neuron_labels[trace.last_layer][i]
        assert probability_typst(p) in label.prob.typst_code
        assert label.prob.get_fill_opacity() == 1  # revealed by show_all_activations
    best = diagram.output_labels[trace.prediction_index]
    assert best.text.text == trace.prediction_label
    assert probability_typst(trace.prediction_probability) in best.prob.typst_code


def test_one_edge_per_nonzero_weight_with_sign_color_and_magnitude_width(trace, diagram):
    expected_keys = {
        (l, to, src)
        for l in range(1, trace.last_layer + 1)
        for to, row in enumerate(trace.weights[l - 1])
        for src, w in enumerate(row)
        if w != 0
    }
    assert set(diagram.edges) == expected_keys
    for (l, to, src), edge in diagram.edges.items():
        w = trace.weights[l - 1][to][src]
        assert edge.weight == w
        assert edge.get_stroke_color().to_hex() == ManimColor(POS if w > 0 else NEG).to_hex()
        assert EDGE_WIDTH_MIN <= edge.get_stroke_width() <= EDGE_WIDTH_MAX
    widths = sorted(diagram.edges.values(), key=lambda e: abs(e.weight))
    assert [e.get_stroke_width() for e in widths] == sorted(e.get_stroke_width() for e in widths)
