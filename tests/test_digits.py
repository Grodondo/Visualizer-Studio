"""Checks for the digit-recognition video: the exported trace is self-consistent and the drawing code shows it faithfully."""
from types import SimpleNamespace

import numpy as np
import pytest
from manim import ManimColor, VGroup

from compute.digits import DATA_DIR, FILES, TRACE_PATH, predict
from digit_network import DigitRecognition, saturating
from studio.beats import BeatScene
from studio.digit_network import PixelGrid, bar_number_typst, weight_gallery
from studio.digits_trace import INK_MAX, DigitsTrace
from studio.theme import BG, sign_color

FRAME_WIDTH = 14.22


@pytest.fixture(scope="module")
def trace() -> DigitsTrace:
    return DigitsTrace.load(TRACE_PATH)


def forward_numpy(trace: DigitsTrace, example):
    """Recompute the network from the exported weights with plain numpy, independent of compute/."""
    x = np.array(example.pixels) / 16.0
    z1 = np.array(trace.w1) @ x + np.array(trace.b1)
    a1 = np.maximum(z1, 0.0)
    z2 = np.array(trace.w2) @ a1 + np.array(trace.b2)
    e = np.exp(z2 - z2.max())
    return z1, a1, z2, e / e.sum()


# ---- the data file -------------------------------------------------------------------------------


def test_shapes(trace):
    assert trace.layer_sizes == (64, 16, 10)
    assert (len(trace.w1), len(trace.w1[0])) == (16, 64)
    assert (len(trace.w2), len(trace.w2[0])) == (10, 16)
    assert len(trace.b1) == 16 and len(trace.b2) == 10
    assert trace.dataset["license"] == "CC BY 4.0"


@pytest.mark.parametrize("role", ["showcase", "mistake"])
def test_example_is_the_network_forward_pass(trace, role):
    example = getattr(trace, role)
    z1, a1, z2, p = forward_numpy(trace, example)
    assert len(example.pixels) == 64 and all(0 <= v <= INK_MAX for v in example.pixels)
    assert np.allclose(example.z1, z1, atol=1e-9)
    assert np.allclose(example.a1, a1, atol=1e-9)
    assert np.allclose(example.z2, z2, atol=1e-9)
    assert np.allclose(example.probabilities, p, atol=1e-9)
    assert sum(example.probabilities) == pytest.approx(1.0)
    assert example.predicted == int(np.argmax(example.probabilities))


def test_showcase_is_right_and_mistake_is_wrong(trace):
    assert trace.showcase.predicted == trace.showcase.label
    assert trace.mistake.predicted != trace.mistake.label


def test_focus_neuron_adds_up(trace):
    f, ex = trace.focus, trace.showcase
    assert f.z == pytest.approx(ex.z1[f.neuron])
    assert trace.weighted_sum(f.neuron, ex) + trace.b1[f.neuron] == pytest.approx(f.z)
    assert sum(trace.products(f.neuron, ex)) == pytest.approx(trace.weighted_sum(f.neuron, ex))
    assert f.neuron == int(np.argmax(ex.a1))


def test_activation_readings_match_the_functions(trace):
    for neuron in (trace.focus, trace.quiet):
        z = neuron.z
        assert neuron.values["relu"] == pytest.approx(max(z, 0.0))
        assert neuron.values["sigmoid"] == pytest.approx(1 / (1 + np.exp(-z)))
        assert neuron.values["tanh"] == pytest.approx(np.tanh(z))
    assert trace.focus.z > 0 > trace.quiet.z
    assert trace.quiet.values["relu"] == 0


def test_curves_cover_the_readings(trace):
    x = trace.curves["x"]
    assert min(x) <= -5 and max(x) >= 5, "the graph must be wide enough to show both readings"
    assert abs(trace.focus.z) <= max(x) and abs(trace.quiet.z) <= max(x)
    assert all(len(trace.curves[k]) == len(x) for k in ("sigmoid", "tanh", "relu"))


def test_history_is_consistent(trace):
    assert [h["epoch"] for h in trace.history] == list(range(1, len(trace.history) + 1))
    assert trace.history[-1]["test_accuracy"] == pytest.approx(trace.test_accuracy)
    assert trace.test_errors == round((1 - trace.test_accuracy) * trace.dataset["n_test"])
    assert trace.test_accuracy > 0.9


@pytest.mark.skipif(not all((DATA_DIR / name).exists() for name in FILES.values()), reason="dataset not downloaded (run `uv run python -m compute.digits`)")
def test_reported_accuracy_matches_the_real_test_set(trace):
    """Re-evaluate the exported weights on the downloaded UCI test file."""
    data = np.loadtxt(DATA_DIR / FILES["test"], delimiter=",", dtype=int)
    x, y = data[:, :64] / 16.0, data[:, 64]
    params = (np.array(trace.w1), np.array(trace.b1), np.array(trace.w2), np.array(trace.b2))
    wrong = int((predict(params, x).argmax(axis=1) != y).sum())
    assert wrong == trace.test_errors
    # The showcase digit really is test item #test_index.
    assert list(data[trace.showcase.test_index, :64]) == list(trace.showcase.pixels)
    assert int(y[trace.mistake.test_index]) == trace.mistake.label


# ---- display helpers -----------------------------------------------------------------------------


def test_saturating_never_shows_a_rounded_one():
    assert saturating(0.99996) == "0.99996"
    assert saturating(0.2651) == "0.27"
    assert saturating(-0.7697) == "- 0.77"


def test_tiny_probabilities_use_scientific_notation():
    assert bar_number_typst(0.82148) == "approx 0.82"
    assert bar_number_typst(3.4e-9) == "approx 3 times 10^(-9)"
    assert bar_number_typst(0.0) == "= 0"


# ---- drawing -------------------------------------------------------------------------------------


def test_ink_grid_colors_follow_the_pixels(trace):
    grid = PixelGrid(trace.showcase.pixels, vmax=INK_MAX)
    assert len(grid.cells) == 64
    for k, v in enumerate(trace.showcase.pixels):
        if v == 0:
            assert grid.cells[k].get_fill_color() == ManimColor(BG)
        if v == INK_MAX:
            assert grid.cells[k].get_fill_color() == ManimColor("#ffffff")
    # Squares are laid out row by row from the top left.
    assert grid.cells[0].get_center()[1] > grid.cells[63].get_center()[1]
    assert grid.cells[0].get_center()[0] < grid.cells[7].get_center()[0]


def test_weight_map_colors_follow_the_signs(trace):
    weights = trace.weight_image(trace.focus.neuron)
    grid = PixelGrid(weights, mode="signed")
    for k, w in enumerate(weights):
        assert grid.cells[k].get_fill_color() == sign_color(w)
    strongest = max(range(64), key=lambda k: abs(weights[k]))
    assert grid.cells[strongest].get_fill_opacity() == pytest.approx(1.0)


def test_gallery_shows_every_hidden_neuron_on_one_scale(trace):
    maps = weight_gallery(trace)
    assert len(maps) == 16
    assert len({m.vmax for m in maps}) == 1
    assert max(abs(w) for row in trace.w1 for w in row) == pytest.approx(maps[0].vmax)


def test_gallery_fits_above_the_caption(trace):
    maps = VGroup(*weight_gallery(trace)).move_to([-2.6, 0.6, 0])  # the position used in the scene
    assert maps.get_top()[1] < 3.9
    assert maps.get_bottom()[1] > BeatScene.caption_y + 0.45


def test_footer_fits_on_screen(trace):
    footer = DigitRecognition.footer(SimpleNamespace(trace=trace))
    assert footer.width < FRAME_WIDTH - 0.6
    assert footer.get_bottom()[1] > -4.0
