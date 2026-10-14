"""The saddle-point video: the mathematics, the exported trace, and what the scene shows of it."""
import json
import random

import numpy as np
import pytest
from manim import ManimColor, tempconfig

from compute.saddle import (
    HESSIAN,
    TRACE_PATH,
    build_trace,
    classify,
    curvature_along,
    function_typst,
    gradient,
    height,
)
from saddle_point import SaddleTrace, curvature_label
from studio.beats import BeatScene
from studio.theme import NEG, POS


# --- the mathematics --------------------------------------------------------------------------

def test_committed_trace_is_up_to_date():
    # Fails if compute/saddle.py changes without re-running `uv run python -m compute.saddle`.
    assert json.loads(TRACE_PATH.read_text(encoding="utf-8")) == json.loads(json.dumps(build_trace()))


def test_the_ground_is_flat_at_the_center():
    assert gradient(0.0, 0.0) == [0.0, 0.0]
    assert height(0.0, 0.0) == 0.0


def test_gradient_matches_finite_differences():
    rng = random.Random(1)  # fixed seed: the check is deterministic
    h = 1e-5
    for _ in range(20):
        x, y = rng.uniform(-2, 2), rng.uniform(-2, 2)
        numeric = [(height(x + h, y) - height(x - h, y)) / (2 * h), (height(x, y + h) - height(x, y - h)) / (2 * h)]
        assert gradient(x, y) == pytest.approx(numeric, abs=1e-6)


def test_curvature_matches_second_differences_along_each_axis():
    h = 1e-2  # exact for a quadratic, apart from rounding
    second_x = (height(h, 0) - 2 * height(0, 0) + height(-h, 0)) / h**2
    second_y = (height(0, h) - 2 * height(0, 0) + height(0, -h)) / h**2
    assert curvature_along([1, 0]) == pytest.approx(second_x, abs=1e-6) == pytest.approx(2.0, abs=1e-6)
    assert curvature_along([0, 1]) == pytest.approx(second_y, abs=1e-6) == pytest.approx(-2.0, abs=1e-6)


@pytest.mark.parametrize(
    "hessian, expected",
    [
        ([[2, 0], [0, 2]], "minimum"),
        ([[-2, 0], [0, -2]], "maximum"),
        ([[2, 0], [0, -2]], "saddle"),
        ([[0, 1], [1, 0]], "saddle"),  # a saddle whose directions are rotated by 45 degrees
        ([[4, 0], [0, 0]], "degenerate"),  # the second-derivative test cannot decide
    ],
)
def test_classification_by_eigenvalues(hessian, expected):
    assert classify(np.linalg.eigvalsh(np.array(hessian, dtype=float))) == expected


def test_the_video_surface_is_a_saddle_that_rises_along_x_and_falls_along_y():
    t = build_trace()
    assert t["classification"] == "saddle"
    xs, zx = t["slices"]["x"]["t"], t["slices"]["x"]["z"]
    ys, zy = t["slices"]["y"]["t"], t["slices"]["y"]["z"]
    assert all(z > 0 for x, z in zip(xs, zx) if x != 0)  # up in every x direction
    assert all(z < 0 for y, z in zip(ys, zy) if y != 0)  # down in every y direction
    assert t["curvature"] == {"x": 2.0, "y": -2.0}


@pytest.mark.parametrize(
    "hessian, expected",
    [([[2, 0], [0, -2]], "x^2 - y^2"), ([[2, 0], [0, 2]], "x^2 + y^2"), ([[-2, 0], [0, -2]], "-x^2 - y^2"), ([[2, 0], [0, -4]], "x^2 - 2 y^2"), ([[2, 1], [1, -2]], "x^2 + x y - y^2")],
)
def test_formula_text(hessian, expected):
    assert function_typst(hessian) == expected


# --- what the scene reads and shows -------------------------------------------------------------

@pytest.fixture(scope="module")
def trace() -> SaddleTrace:
    return SaddleTrace.load(TRACE_PATH)


def test_scene_surface_equals_the_model(trace):
    # The scene rebuilds the surface from the exported Hessian; it must agree with compute.saddle everywhere.
    for x in np.linspace(-2, 2, 9):
        for y in np.linspace(-2, 2, 9):
            assert trace.height(x, y) == pytest.approx(height(x, y, HESSIAN))


def test_exported_slices_lie_on_the_surface(trace):
    for t, z in zip(*trace.x_slice):
        assert z == pytest.approx(trace.height(t, 0.0))
    for t, z in zip(*trace.y_slice):
        assert z == pytest.approx(trace.height(0.0, t))
    assert trace.z_limit == pytest.approx(4.0)


def test_curvature_labels_show_the_traced_numbers_in_their_sign_color(trace):
    up, down = curvature_label("x", trace.curvature_x), curvature_label("y", trace.curvature_y)
    assert "2.00" in up.typst_code and "- 2.00" in down.typst_code
    for label, colour in ((up, POS), (down, NEG)):
        for glyph in label.select("v"):
            assert glyph.get_fill_color().to_hex() == ManimColor(colour).to_hex()


# --- the shared beat machinery ----------------------------------------------------------------

def test_beat_scene_can_render_a_still_image(tmp_path):
    # A still render (-s) has no video file; an early version of BeatScene crashed there.
    class OneBeat(BeatScene):
        def construct(self):
            self.beat("only")

    with tempconfig({"format": "png", "quality": "low_quality", "media_dir": str(tmp_path), "verbosity": "ERROR", "disable_caching": True}):
        OneBeat().render()
    assert list(tmp_path.rglob("*.png"))
