"""The mathematics of a saddle point: a quadratic surface z = f(x, y) around a critical point.

    f(x, y) = 1/2 * (x, y) H (x, y)^T          H = [[h11, h12], [h12, h22]] is the Hessian (second derivatives)

At the origin the gradient is (0, 0) whatever H is, so the surface is flat there. What the ground does next is decided
by the eigenvalues of H: all positive = minimum, all negative = maximum, mixed signs = saddle point.

Run from the project root:  uv run python -m compute.saddle   (writes assets/saddle/saddle_trace.json)
The Manim video reads only that file.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TRACE_PATH = ROOT / "assets" / "saddle" / "saddle_trace.json"

# f(x, y) = x^2 - y^2. Change this matrix to explore other shapes ([[2, 0], [0, 2]] is a bowl).
HESSIAN = [[2.0, 0.0], [0.0, -2.0]]
EXTENT = 2.0  # the surface is drawn for x and y in [-EXTENT, EXTENT]
SLICE_SAMPLES = 41


def height(x, y, hessian=HESSIAN):
    (h11, h12), (_, h22) = hessian
    return 0.5 * (h11 * x * x + 2 * h12 * x * y + h22 * y * y)


def gradient(x, y, hessian=HESSIAN) -> list[float]:
    (h11, h12), (_, h22) = hessian
    return [h11 * x + h12 * y, h12 * x + h22 * y]


def curvature_along(direction, hessian=HESSIAN) -> float:
    """Second derivative of f along a straight line through the origin: d^T H d / d^T d."""
    d = np.asarray(direction, dtype=float)
    return float(d @ np.asarray(hessian, dtype=float) @ d / (d @ d))


def classify(eigenvalues) -> str:
    if any(abs(v) < 1e-12 for v in eigenvalues):
        return "degenerate"  # the second-derivative test cannot decide
    if all(v > 0 for v in eigenvalues):
        return "minimum"
    if all(v < 0 for v in eigenvalues):
        return "maximum"
    return "saddle"


def function_typst(hessian=HESSIAN) -> str:
    """The formula as Typst math, e.g. 'x^2 - y^2'."""
    (h11, h12), (_, h22) = hessian
    parts = []
    for coefficient, term in ((h11 / 2, "x^2"), (h12, "x y"), (h22 / 2, "y^2")):
        if coefficient == 0:
            continue
        magnitude = "" if abs(coefficient) == 1 else f"{abs(coefficient):g} "
        sign = "-" if coefficient < 0 else "+"
        parts.append((sign, f"{magnitude}{term}"))
    first_sign, first = parts[0]
    text = (("-" if first_sign == "-" else "") + first)
    for sign, body in parts[1:]:
        text += f" {sign} {body}"
    return text


def slice_curve(axis: str, hessian=HESSIAN, extent=EXTENT, n=SLICE_SAMPLES) -> dict:
    """Heights along the x axis (y = 0) or the y axis (x = 0)."""
    t = np.linspace(-extent, extent, n)
    z = [height(v, 0.0, hessian) if axis == "x" else height(0.0, v, hessian) for v in t]
    return {"t": t.tolist(), "z": [float(v) for v in z]}


def build_trace(hessian=HESSIAN) -> dict:
    eigenvalues = np.linalg.eigvalsh(np.asarray(hessian, dtype=float))
    return {
        "schema": 1,
        "title": "Why is a saddle point not a minimum?",
        "provenance": {"surface": "computed", "note": "exact quadratic surface; schematic choice of f"},
        "function_typst": function_typst(hessian),
        "hessian": [list(row) for row in hessian],
        "extent": EXTENT,
        "critical_point": [0.0, 0.0],
        "gradient_at_critical_point": gradient(0.0, 0.0, hessian),
        "height_at_critical_point": height(0.0, 0.0, hessian),
        "curvature": {"x": curvature_along([1, 0], hessian), "y": curvature_along([0, 1], hessian)},
        "eigenvalues": [float(v) for v in eigenvalues],
        "classification": classify(eigenvalues),
        "slices": {"x": slice_curve("x", hessian), "y": slice_curve("y", hessian)},
    }


def write_trace() -> None:
    TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRACE_PATH.write_text(json.dumps(build_trace(), indent=1), encoding="utf-8")
    print(f"wrote {TRACE_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    write_trace()
