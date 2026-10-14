"""Write the MLP presets, a golden trace, and the forward trace used by the video.

Run from the project root:  uv run python -m compute.export

Outputs
  assets/mlp/mlp.json            presets (weights, biases, labels)
  assets/mlp/golden.json         forward-pass traces used to test the TypeScript port
  assets/mlp/forward_trace.json  one fully documented forward pass; the Manim video reads only this
  web/public/data/mlp.json       copy of mlp.json for the browser
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from compute.mlp import ACTIVATIONS, forward
from compute.mlp_presets import all_presets

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets" / "mlp"
WEB_DATA = ROOT / "web" / "public" / "data"

# Both sides use IEEE doubles. They can differ only by libm rounding in tanh/exp (~1e-15 per op);
# 1e-9 leaves room for that yet still fails on any real logic error.
TOLERANCE_ABS = 1e-9

# Off-grid points as well as corners, so the check is not satisfied by a lucky symmetric input.
EXTRA_INPUTS_UNIT = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.3, 0.8]]
EXTRA_INPUTS_SIGNED = [[0.8, -0.4, 0.5], [-1.0, 1.0, -0.2], [0.0, 0.0, 0.0]]

# The forward pass shown in the first video. x1 and x2 differ, so every weighted sum has distinct terms.
VIDEO_PRESET_ID = "xor"
VIDEO_INPUT = [0.8, 0.3]
VIDEO_ACTIVATION = "tanh"


def build_mlp_json() -> dict:
    return {"schema": 1, "presets": all_presets()}


def build_forward_trace(mlp: dict) -> dict:
    """Raw double-precision numbers from compute.mlp.forward; rounding is a presentation choice made later."""
    preset = next(p for p in mlp["presets"] if p["id"] == VIDEO_PRESET_ID)
    trace = forward(preset["weights"], preset["biases"], VIDEO_INPUT, VIDEO_ACTIVATION)
    output = trace["activations"][-1]
    best = output.index(max(output))
    return {
        "schema": 1,
        "preset": preset["id"],
        "title": preset["title"],
        "description": preset["description"],
        "provenance": {"network": "schematic", "numbers": "computed"},
        "activation": VIDEO_ACTIVATION,
        "input": VIDEO_INPUT,
        "layer_sizes": preset["layer_sizes"],
        "input_labels": preset["input_labels"],
        "neuron_labels": preset["neuron_labels"],
        "weights": preset["weights"],
        "biases": preset["biases"],
        **trace,
        "prediction": {"index": best, "label": preset["neuron_labels"][-1][best], "probability": output[best]},
    }


def build_golden(mlp: dict) -> dict:
    cases = []
    for preset in mlp["presets"]:
        inputs = EXTRA_INPUTS_UNIT if preset["input_range"][0] == 0.0 else EXTRA_INPUTS_SIGNED
        for activation in ACTIVATIONS:
            for x in inputs:
                trace = forward(preset["weights"], preset["biases"], x, activation)
                cases.append({"preset": preset["id"], "activation": activation, "input": x, **trace})
    return {"schema": 1, "tolerance_abs": TOLERANCE_ABS, "cases": cases}


def write_all() -> None:
    mlp = build_mlp_json()
    golden = build_golden(mlp)
    ASSETS.mkdir(parents=True, exist_ok=True)
    WEB_DATA.mkdir(parents=True, exist_ok=True)
    (ASSETS / "mlp.json").write_text(json.dumps(mlp, indent=1), encoding="utf-8")
    (ASSETS / "golden.json").write_text(json.dumps(golden), encoding="utf-8")
    (ASSETS / "forward_trace.json").write_text(json.dumps(build_forward_trace(mlp), indent=1), encoding="utf-8")
    shutil.copyfile(ASSETS / "mlp.json", WEB_DATA / "mlp.json")
    print(f"wrote {len(mlp['presets'])} presets, {len(golden['cases'])} golden cases and the forward trace")


if __name__ == "__main__":
    write_all()
