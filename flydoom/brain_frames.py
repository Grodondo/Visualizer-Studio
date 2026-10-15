"""Brain images for the flydoom cut (measured FlyWire anatomy only; nothing here is activity).

Reuses compute.fly_render (the connectome video's renderer) without changing it, and writes to media/flydoom_brain/:
    back_all.png       every neuron, seen from behind (so the fly's left is on the viewer's left, like the game)
    back_optic.png     only the optic-lobe neurons, for highlighting
    back_central.png   only the central-brain neurons, for highlighting
    circuit/NNN.png    the giant-fiber circuit's measured synapses (right side), turned about its middle; colored by
                       ROLE for this cut: right-eye inputs in amber shades (LC4 pale, LPLC2 deep), the giant fiber (output) white
The whole-brain turntable comes from media/fly_frames/ (uv run python -m compute.fly_render).

    uv run python -m flydoom.brain_frames
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from compute import fly_render as fr

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "media" / "flydoom_brain"
ANGLE = 180  # seen from behind
PX = (2400, 1333)  # sharper than the 1800 px turntable, because this view is zoomed into
LAYERS = {"optic": ["optic"], "central": ["central"]}
CIRCUIT_COLORS = {"LC4": "#ffd08a", "LPLC2": "#ff9a3d", "GF": "#e8edf3"}
CIRCUIT_STEP = 6  # degrees between close-up frames


def main() -> None:
    xyz, cls, names = fr.load()
    c = fr.center_of(xyz)
    fr.render(xyz, cls, names, c, ANGLE, OUT / "back_all.png", px=PX)
    for layer, classes in LAYERS.items():
        keep = np.isin(np.array(names)[cls], classes)
        fr.render(xyz[keep], cls[keep], names, c, ANGLE, OUT / f"back_{layer}.png", alpha=0.9, px=PX)
    # the circuit close-up (same pivot and window size as the connectome video's close-up)
    layers = fr.circuit_layers()
    hw = fr.zoom_window(c, layers)[2]
    pivot = layers["GF"].mean(axis=0)
    angles = list(range(0, 360, CIRCUIT_STEP))
    for a in angles:
        stack = [(layers[n], CIRCUIT_COLORS[n], fr.SYNAPSE_SIZE[n]) for n in ("LPLC2", "LC4", "GF")]
        fr.render(xyz, cls, names, pivot, a, OUT / "circuit" / f"{a:03d}.png", alpha=fr.ZOOM_DIM_ALPHA, window=(0.0, 0.0, hw), layers=stack)
    (OUT / "manifest.json").write_text(json.dumps({"angle": ANGLE, "px": PX, "center_um": [float(v) for v in c],
                                                   "half_width_um": fr.HALF_WIDTH_UM, "layers": LAYERS,
                                                   "circuit": {"pivot_um": [float(v) for v in pivot], "half_width_um": hw, "angles": angles,
                                                               "colors": CIRCUIT_COLORS,
                                                               "centroids_um": {k: [float(x) for x in v.mean(axis=0)] for k, v in layers.items()}}},
                                                  indent=1), encoding="utf-8")
    print(f"wrote {1 + len(LAYERS) + len(angles)} images to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
