"""Draw all 139,255 FlyWire neurons as transparent PNG images (a turntable), for "Inside a real brain's wiring".

Why images: Manim's Cairo renderer slows to a crawl with ~10^5 vector dots, while matplotlib draws them as one scatter
in about a second. The few neurons the story is about are drawn by Manim as vector dots on top, using the camera
parameters written to manifest.json, so dots and image line up exactly.

    uv run python -m compute.fly_render --check     # three stills into media/check/
    uv run python -m compute.fly_render             # the turntable and the still views into media/fly_frames/

Camera: orthographic. Angle 0 looks at the fly's FACE (front view): the fly's right side is on the viewer's left,
dorsal is up. Positive angles turn the brain about its vertical axis.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
POINTS_PATH = ROOT / "assets" / "connectome" / "brain_points.npz"
TRACE_PATH = ROOT / "assets" / "connectome" / "connectome_trace.json"
SYNAPSES_PATH = ROOT / "assets" / "connectome" / "circuit_synapses.npz"
OUT = ROOT / "media" / "fly_frames"
CHECK_DIR = ROOT / "media" / "check"

# ---- look ----------------------------------------------------------------------------------------
IMAGE_W, IMAGE_H = 1800, 1000  # pixels; the frame spans HALF_WIDTH_UM across
HALF_WIDTH_UM = 450.0  # half of the field of view, micrometres (the brain is about 820 um wide)
POINT_SIZE = 1.0  # matplotlib marker area (points^2) in the whole-brain view
ZOOM_POINT_SIZE = 2.5  # the same neurons seen up close (zoom window and circuit turntable)
POINT_ALPHA = 0.55
# one color per neuron class (FlyWire "super_class"); also used for the video's labels (copied into the manifest)
CLASS_COLORS = {
    "optic": "#4f7cac",  # optic lobes (vision)
    "central": "#c9d1d9",  # central brain
    "sensory": "#8fd694",  # sensory neurons entering the brain
    "visual_projection": "#b69cff",  # optic lobe -> central brain
    "ascending": "#7d8590",
    "descending": "#7d8590",
    "visual_centrifugal": "#7d8590",
    "motor": "#7d8590",
    "endocrine": "#7d8590",
}
CLASS_NAMES = {"optic": "optic lobes", "central": "central brain", "sensory": "sensory neurons", "visual_projection": "visual projection neurons",
               "ascending": "other", "descending": "other", "visual_centrifugal": "other", "motor": "other", "endocrine": "other"}
TURNTABLE_STEP = 4  # degrees between frames
DIM_ALPHA = 0.18  # the "background" version used while the circuit is shown
ZOOM_DIM_ALPHA = 0.3
# the circuit zoom: a window around the giant fiber's input synapses, drawn at full resolution
ZOOM_PAD_UM = 25.0  # margin around the synapses
SYNAPSE_COLORS = {"LPLC2": "#3ad1c0", "LC4": "#ff7ab6", "GF": "#f5c542"}
SYNAPSE_SIZE = {"LPLC2": 1.2, "LC4": 1.2, "GF": 2.6}
CIRCUIT_STEP = 6  # degrees between circuit-turntable frames


def load():
    p = np.load(POINTS_PATH)
    return p["xyz"].astype(float), p["cls"], [str(c) for c in p["class_names"]]


def center_of(xyz: np.ndarray) -> np.ndarray:
    """Middle of the brain: the midpoint of the 1st-99th percentile box (robust to a few far-away anchor points)."""
    lo, hi = np.percentile(xyz, 1, axis=0), np.percentile(xyz, 99, axis=0)
    return (lo + hi) / 2


def camera(angle_deg: float) -> np.ndarray:
    """Rows: screen-right, screen-up, toward-viewer, in data axes (x = fly's right, y = ventral, z = posterior)."""
    a = np.radians(angle_deg)
    # front view (angle 0): screen-right = fly's left = -x, up = dorsal = -y, toward the viewer = anterior = -z
    right = np.array([-np.cos(a), 0.0, np.sin(a)])
    up = np.array([0.0, -1.0, 0.0])
    toward = np.cross(right, up)
    return np.vstack([right, up, toward])


def project(xyz: np.ndarray, center: np.ndarray, angle_deg: float) -> np.ndarray:
    """Micrometres on screen (x right, y up) and depth (larger = nearer the viewer)."""
    return (xyz - center) @ camera(angle_deg).T


def render(xyz, cls, names, center, angle, path: Path, alpha: float = POINT_ALPHA, px=(IMAGE_W, IMAGE_H),
           window=None, layers=()) -> None:
    """window = (screen x, screen y, half width) in micrometres; default: the whole brain, centered.
    layers = [(xyz, color, size)] drawn after (over) the brain; pass cls=None to draw only the layers."""
    cx, cy, hw = window if window is not None else (0.0, 0.0, HALF_WIDTH_UM)
    fig = plt.figure(figsize=(px[0] / 100, px[1] / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    if cls is not None:
        _scatter_brain(ax, xyz, cls, names, center, angle, alpha, POINT_SIZE if window is None else ZOOM_POINT_SIZE)
    for lxyz, color, size in layers:
        q = project(lxyz, center, angle)
        order = np.argsort(q[:, 2])
        ax.scatter(q[order, 0], q[order, 1], s=size, c=color, alpha=0.85, linewidths=0, rasterized=True)
    half_h = hw * px[1] / px[0]
    ax.set_xlim(cx - hw, cx + hw)
    ax.set_ylim(cy - half_h, cy + half_h)
    ax.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, transparent=True)
    plt.close(fig)


def _scatter_brain(ax, xyz, cls, names, center, angle, alpha, size=POINT_SIZE):
    p = project(xyz, center, angle)
    order = np.argsort(p[:, 2])  # far first, so nearer points are drawn over them
    colors = np.array([CLASS_COLORS[names[c]] for c in cls])[order]
    ax.scatter(p[order, 0], p[order, 1], s=size, c=colors, alpha=alpha, linewidths=0, rasterized=True)


def manifest(center) -> dict:
    return {"center_um": [round(float(v), 3) for v in center], "half_width_um": HALF_WIDTH_UM, "image_px": [IMAGE_W, IMAGE_H],
            "class_colors": CLASS_COLORS, "class_names": CLASS_NAMES, "turntable_step_deg": TURNTABLE_STEP,
            "views": {"front": "front.png", "front_dim": "front_dim.png"}, "turntable": "turntable/{angle:03d}.png",
            "camera_note": "screen = (xyz - center) @ camera(angle).T, see compute/fly_render.py"}


def check() -> None:
    xyz, cls, names = load()
    c = center_of(xyz)
    for angle in (0, 45, 90):
        render(xyz, cls, names, c, angle, CHECK_DIR / f"fly_{angle:03d}.png", px=(900, 500))


def circuit_layers():
    """The synapse point clouds of the circuit (focus side): LPLC2 outputs, LC4 outputs, giant-fiber inputs."""
    t = json.loads(TRACE_PATH.read_text(encoding="utf-8"))
    z = np.load(SYNAPSES_PATH)
    pre, post, sxyz = z["pre"], z["post"], z["xyz"].astype(float)
    gf = int(t["focus"]["id"])
    out = {}
    for typ in ("LPLC2", "LC4"):
        ids = [int(n["id"]) for n in t["looming"][typ]]
        out[typ] = sxyz[np.isin(pre, ids)]
    out["GF"] = sxyz[post == gf]
    return out


def zoom_window(center, layers) -> tuple[float, float, float]:
    """Screen-space window (x, y, half width) that holds the giant fiber's input synapses and the contact zones."""
    q = project(np.vstack([layers["GF"]]), center, 0)
    lo, hi = q[:, :2].min(axis=0) - ZOOM_PAD_UM, q[:, :2].max(axis=0) + ZOOM_PAD_UM
    cx, cy = (lo + hi) / 2
    hw = max((hi[0] - lo[0]) / 2, (hi[1] - lo[1]) / 2 * IMAGE_W / IMAGE_H)
    return float(cx), float(cy), float(hw)


def render_all() -> None:
    xyz, cls, names = load()
    c = center_of(xyz)
    render(xyz, cls, names, c, 0, OUT / "front.png")
    render(xyz, cls, names, c, 0, OUT / "front_dim.png", alpha=DIM_ALPHA)
    angles = list(range(0, 360, TURNTABLE_STEP))
    for angle in angles:
        render(xyz, cls, names, c, angle, OUT / "turntable" / f"{angle:03d}.png")
    m = manifest(c)
    m["turntable_angles"] = angles
    # the circuit: one image per layer in the zoom window, a dim brain behind them, and a turntable of all of it
    layers = circuit_layers()
    win = zoom_window(c, layers)
    render(xyz, cls, names, c, 0, OUT / "zoom" / "brain_dim.png", alpha=ZOOM_DIM_ALPHA, window=win)
    for name, pts in layers.items():
        render(pts, None, names, c, 0, OUT / "zoom" / f"{name}.png", window=win, layers=[(pts, SYNAPSE_COLORS[name], SYNAPSE_SIZE[name])])
    # turntable about the circuit's own middle (the giant fiber's input synapses), keeping the zoom window size
    pivot = layers["GF"].mean(axis=0)
    c_angles = list(range(0, 360, CIRCUIT_STEP))
    for angle in c_angles:
        stack = [(layers[n], SYNAPSE_COLORS[n], SYNAPSE_SIZE[n]) for n in ("LPLC2", "LC4", "GF")]
        render(xyz, cls, names, pivot, angle, OUT / "circuit_turntable" / f"{angle:03d}.png", alpha=ZOOM_DIM_ALPHA,
               window=(0.0, 0.0, win[2]), layers=stack)
    m["zoom"] = {"window_um": [round(v, 3) for v in win], "layers": list(layers), "synapse_colors": SYNAPSE_COLORS,
                 "counts": {k: int(len(v)) for k, v in layers.items()}}
    m["circuit_turntable"] = {"angles": c_angles, "pivot_um": [round(float(v), 3) for v in pivot], "files": "circuit_turntable/{angle:03d}.png"}
    (OUT / "manifest.json").write_text(json.dumps(m, indent=1), encoding="utf-8")
    print(f"wrote {len(angles) + len(c_angles) + 6} images and manifest.json to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    check() if parser.parse_args().check else render_all()
