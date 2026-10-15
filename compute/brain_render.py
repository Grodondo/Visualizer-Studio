"""Render the fsaverage cortex, optionally colored by estimated activity, to transparent PNG frames for Manim.

Why matplotlib and not a 3D engine: it is already installed, it renders offscreen everywhere, and a 20,000-triangle
cortex per hemisphere drawn with flat shading is sharp enough for a 1080p video. (PyVista would be faster and smoother;
switch if frames need to be larger or the camera must move continuously.)

Surface: fsaverage pial surface (FreeSurfer; Fischl et al. 1999), sampled at the ico-5 source-space vertices that the
activity estimate lives on, using the source space's own triangulation. Gray shading = sulcal depth (darker = deeper fold).

    uv run python -m compute.brain_render --check            # a few stills into media/check/ to look at
    uv run python -m compute.brain_render                    # every frame of the punch-reaction video into media/brain_frames/punch_reaction/
                                                             # plus manifest.json (frame list, times, region colors) that the scene reads

Activity colors: the 'inferno' colormap (perceptually uniform, for a magnitude). The color limits come from
compute.punch_reaction: a vertex is colored only where the estimate exceeds the largest value anywhere on the cortex
before the stimulus, and the scale tops out at the 99.5th percentile of the values shown.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mne  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SUBJECTS_DIR = ROOT / "assets" / "data" / "fsaverage_subjects"
SOURCES_PATH = ROOT / "assets" / "punch_reaction" / "sources.npz"
FRAMES_DIR = ROOT / "media" / "brain_frames"
CHECK_DIR = ROOT / "media" / "check"

# ---- look ------------------------------------------------------------------------------------------
IMAGE_PX = 900  # square frame size in pixels
GYRUS_GRAY, SULCUS_GRAY = 0.78, 0.42  # base color of the cortex: crowns of folds light, depths darker
LIGHT = np.array([0.35, 0.45, 1.0])  # direction toward the light, in camera coordinates (x right, y up, z toward viewer)
AMBIENT = 0.35
CMAP = "inferno"
COLOR_RANGE = (0.35, 1.0)  # part of the colormap used: skip the near-black start so weak activity is still visible

# camera: (azimuth, elevation) in degrees; azimuth 180 looks at the left side, 0 at the right side, 90 from the front,
# -90 from the back; elevation 90 looks from straight above
VIEWS = {
    "left": (180, 0),
    "right": (0, 0),
    "back": (-90, 10),
    "top": (-90, 90),
    "left_back": (-140, 15),
    "right_back": (-40, 15),
    "left_medial": (0, 0),  # the left hemisphere seen from the midline (the right one is not drawn)
}
HEMIS = {"left_medial": "lh"}  # views that draw one hemisphere only

# Anatomy stages: Desikan-Killiany atlas regions (Desikan et al. 2006) standing in, approximately, for functional areas.
# Colors are repeated in the manifest so the video's labels use exactly the same ones.
STAGES = [
    {"key": "input", "name": "primary visual cortex (V1)", "color": "#5ec8ff", "labels": ["pericalcarine-lh", "pericalcarine-rh"]},
    {"key": "processing", "name": "visual and parietal areas", "color": "#b69cff",
     "labels": [f"{n}-{h}" for n in ("lateraloccipital", "inferiorparietal", "superiorparietal") for h in ("lh", "rh")]},
    {"key": "planning", "name": "premotor cortex", "color": "#f5c542", "labels": ["caudalmiddlefrontal-lh", "caudalmiddlefrontal-rh"]},
    {"key": "movement", "name": "primary motor cortex (left: right hand)", "color": "#ff9a3d", "labels": ["precentral-lh"]},
]
ANATOMY_VIEWS = ("left", "left_medial")
ACTIVITY_VIEWS = ("back", "right_back")
LIMIT_VIEW = "top"
OUT = FRAMES_DIR / "punch_reaction"


@lru_cache(maxsize=None)
def cortex(lh_vertno: tuple, rh_vertno: tuple):
    """Vertices (n x 3, mm), triangles (m x 3) and sulcal depth (n) for both hemispheres, at the source vertices."""
    src = mne.read_source_spaces(SUBJECTS_DIR / "fsaverage" / "bem" / "fsaverage-ico-5-src.fif", verbose=False)
    verts, tris, sulc, offset = [], [], [], 0
    for hemi, s, vertno in (("lh", src[0], lh_vertno), ("rh", src[1], rh_vertno)):
        assert np.array_equal(s["vertno"], vertno), "the activity estimate must use the fsaverage ico-5 source space"
        pial, _ = mne.read_surface(SUBJECTS_DIR / "fsaverage" / "surf" / f"{hemi}.pial", verbose=False)
        depth = mne.surface.read_curvature(SUBJECTS_DIR / "fsaverage" / "surf" / f"{hemi}.sulc", binary=False)
        remap = np.full(len(pial), -1)
        remap[s["vertno"]] = np.arange(len(s["vertno"]))
        t = remap[s["use_tris"]]
        assert (t >= 0).all()
        verts.append(pial[s["vertno"]])
        tris.append(t + offset)
        sulc.append(depth[s["vertno"]])
        offset += len(s["vertno"])
    return np.vstack(verts), np.vstack(tris), np.concatenate(sulc)


def rotation(azimuth: float, elevation: float) -> np.ndarray:
    """World (RAS: x right, y front, z up) -> camera (x right on screen, y up, z toward the viewer)."""
    a, e = np.radians(azimuth), np.radians(elevation)
    view_dir = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])  # from the brain toward the camera
    up = np.array([0.0, 0.0, 1.0]) if abs(e) < np.radians(80) else np.array([0.0, 1.0, 0.0])
    right = np.cross(up, view_dir)
    right /= np.linalg.norm(right)
    up = np.cross(view_dir, right)
    return np.vstack([right, up, view_dir])


def base_colors(sulc: np.ndarray) -> np.ndarray:
    s = np.clip(sulc / np.percentile(np.abs(sulc), 95), -1, 1)  # FreeSurfer sulc: positive = deep
    gray = GYRUS_GRAY + (SULCUS_GRAY - GYRUS_GRAY) * (s + 1) / 2
    return np.repeat(gray[:, None], 3, axis=1)


def activity_colors(base: np.ndarray, values: np.ndarray | None, vmin: float, vmax: float) -> np.ndarray:
    if values is None:
        return base
    colors = base.copy()
    on = values > vmin
    t = np.clip((values[on] - vmin) / (vmax - vmin), 0, 1)
    cmap = plt.get_cmap(CMAP)
    colors[on] = cmap(COLOR_RANGE[0] + t * (COLOR_RANGE[1] - COLOR_RANGE[0]))[:, :3]
    return colors


def render(verts, tris, vertex_colors, view: str, path: Path, px: int = IMAGE_PX, margin: float = 6.0) -> None:
    """Orthographic, flat-shaded, back faces culled, faces painted far to near. Transparent background.
    The framing (center and zoom) depends only on the view, so frames of one view line up exactly."""
    r = rotation(*VIEWS[view])
    if view in HEMIS:  # keep only the triangles of that hemisphere (left = the first half of the vertices)
        n_lh = len(verts) // 2
        tris = tris[(tris < n_lh).all(axis=1)] if HEMIS[view] == "lh" else tris[(tris >= n_lh).all(axis=1)]
        used = np.unique(tris)
        center = verts[used].mean(axis=0)
    else:
        center = verts.mean(axis=0)
    cam = (verts - center) @ r.T
    tri_pts = cam[tris]  # m x 3 x 3
    normals = np.cross(tri_pts[:, 1] - tri_pts[:, 0], tri_pts[:, 2] - tri_pts[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    front = normals[:, 2] > 0
    light = LIGHT / np.linalg.norm(LIGHT)
    shade = AMBIENT + (1 - AMBIENT) * np.clip(normals[front] @ light, 0, 1)
    face_color = vertex_colors[tris[front]].mean(axis=1) * shade[:, None]
    order = np.argsort(tri_pts[front][:, :, 2].mean(axis=1))
    polys = tri_pts[front][order][:, :, :2]
    half = np.abs(cam[np.unique(tris)][:, :2]).max() + margin
    fig = plt.figure(figsize=(px / 100, px / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.add_collection(PolyCollection(polys, facecolors=np.clip(face_color[order], 0, 1), edgecolors="none", antialiased=False))
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_aspect("equal")
    ax.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, transparent=True)
    plt.close(fig)


def region_vertices(names: list[str], lh_vertno, rh_vertno) -> np.ndarray:
    """Indices (into the stacked lh+rh source vertices) of the given atlas labels."""
    labels = {l.name: l for l in mne.read_labels_from_annot("fsaverage", "aparc", subjects_dir=SUBJECTS_DIR, verbose=False)}
    out = []
    for name in names:
        label = labels[name]
        vertno, offset = (lh_vertno, 0) if label.hemi == "lh" else (rh_vertno, len(lh_vertno))
        out.append(np.flatnonzero(np.isin(vertno, label.vertices)) + offset)
    return np.concatenate(out)


def hex_rgb(color: str) -> np.ndarray:
    return np.array([int(color[i:i + 2], 16) / 255 for i in (1, 3, 5)])


def load_sources():
    s = np.load(SOURCES_PATH)
    return s, cortex(tuple(s["lh_vertno"]), tuple(s["rh_vertno"]))


def check() -> None:
    s, (verts, tris, sulc) = load_sources()
    base = base_colors(sulc)
    for view in VIEWS:
        render(verts, tris, base, view, CHECK_DIR / f"brain_{view}.png", px=500)
    t, d = s["stim_times"], s["stim_data"]
    vmin, vmax = (float(v) for v in s["stim_limits"])
    print(f"stimulus-locked color limits: {vmin:.2f} .. {vmax:.2f}")
    for when in (0.12, 0.16, 0.20, 0.24):
        i = int(np.argmin(np.abs(t - when)))
        for view in ("back", "right_back", "right", "left_back"):
            render(verts, tris, activity_colors(base, d[:, i], vmin, vmax), view, CHECK_DIR / f"stim_{view}_{int(when * 1000):03d}.png", px=500)


def render_all() -> None:
    """Every brain image the punch-reaction video uses, and manifest.json describing them."""
    import json

    s, (verts, tris, sulc) = load_sources()
    base = base_colors(sulc)
    lh, rh = s["lh_vertno"], s["rh_vertno"]
    manifest = {"source": str(SOURCES_PATH.relative_to(ROOT)), "cmap": CMAP, "color_range": COLOR_RANGE,
                "stages": [], "activity": {}, "limit": {}}
    # anatomy: stage k shows stages 1..k colored
    colors = base.copy()
    for view in ANATOMY_VIEWS:
        render(verts, tris, base, view, OUT / "anatomy" / f"{view}_0.png")
    for k, stage in enumerate(STAGES, start=1):
        idx = region_vertices(stage["labels"], lh, rh)
        colors[idx] = 0.25 * base[idx] + 0.75 * hex_rgb(stage["color"])  # keep a hint of the folds under the color
        for view in ANATOMY_VIEWS:
            render(verts, tris, colors, view, OUT / "anatomy" / f"{view}_{k}.png")
        manifest["stages"].append({k2: stage[k2] for k2 in ("key", "name", "color", "labels")})
    # estimated activity after the arrows, one frame per sample
    t, d = s["stim_times"], s["stim_data"]
    vmin, vmax = (float(v) for v in s["stim_limits"])
    for i, ti in enumerate(t):
        for view in ACTIVITY_VIEWS:
            render(verts, tris, activity_colors(base, d[:, i], vmin, vmax), view, OUT / "activity" / f"{view}_{i:03d}.png")
    manifest["activity"] = {"views": list(ACTIVITY_VIEWS), "times": [round(float(x), 5) for x in t],
                            "vmin": vmin, "vmax": vmax, "files": "activity/{view}_{index:03d}.png"}
    # the limitation: right-minus-left-hand presses, 150 ms before the press
    lo, hi = (float(v) for v in s["limit_limits"])
    render(verts, tris, base, LIMIT_VIEW, OUT / "limit" / "base.png")
    render(verts, tris, activity_colors(base, s["limit_map"], lo, hi), LIMIT_VIEW, OUT / "limit" / "hand_contrast.png")
    manifest["limit"] = {"view": LIMIT_VIEW, "vmin": lo, "vmax": hi}
    manifest["anatomy"] = {"views": list(ANATOMY_VIEWS), "files": "anatomy/{view}_{stage}.png"}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"wrote {len(t) * len(ACTIVITY_VIEWS) + 2 * (len(STAGES) + 1) + 2} images and manifest.json to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    check() if args.check else render_all()
