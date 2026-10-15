"""Draw the model's SIMULATED spikes, one transparent image per game tic, at the neurons' measured FlyWire positions.

    uv run --group flydoom python -m flydoom.activity_frames      # ~3 minutes -> media/flydoom_fly_activity/

Same camera as flydoom/brain_frames.py (seen from behind, so the fly's left is the viewer's left, like the game) and the same
framing as media/flydoom_brain/back_all.png, so an activity frame lies exactly on top of the anatomy image.
A neuron that spiked during a tic is drawn bright on that tic and fades over the next GLOW_TICS tics (so a 28 ms tic is visible
on video). Colors encode the neuron's ROLE, never activity strength:
  input cells LC4 + LPLC2   left = cyan, right = amber (the side of the eye that drives them)
  steering DNa01 + DNa02    white, large
  giant fibers DNp01        white ring
  every other neuron        magenta (simulated activity anywhere else in the brain)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from compute import fly_render as fr  # noqa: E402

from . import model_data as md  # noqa: E402
from .fly_controller import groups  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "media" / "flydoom_fly_activity"
ANNOTATIONS = ROOT / "assets" / "data" / "flywire" / "Supplemental_file1_neuron_annotations_v2.1.0.tsv"
ANGLE, PX = 180, (2400, 1333)  # must match flydoom/brain_frames.py
GLOW_TICS = 4
COLORS = {"left": "#5ec8ff", "right": "#ffb454", "steer": "#ffffff", "giant": "#ffffff", "other": "#ff5fd2"}
SIZES = {"left": 10, "right": 10, "steer": 90, "giant": 60, "other": 6}


def model_positions(con: md.Connectome) -> np.ndarray:
    """Micrometres, in model index order (soma, or the anchor point when the soma is outside the brain)."""
    a = pd.read_csv(ANNOTATIONS, sep="\t", low_memory=False, usecols=["root_id", "soma_x", "soma_y", "soma_z", "pos_x", "pos_y", "pos_z"])
    a = a.set_index(a["root_id"].astype(np.int64)).loc[con.flywire_ids]
    soma = a[["soma_x", "soma_y", "soma_z"]].to_numpy(float)
    pos = a[["pos_x", "pos_y", "pos_z"]].to_numpy(float)
    return np.where(np.isnan(soma).any(axis=1)[:, None], pos, soma) * np.array([4.0, 4.0, 40.0]) / 1000.0


def main() -> None:
    con = md.load("783")
    xyz_all, _, _ = fr.load()
    center = fr.center_of(xyz_all)  # identical to brain_frames.py
    xy = fr.project(model_positions(con), center, ANGLE)[:, :2]
    g = groups(con)
    role = np.full(con.n, "other", dtype=object)
    for side in ("left", "right"):
        for t in ("LC4", "LPLC2"):
            role[g[(t, side)]] = side
        for t in ("DNa01", "DNa02"):
            role[g[(t, side)]] = "steer"
        role[g[("DNp01", side)]] = "giant"
    act = np.load(ROOT / "assets" / "flydoom" / "fly_activity.npz")
    idx, off = act["idx"], act["offsets"]
    footage = json.loads((ROOT / "assets" / "flydoom" / "fly_footage.json").read_text(encoding="utf-8"))
    n_tics = len(off) - 1
    half_h = fr.HALF_WIDTH_UM * PX[1] / PX[0]
    starts = np.cumsum([0] + [json.loads((ROOT / "assets" / "flydoom" / c).read_text(encoding="utf-8"))["tics"] for c in footage["clips"]])
    OUT.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(PX[0] / 100, PX[1] / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    for t in range(n_tics):
        ax.clear()
        ax.set_xlim(-fr.HALF_WIDTH_UM, fr.HALF_WIDTH_UM)
        ax.set_ylim(-half_h, half_h)
        ax.axis("off")
        episode_start = starts[np.searchsorted(starts, t, side="right") - 1]
        for age in range(GLOW_TICS - 1, -1, -1):  # older (fainter) first
            k = t - age
            if k < episode_start:  # no glow carried across episodes
                continue
            spk = idx[off[k]:off[k + 1]]
            alpha = 1.0 if age == 0 else 0.55 ** age
            for r in ("other", "left", "right", "giant", "steer"):
                sel = spk[role[spk] == r]
                if len(sel):
                    if r == "giant":
                        ax.scatter(xy[sel, 0], xy[sel, 1], s=SIZES[r], facecolors="none", edgecolors=COLORS[r], linewidths=1.5, alpha=alpha)
                    else:
                        ax.scatter(xy[sel, 0], xy[sel, 1], s=SIZES[r], c=COLORS[r], alpha=alpha * (0.8 if r == "other" else 1.0), linewidths=0)
        fig.savefig(OUT / f"{t:05d}.png", transparent=True)
        if t % 300 == 0:
            print(f"  {t}/{n_tics}", flush=True)
    plt.close(fig)
    (OUT / "manifest.json").write_text(json.dumps({"tics": n_tics, "angle": ANGLE, "px": PX, "glow_tics": GLOW_TICS, "colors": COLORS,
                                                   "center_um": [float(v) for v in center], "half_width_um": fr.HALF_WIDTH_UM}, indent=1), encoding="utf-8")
    print(f"wrote {n_tics} activity frames to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
