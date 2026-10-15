"""Exports for the draft video (run after flydoom.trials; it reads the calibrated parameters from results.json).

    uv run --group flydoom python -m flydoom.export

  assets/flydoom/clip_<controller>_<seed>.json   per tic: action, health, eye output, visible fireballs (ground truth)
  media/flydoom_frames/<controller>_<seed>/       the game frames, and eye/ = what the eye counts as "bright" (tinted)
  assets/flydoom/fly_io.json                      FlyWire positions (micrometres) of the cell types the fly pipeline will
                                                  use: inputs LC4 and LPLC2, outputs DNa01, DNa02 (steering) and DNp01
                                                  (giant fiber), both sides. Positions only: nothing is simulated here.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from . import controllers as C
from .game import TICRATE, run_episode
from .metrics import episode_metrics

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets" / "flydoom"
FRAMES = ROOT / "media" / "flydoom_frames"
ANNOTATIONS = ROOT / "assets" / "data" / "flywire" / "Supplemental_file1_neuron_annotations_v2.1.0.tsv"  # from compute.connectome
VOXEL_NM = np.array([4.0, 4.0, 40.0])
EYE_TINT = (94, 200, 255)  # the eye's view, drawn over the game
IO_TYPES = {"LC4": "input: object approach speed", "LPLC2": "input: object size",
            "DNa01": "output: steering", "DNa02": "output: steering", "DNp01": "output: giant fiber (escape)"}
CLIP_CONTROLLER = "reflex"
CLIP_MIN_SECONDS = 40.0  # footage rule: this controller's test episodes IN SEED ORDER, from the first, until this much play
# (no episode is chosen by its outcome, none is repeated, every death stays in)


def export_clip(controller, seed: int) -> Path:
    log = run_episode(controller, seed, keep_frames=True)
    folder = FRAMES / f"{controller.name}_{seed}"
    (folder / "eye").mkdir(parents=True, exist_ok=True)
    eye = C.LoomingEye()
    for i, frame in enumerate(log.frames):
        Image.fromarray(frame).save(folder / f"{i:04d}.png")
        m = eye.bright_mask(frame)
        rgba = np.zeros((*m.shape, 4), np.uint8)
        rgba[m] = (*EYE_TINT, 255)
        Image.fromarray(rgba).save(folder / "eye" / f"{i:04d}.png")
    clip = {"controller": controller.name, "uses": controller.uses, "seed": seed, "tics": log.tics, "died": log.died,
            "ticrate": TICRATE, "actions": log.actions, "health": log.health, "final_health": log.final_health,
            "eye": log.eye, "eye_fields": ["size_left_deg", "size_right_deg", "expansion_left_dps", "expansion_right_dps"],
            "fireballs": log.fireballs, "frames": str(folder.relative_to(ROOT)).replace("\\", "/"), "metrics": episode_metrics(log)}
    path = OUT / f"clip_{controller.name}_{seed}.json"
    path.write_text(json.dumps(clip, separators=(",", ":")), encoding="utf-8")
    return path


def export_fly_io() -> Path:
    a = pd.read_csv(ANNOTATIONS, sep="\t", low_memory=False)
    a["type"] = a["cell_type"].fillna(a["hemibrain_type"])
    a = a[a["type"].isin(IO_TYPES)]
    soma = a[["soma_x", "soma_y", "soma_z"]].to_numpy(float)
    pos = a[["pos_x", "pos_y", "pos_z"]].to_numpy(float)
    xyz = np.where(np.isnan(soma).any(axis=1)[:, None], pos, soma) * VOXEL_NM / 1000.0
    cells = [{"type": t, "side": s, "xyz": [round(float(v), 2) for v in p]} for t, s, p in zip(a["type"], a["side"], xyz)]
    counts = a.groupby(["type", "side"]).size()
    out = {"source": "FlyWire FAFB v783 annotations v2.1.0 (Schlegel et al. 2024), CC BY 4.0", "roles": IO_TYPES,
           "counts": {f"{t}/{s}": int(n) for (t, s), n in counts.items()}, "cells": cells}
    path = OUT / "fly_io.json"
    path.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    return path


def footage_seeds(results: dict) -> list[int]:
    tics = {e["seed"]: e["tics"] for e in results["episodes"] if e["controller"] == CLIP_CONTROLLER}
    chosen, total = [], 0.0
    for seed in results["test_seeds"]:
        chosen.append(seed)
        total += tics[seed] / TICRATE
        if total >= CLIP_MIN_SECONDS:
            break
    return chosen


def main() -> None:
    results = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    threshold, area = results["reflex_calibration"]["chosen"], results["scripted_calibration"]["chosen"]
    ctl = next(c for c in C.all_controllers(threshold, area) if c.name == CLIP_CONTROLLER)
    for old in OUT.glob("clip_*.json"):
        old.unlink()
    seeds = footage_seeds(results)
    clips = [export_clip(ctl, s).name for s in seeds]
    (OUT / "footage.json").write_text(json.dumps({"controller": CLIP_CONTROLLER, "rule": f"test seeds in order until >= {CLIP_MIN_SECONDS:g} s",
                                                  "seeds": seeds, "clips": clips}, indent=1), encoding="utf-8")
    print("wrote", clips, "footage.json and", export_fly_io().name)


if __name__ == "__main__":
    main()
