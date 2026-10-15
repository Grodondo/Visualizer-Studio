"""Exports for the fly-plays cut: the fly controller's own episodes, with the model's simulated activity per tic.

    uv run --group flydoom python -m flydoom.fly_export        # ~2 minutes; then flydoom.activity_frames draws the brain
    uv run --group flydoom python -m flydoom.fly_export --shuffle   # the shuffled-wiring control on the first test seed

Footage rule (same as Milestone 1, fixed in advance): the fly controller's test episodes IN SEED ORDER until >= 40 s.
Each episode is re-run with the calibrated gain from fly_results.json; its actions must equal the recorded test episode
(checked here and in tests/test_fly_visual.py), so the footage is the experiment, not a new run that happens to look good.

Writes (all derived, small):
  assets/flydoom/fly_footage.json            seeds and clip files
  assets/flydoom/fly_clip_<seed>.json        per tic: action, eye, encoder rates, filtered steering rates, cell-group spike counts
  assets/flydoom/fly_activity.npz            per tic: model indices of every neuron that spiked (simulated), all clips concatenated
  media/flydoom_fly_frames/<seed>/           game frames and the eye overlay (regenerated, not tracked)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from .export import EYE_TINT
from .eye import LoomingEye
from .fly_controller import FlyBrain
from .game import TICRATE, run_episode
from .metrics import episode_metrics

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets" / "flydoom"
FRAMES = ROOT / "media" / "flydoom_fly_frames"
SHUFFLE_FRAMES = ROOT / "media" / "flydoom_shuffle_frames"
MIN_SECONDS = 40.0


def record(ctl: FlyBrain, seed: int, expected_hash: str, folder: Path, clip_name: str, gain: float) -> list:
    """Re-run one recorded episode, check it equals the recorded one, save frames and the clip JSON. Returns the per-tic spikes."""
    log = run_episode(ctl, seed, keep_frames=True)
    m = episode_metrics(log)
    assert m["frame_hash"] == expected_hash, f"seed {seed}: the re-run differs from the recorded test episode"
    (folder / "eye").mkdir(parents=True, exist_ok=True)
    eye = LoomingEye()
    for i, frame in enumerate(log.frames):
        Image.fromarray(frame).save(folder / f"{i:04d}.png")
        mask = eye.bright_mask(frame)
        rgba = np.zeros((*mask.shape, 4), np.uint8)
        rgba[mask] = (*EYE_TINT, 255)
        Image.fromarray(rgba).save(folder / "eye" / f"{i:04d}.png")
    L = ctl.log
    clip = {"seed": seed, "controller": ctl.name, "tics": log.tics, "ticrate": TICRATE, "gain": gain, "actions": log.actions, "eye": log.eye,
            "fireballs": log.fireballs, "rates_in": L["rates_in"], "steer_hz": L["steer_hz"], "counts": L["counts"],
            "frames": str(folder.relative_to(ROOT)).replace("\\", "/"), "metrics": m}
    (OUT / clip_name).write_text(json.dumps(clip, separators=(",", ":")), encoding="utf-8")
    print(f"  {ctl.name} seed {seed}: {log.tics} tics, mean {np.mean([len(a) for a in L['active']]):.0f} neurons spiking per tic", flush=True)
    return L["active"]


def save_activity(path: Path, per_clip: list) -> None:
    flat = [a for clip in per_clip for a in clip]
    offsets = np.cumsum([0] + [len(a) for a in flat])
    np.savez_compressed(path, idx=np.concatenate(flat).astype(np.int32), offsets=offsets.astype(np.int64))


def main() -> None:
    res = json.loads((OUT / "fly_results.json").read_text(encoding="utf-8"))
    gain = res["calibration"]["chosen_gain"]
    recorded = {e["seed"]: e for e in res["episodes"] if e["controller"] == "fly"}
    seeds, total = [], 0.0
    for s in res["test_seeds"]:
        seeds.append(s)
        total += recorded[s]["tics"] / TICRATE
        if total >= MIN_SECONDS:
            break
    clips, active = [], []
    for s in seeds:
        clips.append(f"fly_clip_{s}.json")
        active.append(record(FlyBrain(gain, record_all=True), s, recorded[s]["frame_hash"], FRAMES / str(s), clips[-1], gain))
    save_activity(OUT / "fly_activity.npz", active)
    (OUT / "fly_footage.json").write_text(json.dumps({"rule": f"fly test episodes in seed order until >= {MIN_SECONDS:g} s",
                                                      "seeds": seeds, "clips": clips, "gain": gain}, indent=1), encoding="utf-8")
    print(f"wrote {len(clips)} clips ({total:.1f} s), fly_activity.npz, fly_footage.json")


def main_shuffle() -> None:
    """The control for the video: the first test seed with the wiring shuffled (same eye, encoder, decoder and gain)."""
    res = json.loads((OUT / "fly_results.json").read_text(encoding="utf-8"))
    gain, seed = res["calibration"]["chosen_gain"], res["test_seeds"][0]
    recorded = next(e for e in res["episodes"] if e["controller"] == "fly_shuffled" and e["seed"] == seed)
    active = record(FlyBrain(gain, wiring="shuffled", record_all=True), seed, recorded["frame_hash"], SHUFFLE_FRAMES / str(seed), f"shuffle_clip_{seed}.json", gain)
    save_activity(OUT / "shuffle_activity.npz", [active])
    (OUT / "shuffle_footage.json").write_text(json.dumps({"rule": "the first test seed, shuffled wiring", "seeds": [seed], "clips": [f"shuffle_clip_{seed}.json"], "gain": gain}, indent=1), encoding="utf-8")
    print(f"wrote shuffle_clip_{seed}.json, shuffle_activity.npz, shuffle_footage.json")


if __name__ == "__main__":
    main_shuffle() if "--shuffle" in sys.argv else main()
