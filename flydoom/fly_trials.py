"""Milestone 3: the fly controllers on the same distinct seeds as the baselines, scored with the same metrics.

    uv run --group flydoom python -m flydoom.fly_trials      # ~20 minutes -> assets/flydoom/fly_results.json

1. Calibration (CALIBRATION seeds only): GAIN for controller A, from GAIN_GRID, best median survival (ties -> smaller gain).
   B and D reuse A's gain unchanged: they are controls, not separately tuned.
2. Test: A (fly), B (shuffled wiring), D (giant fibers silenced) on the 10 test seeds; A's first test seed is run twice to
   check the whole loop (game + model) is reproducible bit for bit.
3. The baselines (idle, constant, random, reflex, scripted) are read from results.json, not re-run.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .fly_controller import FlyBrain, RATE_MAX, TAU_MS, THRESHOLD_HZ
from .game import run_episode
from .metrics import episode_metrics, summarise
from .seeds import load as load_seeds

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets" / "flydoom" / "fly_results.json"
RESULTS = ROOT / "assets" / "flydoom" / "results.json"
GAIN_GRID = [0.5, 1.0, 2.0, 4.0, 8.0]  # Hz of input per deg/s of expansion
CONTROLS = {"fly": {}, "fly_shuffled": {"wiring": "shuffled"}, "fly_no_giant_fiber": {"silence_giant_fibers": True}}


def fly_extra(ctl: FlyBrain) -> dict:
    c = ctl.log["counts"]
    return {"spikes": {k: int(sum(v)) for k, v in c.items()}, "mean_active_per_tic": round(float(np.mean(ctl.log["active"])), 1)}


def main() -> None:
    t0 = time.time()
    cal_seeds, test_seeds = load_seeds()
    table = {}
    for gain in GAIN_GRID:
        table[gain] = float(np.median([run_episode(FlyBrain(gain), s).survival_s for s in cal_seeds]))
        print(f"  calibration gain {gain}: median survival {table[gain]:.2f} s", flush=True)
    gain = max(GAIN_GRID, key=lambda g: (table[g], -g))
    rows, extra, hashes = [], {}, {}
    for name, kw in CONTROLS.items():
        ctl = FlyBrain(gain, **kw)
        for s in test_seeds:
            log = run_episode(ctl, s)
            m = episode_metrics(log)
            rows.append(m)
            extra[f"{name}/{s}"] = fly_extra(ctl)
            hashes[(name, s)] = (log.frame_hash, log.actions)
        print(f"  {name} done ({time.time() - t0:.0f} s)", flush=True)
    again = run_episode(FlyBrain(gain), test_seeds[0])
    repro = (again.frame_hash, again.actions) == hashes[("fly", test_seeds[0])]
    by = {}
    for r in rows:
        by.setdefault(r["controller"], []).append(r)
    base = json.loads(RESULTS.read_text(encoding="utf-8"))
    out = {"design": {"encoder": f"rate = min({RATE_MAX:g} Hz, gain x expansion), both LC4 and LPLC2 of that side",
                      "decoder": f"DNa01+DNa02 per side, low-pass {TAU_MS:g} ms, dead band {THRESHOLD_HZ:g} Hz, ipsilateral step",
                      "model": "Shiu et al. 2024 LIF on FlyWire v783 (flydoom/lif.py)"},
           "calibration": {"grid": {str(k): round(v, 3) for k, v in table.items()}, "chosen_gain": gain, "seeds": cal_seeds},
           "test_seeds": test_seeds, "reproducible_rerun": {"seed": test_seeds[0], "identical": bool(repro)},
           "distinct_episodes": {k: len({r["frame_hash"] for r in v}) for k, v in by.items()},
           "controllers": {k: {"summary": summarise(v)} for k, v in by.items()},
           "episodes": rows, "neural": extra,
           "baselines": {k: v["summary"] for k, v in base["controllers"].items()}, "runtime_s": round(time.time() - t0)}
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} in {out['runtime_s']} s; rerun identical: {repro}")


if __name__ == "__main__":
    main()
