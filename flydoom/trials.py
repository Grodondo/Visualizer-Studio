"""Run every controller on the same seeds, check reproducibility, and write the results.

    uv run --group flydoom python -m flydoom.trials

1. Calibration (seeds CALIBRATION_SEEDS only): the reflex threshold (REFLEX_GRID) and the scripted AI's minimum blob
   area (SCRIPTED_GRID) are the values with the best median survival. Nothing else is tuned; test seeds never choose anything.
2. Test (seeds TEST_SEEDS): every controller plays every seed TWICE; both runs must give the same frame hash.
3. Writes assets/flydoom/results.json (no personal data), then runs flydoom.export (the clip for the draft video).
"""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
import vizdoom as vzd

from . import controllers as C
from .eye import BRIGHT_LUMINANCE
from .game import EPISODE_CAP_TICS, SCENARIO, TICRATE, run_episode
from .metrics import EVENT_DPS, REACTION_WINDOW_S, episode_metrics, summarise

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets" / "flydoom"
# Seeds screened for DISTINCT episodes by flydoom.seeds (many ViZDoom seeds give identical take_cover games; see that module).
# The first version used 9001-9005 and 1001-1010 unscreened: 9001-9004 were one and the same game, and so were 1007, 1008, 1010.
from .seeds import load as _load_seeds  # noqa: E402

CALIBRATION_SEEDS, TEST_SEEDS = _load_seeds()
REFLEX_GRID = [10.0, 20.0, 40.0, 80.0]  # deg/s of expansion
SCRIPTED_GRID = [6.0, 25.0, 100.0, 400.0]  # pixels of the biggest blob


def calibrate(make, grid) -> dict:
    table = {v: float(np.median([run_episode(make(v), s).survival_s for s in CALIBRATION_SEEDS])) for v in grid}
    best = max(grid, key=lambda v: (table[v], -v))  # ties -> the smaller value
    return {"grid": {str(k): round(v, 3) for k, v in table.items()}, "chosen": best, "seeds": CALIBRATION_SEEDS}


def run_tests(threshold: float, area: float) -> tuple[list[dict], list[dict]]:
    rows, mismatches = [], []
    for ctl in C.all_controllers(threshold, area):
        for seed in TEST_SEEDS:
            a, b = run_episode(ctl, seed), run_episode(ctl, seed)
            if (a.frame_hash, a.actions) != (b.frame_hash, b.actions):
                mismatches.append({"controller": ctl.name, "seed": seed})
            rows.append(episode_metrics(a))
    return rows, mismatches


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    cal = calibrate(C.Reflex, REFLEX_GRID)
    cal_s = calibrate(C.Scripted, SCRIPTED_GRID)
    print("reflex calibration:", cal)
    print("scripted calibration:", cal_s)
    rows, mismatches = run_tests(cal["chosen"], cal_s["chosen"])
    by = {}
    for r in rows:
        by.setdefault(r["controller"], []).append(r)
    ctl_info = {c.name: c.uses for c in C.all_controllers(cal["chosen"], cal_s["chosen"])}
    results = {
        "scenario": SCENARIO, "vizdoom": vzd.__version__, "python": platform.python_version(), "platform": platform.platform(),
        "episode_cap_s": EPISODE_CAP_TICS / TICRATE, "test_seeds": TEST_SEEDS, "eye": {"bright_luminance": BRIGHT_LUMINANCE},
        "metrics": {"event_dps": EVENT_DPS, "reaction_window_s": REACTION_WINDOW_S},
        "reflex_calibration": cal, "scripted_calibration": cal_s,
        "distinct_episodes": {name: len({r["frame_hash"] for r in rs}) for name, rs in by.items()},
        "reproducibility": {"episodes_run_twice": len(rows), "mismatches": mismatches},
        "controllers": {name: {"uses": ctl_info[name], "summary": summarise(rs)} for name, rs in by.items()},
        "episodes": rows,
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"wrote {OUT / 'results.json'} in {time.time() - t0:.0f} s; reproducibility mismatches: {len(mismatches)}")
    from . import export  # the clip and anatomy exports for the video

    export.main()


if __name__ == "__main__":
    main()
