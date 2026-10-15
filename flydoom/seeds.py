"""Seed sets whose episodes are genuinely different.

Finding (2026-10-15): many ViZDoom seeds produce the SAME take_cover episode. Among seeds 1-120, 69 gave one identical
fireball pattern; the original test seeds 1007, 1008 and 1010 were identical episodes, and calibration seeds 9001-9004 were
identical to each other. Why is not known (probably: the scenario draws few random numbers); it is checked here, not assumed.

Rule: walk upward from a start seed, play each candidate with the idle controller (a fixed, input-independent player),
and keep a seed only if its full episode (SHA-256 of every frame) differs from every seed kept so far, in BOTH sets.
If two seeds give the same idle episode the game world behaves identically, so every controller would play the same game;
distinct idle episodes therefore mean distinct test conditions. flydoom.trials re-checks distinctness per controller.

    uv run --group flydoom python -m flydoom.seeds     # writes assets/flydoom/seeds.json
"""
from __future__ import annotations

import json
from pathlib import Path

from . import controllers as C
from .game import run_episode

ROOT = Path(__file__).resolve().parent.parent
SEEDS_PATH = ROOT / "assets" / "flydoom" / "seeds.json"
N_CALIBRATION, N_TEST = 5, 10
CALIBRATION_START, TEST_START = 9001, 1001
MAX_CANDIDATES = 400  # give up instead of searching forever


def screen(start: int, n: int, taken: dict) -> tuple[list[int], dict, list[int]]:
    kept, rejected = [], []
    for seed in range(start, start + MAX_CANDIDATES):
        h = run_episode(C.Idle(), seed).frame_hash
        if h in taken:
            rejected.append(seed)
            continue
        taken[h] = seed
        kept.append(seed)
        if len(kept) == n:
            return kept, taken, rejected
    raise RuntimeError(f"only {len(kept)} distinct episodes among {MAX_CANDIDATES} seeds from {start}")


def main() -> None:
    taken: dict = {}
    cal, taken, cal_rej = screen(CALIBRATION_START, N_CALIBRATION, taken)
    test, taken, test_rej = screen(TEST_START, N_TEST, taken)
    hashes = {seed: h for h, seed in taken.items()}
    out = {"rule": "first seeds upward from the start whose idle episode differs from every kept seed (both sets)",
           "calibration": cal, "test": test, "rejected_as_duplicates": {"calibration": cal_rej, "test": test_rej},
           "idle_episode_sha256": {str(s): hashes[s] for s in cal + test}}
    SEEDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SEEDS_PATH.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"calibration {cal} (skipped {len(cal_rej)} duplicates)\ntest {test} (skipped {len(test_rej)} duplicates)")


def load() -> tuple[list[int], list[int]]:
    s = json.loads(SEEDS_PATH.read_text(encoding="utf-8"))
    return s["calibration"], s["test"]


if __name__ == "__main__":
    main()
