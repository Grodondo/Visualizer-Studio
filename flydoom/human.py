"""Controller G: you. Play the same seeded episodes as the controllers, with the keyboard, in a window.

    uv run --group flydoom python -m flydoom.human               # the five test seeds 1001-1005, in order
    uv run --group flydoom python -m flydoom.human --practice    # one practice episode (seed 4242), nothing saved

Keys: "," steps left and "." steps right (the bindings ViZDoom writes to _vizdoom.ini; edit that file to change them).
Nothing else does anything in this scenario.
Each episode waits for Enter, so you can rest between them.

Personal data stays local: demos (.lmp) and results go to assets/data/flydoom/human/, which Git ignores.
Scoring: every demo is REPLAYED at the controllers' resolution (320x240) and scored by flydoom.metrics, exactly like
the controllers. Treat the result as exploratory (one person, five episodes, a first session), not a benchmark.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import vizdoom as vzd

from .game import TICRATE, make_game, replay_lmp
from .metrics import episode_metrics, summarise
from .trials import TEST_SEEDS

ROOT = Path(__file__).resolve().parent.parent
HUMAN_DIR = ROOT / "assets" / "data" / "flydoom" / "human"
HUMAN_SEEDS = TEST_SEEDS[:5]
PRACTICE_SEED = 4242


def play(seed: int, lmp: Path | None) -> float:
    game = make_game(seed, visible=True, spectator=True, resolution=vzd.ScreenResolution.RES_640X480)
    game.new_episode(str(lmp) if lmp else "")
    while not game.is_episode_finished():
        game.advance_action()  # in spectator mode the engine reads the keyboard; this just steps time
    tics = game.get_episode_time()
    game.close()
    return tics / TICRATE


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--practice", action="store_true")
    args = parser.parse_args()
    if args.practice:
        input(f"Practice episode (seed {PRACTICE_SEED}, not saved). Press Enter, then use , (left) and . (right) to dodge... ")
        print(f"  survived {play(PRACTICE_SEED, None):.1f} s")
        return
    HUMAN_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, seed in enumerate(HUMAN_SEEDS, start=1):
        lmp = HUMAN_DIR / f"seed_{seed}.lmp"
        input(f"Episode {i}/{len(HUMAN_SEEDS)} (seed {seed}). Press Enter, then use , (left) and . (right) to dodge... ")
        print(f"  survived {play(seed, lmp):.1f} s (as played)")
        log = replay_lmp(lmp, seed)  # re-rendered at 320x240 and scored like every controller
        log.controller = "human"
        rows.append(episode_metrics(log))
    out = {"recorded": time.strftime("%Y-%m-%d %H:%M"), "seeds": HUMAN_SEEDS, "summary": summarise(rows), "episodes": rows,
           "note": "exploratory: one person, first session, five episodes"}
    (HUMAN_DIR / "results.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"saved {len(rows)} episodes to {HUMAN_DIR} (ignored by Git)")
    print(json.dumps(out["summary"], indent=1))


if __name__ == "__main__":
    main()
