"""Checks for Milestone 3 (the fly model closing the loop with the game).
    uv run --group flydoom pytest -q tests/test_closed_loop.py
"""
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("numba")
pytest.importorskip("vizdoom")

from flydoom import lif  # noqa: E402
from flydoom import model_data as md  # noqa: E402
from flydoom.fly_controller import FlyBrain, connectome, groups  # noqa: E402
from flydoom.seeds import load as load_seeds  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FLY = ROOT / "assets" / "flydoom" / "fly_results.json"


def test_stepping_brain_equals_the_validated_kernel():
    c = md.load("783")
    stim = groups(c)[("LC4", "right")]
    one, _ = lif.simulate(c, {100.0: stim}, t_run_s=0.2, trials=1, seed=11)
    b = lif.Brain(c, stim, seed=11)
    total = sum(b.advance(1000 / 35, np.full(len(stim), 100.0))[0] for _ in range(7))  # 7 tics = 200 ms
    assert b.t == 2000 and np.array_equal(total, one.counts[0])


def expanding_disc(cx, frames=30, w=320, h=240):
    yy, xx = np.mgrid[:h, :w]
    for k in range(frames):
        f = np.full((h, w, 3), 60, np.uint8)
        f[(xx - cx) ** 2 + (yy - h / 2) ** 2 <= (2 + 1.5 * k) ** 2] = 255
        yield f


@pytest.mark.parametrize("cx, away", [(250, "left"), (70, "right")])
def test_fly_steps_away_from_a_looming_object(cx, away):
    """End to end through the real model: object looms in one half of the screen -> the fly steps to the other side."""
    ctl = FlyBrain(gain=2.0)
    ctl.reset(5)
    acts = [ctl.act(f) for f in expanding_disc(cx)]
    toward = "right" if away == "left" else "left"
    assert acts.count(away) > 0 and acts.count(toward) == 0
    side = "right" if cx > 160 else "left"
    assert sum(ctl.log["counts"][f"LC4_{side}"]) > 0  # the stimulated side's input cells fired


def test_nothing_on_screen_means_no_input_and_no_movement():
    ctl = FlyBrain(gain=2.0)
    ctl.reset(5)
    acts = [ctl.act(np.full((240, 320, 3), 60, np.uint8)) for _ in range(20)]
    assert set(acts) == {"none"} and sum(ctl.log["active"]) == 0


def test_shuffled_control_keeps_neurons_and_degrees():
    real, null = connectome("real"), connectome("shuffled")
    assert np.array_equal(real.flywire_ids, null.flywire_ids) and np.array_equal(real.indptr, null.indptr)
    assert np.array_equal(np.bincount(real.indices, minlength=real.n), np.bincount(null.indices, minlength=null.n))


@pytest.mark.skipif(not FLY.exists(), reason="run uv run --group flydoom python -m flydoom.fly_trials")
def test_fly_results_file():
    r = json.loads(FLY.read_text(encoding="utf-8"))
    cal, test = load_seeds()
    assert r["calibration"]["seeds"] == cal and r["test_seeds"] == test
    assert r["calibration"]["chosen_gain"] in [float(g) for g in r["calibration"]["grid"]]
    assert r["reproducible_rerun"]["identical"]
    assert all(n == len(test) for n in r["distinct_episodes"].values())
    for name in ("fly", "fly_shuffled", "fly_no_giant_fiber"):
        rows = [e for e in r["episodes"] if e["controller"] == name]
        assert [e["seed"] for e in rows] == test


@pytest.mark.skipif(not FLY.exists(), reason="run uv run --group flydoom python -m flydoom.fly_trials")
def test_what_the_experiment_showed_and_did_not():
    """The sentences a video may say about Milestone 3. If a re-run changes them, change the sentence, not the test."""
    r = json.loads(FLY.read_text(encoding="utf-8"))
    base = json.loads((ROOT / "assets" / "flydoom" / "results.json").read_text(encoding="utf-8"))
    surv = {}
    for e in base["episodes"] + r["episodes"]:
        surv.setdefault(e["controller"], {})[e["seed"]] = e["survival_s"]
    seeds = r["test_seeds"]
    wins = lambda a, b: sum(surv[a][s] > surv[b][s] for s in seeds)  # noqa: E731
    # "with shuffled wiring the same input never reaches the steering neurons: the fly behaves exactly like idle"
    assert all(surv["fly_shuffled"][s] == surv["idle"][s] for s in seeds)
    assert r["controllers"]["fly_shuffled"]["summary"]["move_fraction_q"][2] == 0
    # "the real wiring does turn looming into steering away from it"
    assert r["controllers"]["fly"]["summary"]["away_fraction"] > 0.7
    assert wins("fly", "idle") >= 7
    # "it is NOT better than the brainless reflex that gets the same eye signal" (no more than half the seeds)
    assert wins("fly", "reflex") <= len(seeds) // 2
    # "and the scripted AI, which gets more information, beats it on most seeds"
    assert wins("scripted", "fly") >= 7
    # "silencing the giant fibers does not hurt" (the decoder does not read them)
    assert wins("fly", "fly_no_giant_fiber") <= len(seeds) // 2
