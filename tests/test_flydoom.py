"""Checks for the flydoom prototype (Milestone 1). Skipped when the optional 'flydoom' dependency group is not installed:
    uv run --group flydoom pytest -q tests/test_flydoom.py
"""
import json
from pathlib import Path

import numpy as np
import pytest

vzd = pytest.importorskip("vizdoom")

from flydoom import controllers as C  # noqa: E402
from flydoom import metrics  # noqa: E402
from flydoom.eye import BRIGHT_LUMINANCE, LoomingEye  # noqa: E402
from flydoom.game import ACTIONS, TICRATE, make_game, replay_lmp, run_episode  # noqa: E402
from flydoom.trials import CALIBRATION_SEEDS, TEST_SEEDS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "assets" / "flydoom" / "results.json"
SEEDS = ROOT / "assets" / "flydoom" / "seeds.json"
FOOTAGE = ROOT / "assets" / "flydoom" / "footage.json"


# ---- the game is reproducible ------------------------------------------------------------------

def test_same_seed_same_actions_identical_frames():
    a, b = run_episode(C.Random(), 1001), run_episode(C.Random(), 1001)
    assert a.frame_hash == b.frame_hash and a.actions == b.actions and a.health == b.health


def test_different_seeds_differ():
    assert run_episode(C.Idle(), 1001).frame_hash != run_episode(C.Idle(), 1002).frame_hash


def test_lmp_demo_replays_to_the_same_episode(tmp_path):
    lmp = tmp_path / "reflex.lmp"
    played = run_episode(C.Reflex(20.0), 1003, record_lmp=lmp)
    replayed = replay_lmp(lmp, 1003)
    assert replayed.health == played.health
    assert replayed.tics == played.tics and replayed.died == played.died
    assert replayed.actions == played.actions


def test_demo_recorded_at_another_resolution_replays_the_same(tmp_path):
    """Human demos are recorded in a 640x480 window and scored by replaying at 320x240."""
    lmp = tmp_path / "big.lmp"
    game = make_game(1004, resolution=vzd.ScreenResolution.RES_640X480)
    game.new_episode(str(lmp))
    ctl, actions = C.Random(), []
    ctl.reset(1004)
    while not game.is_episode_finished():
        a = ctl.act(game.get_state().screen_buffer)
        actions.append(a)
        game.make_action(ACTIONS[a])
    tics = len(actions)
    game.close()
    replayed = replay_lmp(lmp, 1004)
    assert replayed.tics == tics and replayed.actions == actions


# ---- controllers see pixels only ---------------------------------------------------------------

def test_controllers_receive_only_the_screen(monkeypatch):
    seen = []
    for ctl in C.all_controllers():
        original = ctl.act

        def spy(*args, _orig=original, **kwargs):
            seen.append((len(args), tuple(kwargs)))
            assert isinstance(args[0], np.ndarray) and args[0].ndim == 3 and args[0].shape[2] == 3
            return _orig(*args, **kwargs)

        monkeypatch.setattr(ctl, "act", spy)
        log = run_episode(ctl, 1005)
        assert log.tics > 0
    assert all(s == (1, ()) for s in seen)


def test_reflex_and_fly_pipeline_share_one_eye():
    """C (reflex) decides from LoomingEye only; the fly controllers will use the same class."""
    assert isinstance(C.Reflex().eye, LoomingEye)
    assert set(C.LoomingEye().see(np.zeros((240, 320, 3), np.uint8)).as_list()) == {0.0}


# ---- the eye on synthetic frames -----------------------------------------------------------------

def disc_frame(cx, r, w=320, h=240):
    yy, xx = np.mgrid[:h, :w]
    f = np.full((h, w, 3), 60, np.uint8)
    f[(xx - cx) ** 2 + (yy - h / 2) ** 2 <= r * r] = 255
    return f


def test_eye_reports_growth_on_the_correct_side():
    eye = LoomingEye()
    for r in (2, 4, 6, 8, 10):
        out = eye.see(disc_frame(60, r))
    assert out.expansion_dps[0] > 0 and out.expansion_dps[1] == 0
    assert out.size_deg[0] == pytest.approx(2 * 10 * 90 / 320, rel=0.1)  # diameter 20 px at 0.28 deg/px


def test_eye_ignores_a_static_object_and_dim_pixels():
    eye = LoomingEye()
    for _ in range(12):
        out = eye.see(disc_frame(250, 8))
    assert max(out.expansion_dps) < 1e-3
    dim = np.full((240, 320, 3), int(BRIGHT_LUMINANCE) - 5, np.uint8)
    assert LoomingEye().see(dim).size_deg == (0.0, 0.0)


# ---- metrics ---------------------------------------------------------------------------------------

def test_reaction_time_and_direction_from_ground_truth_events():
    fireballs = [[] for _ in range(40)]
    for t in range(10, 40):
        fireballs[t] = [[7, 80.0, 30]]  # one fireball on the left from tic 10
    actions = ["none"] * 13 + ["right"] * 27  # first press 3 tics later, away from it
    r = metrics.reactions(actions, fireballs)
    assert r["events"] == 1 and r["reaction_s"] == [pytest.approx(3 / TICRATE)] and r["away"] == [True]
    moving = metrics.reactions(["left"] * 40, fireballs)
    assert moving["already_moving"] == 1 and moving["reaction_s"] == []


def test_calibration_and_test_seeds_are_disjoint():
    assert not set(CALIBRATION_SEEDS) & set(TEST_SEEDS)


def test_seed_sets_are_distinct_episodes():
    """Many ViZDoom seeds give the same take_cover game; the screened seeds must not."""
    s = json.loads(SEEDS.read_text(encoding="utf-8"))
    hashes = s["idle_episode_sha256"]
    assert len(set(hashes.values())) == len(hashes) == len(CALIBRATION_SEEDS) + len(TEST_SEEDS)
    for seed in (TEST_SEEDS[0], TEST_SEEDS[-1], CALIBRATION_SEEDS[-1]):  # spot-check: still the same episode today
        assert run_episode(C.Idle(), seed).frame_hash == hashes[str(seed)]


def test_known_duplicate_seeds_are_excluded():
    """1007 and 1008 were in the first test set and turned out to be the same game."""
    assert run_episode(C.Idle(), 1007).frame_hash == run_episode(C.Idle(), 1008).frame_hash
    assert not {1007, 1008, 1010} & set(TEST_SEEDS)


# ---- the committed results -------------------------------------------------------------------------

@pytest.mark.skipif(not RESULTS.exists(), reason="run uv run --group flydoom python -m flydoom.trials")
def test_results_file():
    r = json.loads(RESULTS.read_text(encoding="utf-8"))
    assert r["vizdoom"] == vzd.__version__ == "1.3.1"
    assert r["reproducibility"]["mismatches"] == [] and r["reproducibility"]["episodes_run_twice"] == len(r["episodes"])
    assert r["test_seeds"] == TEST_SEEDS
    assert all(n == len(TEST_SEEDS) for n in r["distinct_episodes"].values())  # every controller: 10 different games
    for name, c in r["controllers"].items():
        rows = [e for e in r["episodes"] if e["controller"] == name]
        assert [e["seed"] for e in rows] == r["test_seeds"]
        assert c["summary"]["survival_s_q"][1] == pytest.approx(float(np.median([e["survival_s"] for e in rows])), abs=1e-3)
    by = {(e["controller"], e["seed"]): e["frame_hash"] for e in r["episodes"]}
    # recorded finding: the first scripted AI behaved like "always step left" (same survival on every test seed)
    surv = {(e["controller"], e["seed"]): e["survival_s"] for e in r["episodes"]}
    assert all(surv[("scripted_v1", s)] == surv[("constant", s)] for s in r["test_seeds"])


@pytest.mark.skipif(not FOOTAGE.exists(), reason="run uv run --group flydoom python -m flydoom.export")
def test_footage_follows_the_rule_and_matches_fresh_runs():
    """Footage = the reflex's test episodes in seed order until >= 40 s: no selection by outcome, no repeats."""
    from flydoom.export import CLIP_MIN_SECONDS

    f = json.loads(FOOTAGE.read_text(encoding="utf-8"))
    r = json.loads(RESULTS.read_text(encoding="utf-8"))
    n = len(f["seeds"])
    assert f["seeds"] == r["test_seeds"][:n] and len(set(f["seeds"])) == n
    clips = [json.loads((ROOT / "assets" / "flydoom" / name).read_text(encoding="utf-8")) for name in f["clips"]]
    total = sum(c["tics"] for c in clips) / TICRATE
    assert total >= CLIP_MIN_SECONDS and total - clips[-1]["tics"] / TICRATE < CLIP_MIN_SECONDS
    for clip in clips:
        fresh = run_episode(C.Reflex(r["reflex_calibration"]["chosen"]), clip["seed"])
        assert fresh.actions == clip["actions"] and fresh.health == clip["health"]
        assert len(clip["eye"]) == len(clip["fireballs"]) == clip["tics"]
