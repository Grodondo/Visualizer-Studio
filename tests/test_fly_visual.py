"""Checks for the FlyPlays video: the footage is the recorded experiment, the story told is in the data, the drawing fits."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from flydoom_fly import FlyPlays

ROOT = Path(__file__).resolve().parent.parent
FLY = ROOT / "assets" / "flydoom"
MEDIA = ROOT / "media"
FRAME_WIDTH, FRAME_HEIGHT = 14.22, 8.0


def read(name):
    return json.loads((FLY / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def results():
    return read("fly_results.json")


@pytest.fixture(scope="module")
def fly_clips():
    return [read(c) for c in read("fly_footage.json")["clips"]]


@pytest.fixture(scope="module")
def shuffle_clip():
    return read(read("shuffle_footage.json")["clips"][0])


def episode(results, controller, seed):
    return next(e for e in results["episodes"] if e["controller"] == controller and e["seed"] == seed)


# ---- the footage is the recorded experiment ------------------------------------------------------------


def test_fly_footage_follows_the_fixed_rule(results, fly_clips):
    seconds = [c["tics"] / c["ticrate"] for c in fly_clips]
    assert [c["seed"] for c in fly_clips] == results["test_seeds"][: len(fly_clips)], "test episodes, in seed order"
    assert sum(seconds) >= 40.0 and sum(seconds[:-1]) < 40.0, "episodes are added until 40 s is reached, and no further"


def test_every_clip_is_its_recorded_episode(results, fly_clips, shuffle_clip):
    pairs = [("fly", c) for c in fly_clips] + [("fly_shuffled", shuffle_clip)]
    for controller, clip in pairs:
        rec = episode(results, controller, clip["seed"])
        assert clip["tics"] == rec["tics"]
        assert clip["metrics"]["frame_hash"] == rec["frame_hash"], "the footage must be the recorded episode, not a new run"
        assert clip["gain"] == results["calibration"]["chosen_gain"]
        for key in ("actions", "eye", "fireballs", "rates_in", "steer_hz"):
            assert len(clip[key]) == clip["tics"]
        assert all(len(v) == clip["tics"] for v in clip["counts"].values())


def test_spike_files_have_one_entry_per_tic(fly_clips, shuffle_clip):
    for name, tics in (("fly_activity.npz", sum(c["tics"] for c in fly_clips)), ("shuffle_activity.npz", shuffle_clip["tics"])):
        z = np.load(FLY / name)
        assert len(z["offsets"]) == tics + 1 and z["offsets"][0] == 0 and z["offsets"][-1] == len(z["idx"])
        assert np.all(np.diff(z["offsets"]) >= 0)


@pytest.mark.skipif(not (MEDIA / "flydoom_fly_activity" / "manifest.json").exists(), reason="spike images not rendered (flydoom.activity_frames)")
def test_rendered_images_match_the_tics(fly_clips, shuffle_clip):
    for folder, tics in (("flydoom_fly_activity", sum(c["tics"] for c in fly_clips)), ("flydoom_shuffle_activity", shuffle_clip["tics"])):
        assert len(list((MEDIA / folder).glob("[0-9]*.png"))) == tics
    for clip in fly_clips + [shuffle_clip]:
        assert len(list((ROOT / clip["frames"]).glob("[0-9]*.png"))) == clip["tics"]


# ---- the story the captions tell -----------------------------------------------------------------------


def test_first_reaction_story(fly_clips):
    s = FlyPlays.read_story(fly_clips[0])  # the checks that the captions depend on are asserted inside
    assert (s.side, s.winner) == ("right", "left")
    assert s.onset <= s.first_spike <= s.act
    assert s.act - s.onset <= 3, "the whole reaction fits in a few 1/35 s steps"


def test_shuffled_wiring_never_reaches_the_steering_neurons(results, shuffle_clip):
    inputs, steering = FlyPlays.spike_totals(shuffle_clip)
    assert inputs > 1000 and steering == 0
    assert set(shuffle_clip["actions"]) == {"none"}


def test_shuffled_equals_idle_on_every_seed(results):
    idle = json.loads((FLY / "results.json").read_text(encoding="utf-8"))["episodes"]
    for seed in results["test_seeds"]:
        shuffled = episode(results, "fly_shuffled", seed)["survival_s"]
        assert shuffled == next(e["survival_s"] for e in idle if e["controller"] == "idle" and e["seed"] == seed)


def test_comparison_claims(results):
    base = json.loads((FLY / "results.json").read_text(encoding="utf-8"))["episodes"]
    seeds = results["test_seeds"]
    surv = {}
    for e in base + results["episodes"]:
        if e["seed"] in seeds:
            surv.setdefault(e["controller"], {})[e["seed"]] = e["survival_s"]
    median = {k: round(float(np.median([v[s] for s in seeds])), 1) for k, v in surv.items()}
    assert (median["fly"], median["reflex"], median["scripted"], median["idle"]) == (8.0, 7.6, 11.3, 4.3)
    assert sum(surv["fly"][s] > surv["reflex"][s] for s in seeds) == 3
    assert sum(surv["scripted"][s] > surv["fly"][s] for s in seeds) == 9


# ---- the drawing -----------------------------------------------------------------------------------------


def stub(**kw):
    return SimpleNamespace(gtic=lambda: 0, eye=np.zeros((2, 4)), steer=np.zeros((2, 2)), actions=["none", "none"], **kw)


def test_controls_fit_between_the_game_and_the_brain():
    box = FlyPlays.controls(stub())
    assert box.get_left()[0] > -FRAME_WIDTH / 2 + 0.3
    assert box.get_right()[0] < -2.5, "must not run into the anatomy image"
    assert box.get_bottom()[1] > -2.6, "must stay above the timeline"


def test_legend_stays_clear_of_the_spikes_and_the_caption():
    box = FlyPlays.legend(stub())
    assert box.get_right()[0] < FRAME_WIDTH / 2 - 0.1 and box.get_top()[1] < FRAME_HEIGHT / 2
    assert box.get_bottom()[1] > 1.5, "the amber input cells sit around y = 1.0 and below"
