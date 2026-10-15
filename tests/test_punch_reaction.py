"""Checks for the punch-reaction video: the export is self-consistent, and every sentence the video says about the
data is true for the exported numbers (if one fails after an analysis change, change the sentence, not the test)."""
import json

import numpy as np
import pytest

from compute import brain_render, punch_reaction
from studio.punch import FRAMES_DIR, PunchTrace

VISUAL_OR_PARIETAL = {"lateraloccipital", "pericalcarine", "cuneus", "lingual", "inferiorparietal", "superiorparietal", "precuneus"}


@pytest.fixture(scope="module")
def trace() -> PunchTrace:
    return PunchTrace.load(punch_reaction.TRACE_PATH)


def test_dataset_and_licence(trace):
    d = trace["dataset"]
    assert d["license"] == "CC BY-SA 4.0" and d["doi"] == "10.1016/j.neuroimage.2020.117465"
    assert d["inverse"]["method"] == "dSPM"


def test_trials_and_reaction_times(trace):
    t, rt = trace["trials"], trace["rt"]
    assert t["total"] == 400 and 0 < t["counted"] <= t["correct"] <= t["total"]
    assert len(rt["values_s"]) == t["counted"]
    assert all(punch_reaction.RT_RANGE[0] <= v <= punch_reaction.RT_RANGE[1] for v in rt["values_s"])
    assert rt["median_s"] == pytest.approx(float(np.median(rt["values_s"])), abs=1e-3)
    assert rt["compatible_median_s"] < rt["incompatible_median_s"]  # "when the outer arrows point the other way, slower"
    assert t["stim_epochs_kept"] / t["counted"] > 0.85


def test_claim_visual_waves(trace):
    p1, n1 = trace["visual"]["p1"], trace["visual"]["n1"]
    assert p1["amplitude_uv"] > 0 and n1["amplitude_uv"] < 0
    assert 0.10 <= p1["latency_s"] <= 0.14 and 0.18 <= n1["latency_s"] <= 0.22  # "about 130 ms", "about 200 ms"
    assert abs(n1["amplitude_uv"]) > abs(p1["amplitude_uv"])  # "then a larger wave"


def test_claim_motor_preparation(trace):
    times, uv = trace.lrp()
    before = times <= 0
    assert uv[before].min() < -2.0  # a clear negative LRP before the press
    assert -0.2 < trace["lrp_onset_s"] < -0.05
    assert times[np.argmin(np.where(before, uv, np.inf))] < 0


def test_claim_where_visual_then_parietal_strongest_on_right(trace):
    trusted = trace["trusted_until_s"]
    for step in trace["where"]["stim"]:
        if 0.12 <= step["t"] <= 0.20:
            top = step["top_labels"][0][0]
            assert top.rsplit("-", 1)[0] in VISUAL_OR_PARIETAL, step
    right = [s for s in trace["where"]["stim"] if s["t"] in (0.18, 0.2)]
    assert all(s["top_labels"][0][0].endswith("-rh") for s in right)  # "strongest on the right"
    after = [s for s in trace["where"]["stim"] if s["t"] > trusted + 1e-9]
    assert after and all(s["top_labels"][0][0] == "unknown" or "cingulate" in s["top_labels"][0][0] for s in after)  # why we stop


def test_claim_hand_contrast_smears_over_both_halves(trace):
    split = trace["where"]["hand_contrast_central_mean"]
    assert 0.8 < split["lh"] / split["rh"] < 1.25


def test_electrodes(trace):
    e = trace["electrodes"]
    assert len(e) == 30 and e["C3"][0] < -0.3 and e["C4"][0] > 0.3 and e["Oz"][1] < -0.8
    assert np.hypot(*e["Cz"]) < 0.05


def test_sources_export():
    s = np.load(punch_reaction.SOURCES_PATH)
    assert s["stim_data"].shape == (len(s["lh_vertno"]) + len(s["rh_vertno"]), len(s["stim_times"]))
    assert s["stim_times"][0] == pytest.approx(punch_reaction.SOURCE_WINDOW[0], abs=0.005)
    lo, hi = s["stim_limits"]
    assert 0 < lo < hi
    assert 0 < s["limit_limits"][0] < s["limit_limits"][1]


RENDERED = (FRAMES_DIR / "manifest.json").exists()


@pytest.mark.skipif(not RENDERED, reason="brain frames not rendered (uv run python -m compute.brain_render)")
def test_rendered_frames_match_the_export():
    m = json.loads((FRAMES_DIR / "manifest.json").read_text(encoding="utf-8"))
    s = np.load(punch_reaction.SOURCES_PATH)
    assert np.allclose(m["activity"]["times"], s["stim_times"], atol=1e-4)
    assert [st["color"] for st in m["stages"]] == [st["color"] for st in brain_render.STAGES]
    for view in m["activity"]["views"]:
        for i in range(len(m["activity"]["times"])):
            assert (FRAMES_DIR / "activity" / f"{view}_{i:03d}.png").exists()
    for view in m["anatomy"]["views"]:
        for k in range(len(m["stages"]) + 1):
            assert (FRAMES_DIR / "anatomy" / f"{view}_{k}.png").exists()


def test_stage_atlas_labels_exist():
    if not (punch_reaction.SUBJECTS_DIR / "fsaverage" / "label" / "lh.aparc.annot").exists():
        pytest.skip("fsaverage not downloaded")
    import mne

    names = {l.name for l in mne.read_labels_from_annot("fsaverage", "aparc", subjects_dir=punch_reaction.SUBJECTS_DIR, verbose=False)}
    for stage in brain_render.STAGES:
        assert set(stage["labels"]) <= names
