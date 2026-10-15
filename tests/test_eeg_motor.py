"""Checks for the EEG movement/imagery video: the export is self-consistent, still matches the analysis code,
and every sentence the video says about the data is true for the exported numbers."""
import numpy as np
import pytest
from manim import ManimColor

from compute import eeg_motor
from studio.eeg import POWER_VMAX, EegTrace, HeadMap, color_bar, power_color
from studio.theme import NEG, POS

FRAME_WIDTH = 14.22


@pytest.fixture(scope="module")
def trace() -> EegTrace:
    return EegTrace.load(eeg_motor.TRACE_PATH)


# ---- the data file -------------------------------------------------------------------------------

def test_dataset_and_processing_are_recorded(trace):
    d, p = trace.raw["dataset"], trace.raw["processing"]
    assert d["doi"] == "10.13026/C28G6P" and d["license"] == "ODC-By 1.0"
    assert d["subjects"] == eeg_motor.SUBJECTS and d["runs"] == eeg_motor.RUNS
    assert p["bands_hz"]["mu"] == [8.0, 13.0] and p["baseline_s"] == [-1.0, 0.0]
    assert "median" in p["reject"]["rule"]


def test_trial_counts(trace):
    rows = trace.raw["trial_counts"]
    assert [r["subject"] for r in rows] == eeg_motor.SUBJECTS
    for r in rows:
        for cond in ("executed", "imagined"):
            for hand in ("left", "right"):
                kept, total = r[cond]["kept"][hand], r[cond]["total"][hand]
                assert 0 < kept <= total
                assert kept / total >= 0.7, (r["subject"], cond, hand)  # no volunteer loses most of a condition
    kept, total = trace.trial_counts()
    assert kept / total > 0.9


def test_shapes(trace):
    n_t, n_f = len(trace.times), len(trace.freqs)
    assert trace.times[0] == pytest.approx(-1.0) and trace.times[-1] == pytest.approx(4.0, abs=0.06)
    for cond in ("executed", "imagined"):
        c = trace.condition(cond)
        assert np.array(c["tfr_contra"]).shape == (n_f, n_t)
        for band in ("mu", "beta"):
            assert len(c["courses"][band]["contra_mean"]) == n_t
            assert len(c["courses"][band]["contra_window_per_subject"]) == trace.n_subjects
            assert set(c["topo"][band]["right"]) == set(trace.electrodes)


def test_course_and_window_numbers_agree(trace):
    """The per-volunteer window numbers and the averaged curve were computed separately; they must tell the same story."""
    t = trace.times
    inside = (t >= 0.5) & (t <= 3.5)
    for cond in ("executed", "imagined"):
        mean, _ = trace.course(cond, "mu", "contra")
        assert mean[inside].mean() == pytest.approx(trace.window_mean(cond, "mu"), abs=1.0)


def test_electrode_layout(trace):
    e = trace.electrodes
    assert len(e) == 64
    assert np.hypot(*e["Cz"]) < 0.05
    assert e["C3"][0] < -0.3 and e["C4"][0] > 0.3  # left hemisphere on the left (seen from above, nose up)
    assert e["Fpz"][1] > 0.9 and e["Oz"][1] < -0.9
    for ring in ("Fpz", "T7", "Oz", "T8"):
        assert np.hypot(*e[ring]) == pytest.approx(1.0, abs=0.08)


def test_example_recording(trace):
    ex = trace.example
    assert ex["subject"] == eeg_motor.EXAMPLE_SUBJECT and ex["channels"] == ["C3", "C4"]
    n = int(round((eeg_motor.EXAMPLE_WINDOW[1] - eeg_motor.EXAMPLE_WINDOW[0]) * ex["sfreq"]))
    for key in ("raw_uv", "mu_uv", "mu_envelope_uv"):
        assert np.array(ex[key]).shape == (2, n)
    assert np.all(np.array(ex["mu_envelope_uv"]) >= 0)
    labels = [c["label"] for c in ex["cues"]]
    assert labels[:4] == ["T0", "T2", "T0", "T1"]  # rest, right fist, rest, left fist: what beat "raw" labels


DATA_PRESENT = all((eeg_motor.LOCAL_FILES / f"S001/S001R{r:02d}.edf").exists() for r in (3, 4, 7, 8, 11, 12))


@pytest.mark.skipif(not DATA_PRESENT, reason="EEG files not downloaded (uv run python -m compute.eeg_motor)")
def test_export_is_up_to_date_for_volunteer_1(trace):
    """Re-run the analysis for volunteer 1 and compare with the committed export (catches 'changed code, forgot to export')."""
    result = eeg_motor.analyse_subject(1)
    row = trace.raw["trial_counts"][0]
    for cond in ("executed", "imagined"):
        assert result[cond]["n_kept"] == row[cond]["kept"]
        tfr = result[cond]["tfr"]["right"]
        lo, hi = eeg_motor.BANDS["mu"]
        rows = (tfr.freqs >= lo) & (tfr.freqs <= hi)
        cols = (tfr.times >= 0.5) & (tfr.times <= 3.5)
        right_c3 = tfr.data[tfr.ch_names.index("C3")][rows][:, cols].mean()
        tfr_l = result[cond]["tfr"]["left"]
        left_c4 = tfr_l.data[tfr_l.ch_names.index("C4")][rows][:, cols].mean()
        exported = trace.window_values(cond, "mu", "contra")[0]
        assert (right_c3 + left_c4) / 2 == pytest.approx(exported, abs=0.06)


# ---- what the video says ------------------------------------------------------------------------------

def test_claim_rhythm_weakens_more_when_moving_than_imagining(trace):
    e, i = trace.window_mean("executed", "mu"), trace.window_mean("imagined", "mu")
    assert e < 0 and i < 0 and abs(i) < abs(e)
    assert trace.n_with_drop("executed", "mu") > trace.n_subjects / 2
    assert trace.n_with_drop("imagined", "mu") > trace.n_subjects / 2


def test_claim_within_half_a_second(trace):
    for cond in ("executed", "imagined"):
        assert 0.35 < trace.condition(cond)["drop_time"] < 0.5


def test_claim_blip_is_beside_the_eyes_and_slow_waves_are_frontal(trace):
    beside_eyes = {"F7", "F8", "FT7", "FT8", "AF7", "AF8", "T9", "T10"}
    frontal = {"Fp1", "Fpz", "Fp2", "AF3", "AFz", "AF4", "AF7", "AF8"}
    for cond in ("executed", "imagined"):
        assert set(trace.strongest_channels(cond, "burst", 2)) <= beside_eyes
        assert set(trace.strongest_channels(cond, "slow", 3)) <= frontal


def test_claim_lateralization(trace):
    """'Real movement: opposite side larger in only k of n', 'Imagery: opposite side larger in k of n' (a majority)."""
    n = trace.n_subjects
    assert trace.n_contra_stronger("executed", "mu") <= n / 2
    assert trace.n_contra_stronger("imagined", "mu") > n / 2


# ---- drawing --------------------------------------------------------------------------------------------

def test_power_color_follows_the_sign_rule():
    assert power_color(-POWER_VMAX) == ManimColor(NEG)
    assert power_color(POWER_VMAX * 2) == ManimColor(POS)
    assert power_color(-1) != power_color(1)


def test_head_and_key_fit(trace):
    head = HeadMap(trace.electrodes, radius=1.55, values=trace.topo("executed", "mu", "right"))
    assert len(head.dots) == 64
    assert head.width < 1.55 * 2 * 1.4  # T9/T10 sit below the ear line, at about 1.3 head radii
    assert color_bar().width < FRAME_WIDTH / 2
