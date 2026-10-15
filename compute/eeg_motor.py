"""Real EEG during real and imagined fist clenching, exported for the Manim video "Can we see the brain preparing to move?".

Dataset: EEG Motor Movement/Imagery Dataset v1.0.0 (PhysioNet), recorded with BCI2000.
  Schalk G. et al. (2004) BCI2000: a general-purpose brain-computer interface system. IEEE TBME 51(6):1034-1043.
  Schalk G. (2009) EEG Motor Movement/Imagery Dataset (v1.0.0). PhysioNet. https://doi.org/10.13026/C28G6P
  Licence: Open Data Commons Attribution License v1.0 (ODC-By). 109 volunteers, 64 EEG channels, 160 samples/s.

What the volunteers did (runs used here):
  runs 3, 7, 11  a target appears on the left or right of a screen; open and close that fist until it disappears
  runs 4, 8, 12  the same, but only IMAGINE opening and closing the fist
  Annotations: T0 = rest, T1 = left-fist target, T2 = right-fist target. They mark when the target (cue) appeared and
  how long it stayed (4.1 s). There is no EMG or movement sensor, so the moment the hand actually moved is NOT recorded.

Run from the project root:  uv run python -m compute.eeg_motor
  Files are fetched from PhysioNet's open-data mirror on AWS (byte-identical to physionet.org, which is very slow),
  stored in MNE's normal folder layout under assets/data/eegbci/ (not tracked by Git), and then opened through
  mne.datasets.eegbci.load_data, which checks every file against MNE's SHA-256 registry before use.
  Output: assets/eeg_motor/eeg_motor_trace.json, the only file the video reads.

Every analysis choice is fixed below and was chosen before looking at the results.
"""
from __future__ import annotations

import json
import time
import urllib.request
from importlib.resources import files
from pathlib import Path

import mne
import numpy as np
from mne.datasets import eegbci

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "assets" / "data" / "eegbci"
LOCAL_FILES = DATA_DIR / "MNE-eegbci-data" / "files" / "eegmmidb" / "1.0.0"
TRACE_PATH = ROOT / "assets" / "eeg_motor" / "eeg_motor_trace.json"
MIRROR_URL = "https://physionet-open.s3.amazonaws.com/eegmmidb/1.0.0/"

# ---- which data -------------------------------------------------------------------------------
# The first 20 volunteers. (Subjects 88, 92 and 100 were recorded at a different rate and 104 is incomplete;
# none of them is in this range.) No volunteer is dropped for "not showing the effect".
SUBJECTS = list(range(1, 21))
RUNS = {"executed": [3, 7, 11], "imagined": [4, 8, 12]}
EXAMPLE_SUBJECT = 1  # the raw traces in the video come from the first volunteer, not a hand-picked one
EXAMPLE_RUN = 3  # executed left/right fist, first run
EXAMPLE_WINDOW = (0.0, 30.0)  # seconds of that run exported for the scrolling raw trace

# ---- processing (fixed in advance) ---------------------------------------------------------------
MONTAGE = "colin27_1005"  # standard 10-05 electrode positions (MNE's new name for "standard_1005")
FILTER = (1.0, 40.0)  # Hz, removes slow drift and line noise; keeps mu and beta
EPOCH = (-1.5, 4.5)  # s around the cue; 0.5 s padding on each side is cropped away after the wavelet transform
SHOW = (-1.0, 4.0)  # s, the part that is analysed and shown
BASELINE = (-1.0, 0.0)  # s of rest before the cue; power is expressed as % change from it
REJECT_CHANNELS = ["FC3", "FC1", "FCz", "FC2", "FC4", "C3", "C1", "Cz", "C2", "C4", "CP3", "CP1", "CPz", "CP2", "CP4"]
# An epoch is dropped if its largest central peak-to-peak swing is an outlier FOR THAT VOLUNTEER:
# above median + REJECT_MADS x MAD (MAD scaled to match a standard deviation). Why not a fixed microvolt limit:
# a first try with 150 uV dropped every trial of volunteer 3 and half of volunteer 1, because this recording's typical
# amplitudes (median central swing 157-374 uV in volunteers 1-3) are far larger than such a limit assumes.
# That change was made before any power-change result was looked at.
REJECT_MADS = 3.0
FREQS = np.arange(4.0, 36.0, 1.0)  # Hz, rows of the time-frequency map
N_CYCLES = FREQS / 2.0  # wavelet length = half a second at every frequency
DECIM = 4  # 160 Hz -> 40 time points per second in the time-frequency output
BANDS = {"mu": (8.0, 13.0), "beta": (13.0, 30.0)}  # Hz, conventional sensorimotor bands (Pfurtscheller & Lopes da Silva 1999)
ERD_WINDOW = (0.5, 3.5)  # s after the cue, used for the single numbers and the scalp maps
TIME_STEP = 2  # keep every 2nd TFR time point in the export (20 per second)

# Checks on things that are NOT motor-cortex rhythms (see "artifact_checks" in the export):
BURST = {"freqs": (4.0, 13.0), "times": (0.15, 0.35)}  # brief power rise right after the cue
SLOW = {"freqs": (4.0, 7.0), "times": (1.0, 3.5)}  # slow-wave power during the task
DROP_AFTER = 0.35  # s; the "drop time" is searched after the brief post-cue burst

CONTRA = {"left": "C4", "right": "C3"}  # the hemisphere opposite to the hand
IPSI = {"left": "C3", "right": "C4"}
EVENT_ID = {"T1": 1, "T2": 2}  # T1 = left fist target, T2 = right fist target
HAND = {1: "left", 2: "right"}


# ---- data access ---------------------------------------------------------------------------------

def _registry() -> set[str]:
    text = files("mne").joinpath("data", "eegbci_checksums.txt").read_text()
    return {line.split()[0] for line in text.splitlines() if line.strip()}


def prefetch(subjects, runs) -> None:
    """Download missing files from the mirror into MNE's folder layout (MNE verifies them afterwards)."""
    known = _registry()
    for s in subjects:
        for r in runs:
            name = f"S{s:03d}/S{s:03d}R{r:02d}.edf"
            target = LOCAL_FILES / name
            if target.exists():
                continue
            assert name in known, f"{name} is not in MNE's checksum registry"
            target.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(MIRROR_URL + name, timeout=120) as response:  # noqa: S310 (fixed https URL)
                data = response.read()
            target.with_suffix(".part").write_bytes(data)
            target.with_suffix(".part").replace(target)


def load_raw(subject: int, runs: list[int]) -> mne.io.BaseRaw:
    """Concatenate the runs of one volunteer; standard names, positions, average reference, 1-40 Hz."""
    paths = eegbci.load_data(subjects=subject, runs=runs, path=str(DATA_DIR), update_path=False, verbose=False)
    raws = [mne.io.read_raw_edf(p, preload=True, verbose=False) for p in paths]
    raw = mne.concatenate_raws(raws, verbose=False)
    eegbci.standardize(raw)
    raw.set_montage(mne.channels.make_standard_montage(MONTAGE), verbose=False)
    raw.set_eeg_reference("average", projection=False, verbose=False)
    raw.filter(*FILTER, verbose=False)
    return raw


def make_epochs(raw: mne.io.BaseRaw) -> tuple[mne.Epochs, dict]:
    """Cue-locked epochs with noisy ones dropped, and how many cues of each hand there were before dropping."""
    events, _ = mne.events_from_annotations(raw, event_id=EVENT_ID, verbose=False)
    totals = {hand: int((events[:, 2] == code).sum()) for code, hand in HAND.items()}
    epochs = mne.Epochs(raw, events, EVENT_ID, tmin=EPOCH[0], tmax=EPOCH[1], baseline=None, preload=True, verbose=False)
    # peak-to-peak rejection on the central strip only: frontal eye blinks do not matter for motor channels
    swing = np.ptp(epochs.get_data(picks=REJECT_CHANNELS) * 1e6, axis=2).max(axis=1)
    keep = swing <= outlier_limit(swing)
    epochs.drop(np.flatnonzero(~keep), reason="central peak-to-peak outlier", verbose=False)
    return epochs, totals


def outlier_limit(values: np.ndarray) -> float:
    """median + REJECT_MADS robust standard deviations (1.4826 x median absolute deviation)."""
    med = float(np.median(values))
    return med + REJECT_MADS * 1.4826 * float(np.median(np.abs(values - med)))


# ---- analysis -------------------------------------------------------------------------------------

def power_change(epochs: mne.Epochs) -> mne.time_frequency.AverageTFR:
    """Trial-averaged power, % change from the pre-cue rest (negative = the rhythm got weaker)."""
    tfr = epochs.compute_tfr("morlet", FREQS, n_cycles=N_CYCLES, use_fft=True, return_itc=False, average=True,
                             decim=DECIM, verbose=False)
    tfr.crop(*SHOW)
    tfr.apply_baseline(BASELINE, mode="percent", verbose=False)
    tfr.data *= 100.0
    return tfr


def window_mean(times: np.ndarray, values: np.ndarray) -> float:
    inside = (times >= ERD_WINDOW[0]) & (times <= ERD_WINDOW[1])
    return float(values[..., inside].mean(axis=-1)) if values.ndim == 1 else values[..., inside].mean(axis=-1)


def analyse_subject(subject: int) -> dict:
    """Per condition and hand: the power-change map, plus how many cues were kept out of how many."""
    out = {}
    for condition, runs in RUNS.items():
        epochs, totals = make_epochs(load_raw(subject, runs))
        out[condition] = {"n_kept": {}, "n_total": totals, "tfr": {}}
        for code, hand in HAND.items():
            out[condition]["n_kept"][hand] = len(epochs[f"T{code}"])
            out[condition]["tfr"][hand] = power_change(epochs[f"T{code}"])
    return out


# ---- the example recording --------------------------------------------------------------------------

def example_trace() -> dict:
    """A 30-second stretch of the first volunteer's first executed run: raw C3/C4, the mu-band signal and its envelope."""
    raw = load_raw(EXAMPLE_SUBJECT, [EXAMPLE_RUN])
    t0, t1 = EXAMPLE_WINDOW
    mu = raw.copy().filter(*BANDS["mu"], verbose=False)
    envelope = mu.copy().apply_hilbert(envelope=True, verbose=False)
    picks = ["C3", "C4"]
    start, stop = raw.time_as_index([t0, t1])
    cues = [{"label": a["description"], "onset": round(float(a["onset"]), 4), "duration": round(float(a["duration"]), 4)}
            for a in raw.annotations if a["onset"] < t1 and a["onset"] + a["duration"] > t0]
    uv = lambda r: np.round(r.get_data(picks=picks, start=start, stop=stop) * 1e6, 2)  # noqa: E731
    return {
        "subject": EXAMPLE_SUBJECT, "run": EXAMPLE_RUN, "sfreq": raw.info["sfreq"], "t0": t0,
        "channels": picks, "raw_uv": uv(raw).tolist(), "mu_uv": uv(mu).tolist(), "mu_envelope_uv": uv(envelope).tolist(),
        "cues": cues,
    }


# ---- electrode layout --------------------------------------------------------------------------------

def electrode_layout(info: mne.Info) -> dict:
    """Flat 2D electrode positions: fit a sphere to the 3D positions, then unroll it from the top (azimuthal equidistant).

    The result is in units of the head radius: Cz is near (0, 0), the ring through Fpz, T7, Oz and T8 is near radius 1,
    nose up (+y), right ear to the right (+x).
    """
    pos = np.array([info["chs"][i]["loc"][:3] for i in range(info["nchan"])])
    # least-squares sphere |p - c|^2 = r^2  ->  2 p.c + (r^2 - |c|^2) = |p|^2, linear in (c, k)
    a = np.c_[2 * pos, np.ones(len(pos))]
    sol, *_ = np.linalg.lstsq(a, (pos**2).sum(axis=1), rcond=None)
    center = sol[:3]
    v = pos - center
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    # the head's own axes, from the electrodes: up = toward Cz, right = from T7 toward T8, front = up x right
    at = {name: v[i] for i, name in enumerate(info.ch_names)}
    up = at["Cz"] / np.linalg.norm(at["Cz"])
    right = at["T8"] - at["T7"]
    right -= right.dot(up) * up
    right /= np.linalg.norm(right)
    front = np.cross(up, right)
    theta = np.arccos(np.clip(v @ up, -1, 1))  # angle from the top of the head
    phi = np.arctan2(v @ front, v @ right)
    ring = np.mean([theta[info.ch_names.index(c)] for c in ("Fpz", "T7", "Oz", "T8")])
    xy = np.c_[np.cos(phi), np.sin(phi)] * (theta / ring)[:, None]
    return {name: [round(float(x), 4), round(float(y), 4)] for name, (x, y) in zip(info.ch_names, xy)}


# ---- export -------------------------------------------------------------------------------------------

def grand_average(per_subject: list[dict]) -> dict:
    """Average the % power change across volunteers, keeping both hands apart and as contra/ipsi."""
    first = per_subject[0]["executed"]["tfr"]["right"]
    times, freqs, ch_names = first.times, first.freqs, first.ch_names
    stack = lambda cond, hand: np.stack([s[cond]["tfr"][hand].data for s in per_subject])  # noqa: E731  subj x ch x f x t
    result = {"times": times, "freqs": freqs, "ch_names": ch_names, "conditions": {}}
    for cond in RUNS:
        maps = {hand: stack(cond, hand) for hand in ("left", "right")}
        entry = {"courses": {}, "tfr_contra": None, "topo": {}, "per_subject": []}
        for band in BANDS:
            lo, hi = BANDS[band]
            rows = (freqs >= lo) & (freqs <= hi)
            band_of = lambda hand, ch: maps[hand][:, ch_names.index(ch)][:, rows].mean(axis=1)  # noqa: E731  subj x t
            contra = (band_of("left", CONTRA["left"]) + band_of("right", CONTRA["right"])) / 2
            ipsi = (band_of("left", IPSI["left"]) + band_of("right", IPSI["right"])) / 2
            entry["courses"][band] = {
                "contra_mean": contra.mean(axis=0), "ipsi_mean": ipsi.mean(axis=0),
                "contra_sem": contra.std(axis=0, ddof=1) / np.sqrt(len(contra)),
                "ipsi_sem": ipsi.std(axis=0, ddof=1) / np.sqrt(len(ipsi)),
                "contra_window_per_subject": window_mean(times, contra),
                "ipsi_window_per_subject": window_mean(times, ipsi),
            }
            # scalp map: mean change per electrode in the window, per hand
            for hand in ("left", "right"):
                values = maps[hand][:, :, rows].mean(axis=2).mean(axis=0)  # ch x t
                entry["topo"].setdefault(band, {})[hand] = window_mean(times, values)
        # per-electrode checks, both hands pooled
        def region(spec):
            rows = (freqs >= spec["freqs"][0]) & (freqs <= spec["freqs"][1])
            cols = (times >= spec["times"][0]) & (times <= spec["times"][1])
            both = (maps["left"] + maps["right"]) / 2  # subj x ch x f x t
            return both[:, :, rows][:, :, :, cols].mean(axis=(0, 2, 3))
        entry["burst_by_channel"], entry["slow_by_channel"] = region(BURST), region(SLOW)
        mu = entry["courses"]["mu"]["contra_mean"]
        half = mu[(times >= ERD_WINDOW[0]) & (times <= ERD_WINDOW[1])].mean() / 2
        after = np.flatnonzero((times > DROP_AFTER) & (mu < half))
        entry["drop_time"] = float(times[after[0]]) if len(after) else None
        # time-frequency map at the contralateral electrode, both hands pooled
        contra_map = (maps["left"][:, ch_names.index("C4")] + maps["right"][:, ch_names.index("C3")]) / 2
        entry["tfr_contra"] = contra_map.mean(axis=0)
        result["conditions"][cond] = entry
    return result


def _r(a, nd=2):
    return np.round(np.asarray(a, float), nd).tolist()


def build_trace(subjects=SUBJECTS) -> dict:
    prefetch(subjects, RUNS["executed"] + RUNS["imagined"])
    per_subject, counts = [], []
    for s in subjects:
        t = time.time()
        result = analyse_subject(s)
        per_subject.append(result)
        counts.append({"subject": s, **{c: {"kept": result[c]["n_kept"], "total": result[c]["n_total"]} for c in RUNS}})
        print(f"  subject {s:3d}: {counts[-1]}  ({time.time() - t:.1f} s)")
    ga = grand_average(per_subject)
    info = load_raw(subjects[0], [RUNS["executed"][0]]).info
    keep_t = slice(None, None, TIME_STEP)
    conditions = {}
    for cond, e in ga["conditions"].items():
        conditions[cond] = {
            "courses": {band: {
                "contra_mean": _r(c["contra_mean"][keep_t]), "ipsi_mean": _r(c["ipsi_mean"][keep_t]),
                "contra_sem": _r(c["contra_sem"][keep_t]), "ipsi_sem": _r(c["ipsi_sem"][keep_t]),
                "contra_window_per_subject": _r(c["contra_window_per_subject"], 1),
                "ipsi_window_per_subject": _r(c["ipsi_window_per_subject"], 1),
            } for band, c in e["courses"].items()},
            "tfr_contra": _r(e["tfr_contra"][:, keep_t], 1),
            "drop_time": e["drop_time"],
            "artifact_checks": {
                "burst": {**BURST, "by_channel": dict(zip(ga["ch_names"], _r(e["burst_by_channel"], 1)))},
                "slow": {**SLOW, "by_channel": dict(zip(ga["ch_names"], _r(e["slow_by_channel"], 1)))},
            },
            "topo": {band: {hand: dict(zip(ga["ch_names"], _r(v, 1))) for hand, v in hands.items()}
                     for band, hands in e["topo"].items()},
        }
    return {
        "dataset": {
            "name": "EEG Motor Movement/Imagery Dataset v1.0.0 (PhysioNet)",
            "doi": "10.13026/C28G6P",
            "citation": "Schalk et al. (2004) IEEE TBME 51(6):1034-1043; Schalk (2009) PhysioNet",
            "license": "ODC-By 1.0",
            "subjects": list(subjects),
            "runs": RUNS,
        },
        "processing": {
            "montage": MONTAGE, "reference": "common average", "filter_hz": FILTER, "epoch_s": EPOCH, "show_s": SHOW,
            "baseline_s": BASELINE, "reject": {"channels": REJECT_CHANNELS, "rule": f"per volunteer and condition: central peak-to-peak > median + {REJECT_MADS:g} x 1.4826 x MAD"},
            "tfr": {"method": "morlet", "n_cycles": "freq / 2", "decim": DECIM}, "bands_hz": BANDS,
            "erd_window_s": ERD_WINDOW, "power_change": "% change from baseline mean, averaged over trials, then volunteers",
            "contra": CONTRA,
        },
        "trial_counts": counts,
        "times": _r(ga["times"][keep_t], 3),
        "freqs": _r(ga["freqs"], 1),
        "conditions": conditions,
        "electrodes": electrode_layout(info),
        "example": example_trace(),
    }


def main(subjects=SUBJECTS) -> None:
    trace = build_trace(subjects)
    TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRACE_PATH.write_text(json.dumps(trace, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {TRACE_PATH.relative_to(ROOT)} ({TRACE_PATH.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    import sys

    main([int(s) for s in sys.argv[1:]] or SUBJECTS)
