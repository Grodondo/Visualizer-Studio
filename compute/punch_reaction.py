"""Real EEG of a person seeing a visual target and pressing a button as fast as possible, exported for the video
"What happens in your brain when you react to a punch?".

Dataset: ERP CORE, Flankers task, subject 001 (the copy shipped by MNE-Python as mne.datasets.erp_core).
  Kappenman E.S., Farrens J.L., Zhang W., Stewart A.X., Luck S.J. (2021) ERP CORE: An open resource for human
  event-related potential research. NeuroImage 225:117465. https://doi.org/10.1016/j.neuroimage.2020.117465
  Licence: CC BY-SA 4.0 (README of the MNE copy). Anything published with these data must credit the authors and
  be shared under the same licence.
  Task: a row of arrows appears; press the left or right button for the direction of the CENTRAL arrow, quickly.
  30 scalp electrodes (Biosemi), 1024 samples/s. Events: stimulus/{compatible,incompatible}/target_{left,right},
  response/{left,right}.

Brain template: fsaverage (FreeSurfer; Fischl, Sereno, Tootell & Dale 1999, Hum Brain Mapp 8:272), with MNE's
  precomputed head model; regions from the Desikan-Killiany atlas (Desikan et al. 2006, NeuroImage 31:968).
  This person's own MRI is not available, so WHERE activity comes from is ESTIMATED on a template brain (dSPM,
  Dale et al. 2000, Neuron 26:55). With 30 electrodes this is a blurred, approximate picture, not a measurement.

Run from the project root:  uv run python -m compute.punch_reaction
  downloads ERP CORE (92 MB) and fsaverage (about 430 MB) once into assets/data/ (not tracked by Git), then writes
  assets/punch_reaction/punch_trace.json (waveforms, reaction times, numbers) and
  assets/punch_reaction/sources.npz (estimated cortical activity over time, read by compute.brain_render).

Every analysis choice is fixed below and was chosen before looking at the results.
"""
from __future__ import annotations

import json
from pathlib import Path

import mne
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "assets" / "data"
ERP_DIR = DATA_DIR / "erp_core"
SUBJECTS_DIR = DATA_DIR / "fsaverage_subjects"
OUT_DIR = ROOT / "assets" / "punch_reaction"
TRACE_PATH = OUT_DIR / "punch_trace.json"
SOURCES_PATH = OUT_DIR / "sources.npz"

STIMULI = ["stimulus/compatible/target_left", "stimulus/compatible/target_right",
           "stimulus/incompatible/target_left", "stimulus/incompatible/target_right"]
RESPONSES = ["response/left", "response/right"]

# ---- processing (fixed in advance) -----------------------------------------------------------------
FILTER = (0.1, 30.0)  # Hz; 0.1 Hz keeps slow ERPs such as the motor-preparation signal (ERP CORE used 0.1 Hz too)
RESAMPLE = 256.0  # Hz; ample for activity below 30 Hz
RT_RANGE = (0.2, 1.0)  # s; responses outside this window are not counted as reactions (the ERP CORE paper's window)
STIM_EPOCH = (-0.2, 0.8)  # s around the arrows; baseline is the 0.2 s before them
RESP_EPOCH = (-0.8, 0.2)  # s around the button press
RESP_BASELINE = (-0.8, -0.6)  # s; the ERP CORE paper's baseline for the response-locked motor signal
REJECT_MADS = 3.0  # drop an epoch whose largest EEG swing is an outlier: above median + 3 robust SD
VISUAL_CHANNELS = ["PO7", "PO8", "O1", "O2", "Oz"]  # over visual cortex
P1_WINDOW, N1_WINDOW = (0.06, 0.14), (0.14, 0.22)  # s; where the first positive and negative visual peaks are searched
INVERSE = {"method": "dSPM", "snr": 3.0, "loose": 0.2, "depth": 0.8}  # MNE's standard settings for evoked data
SOURCE_WINDOW = (-0.05, 0.30)  # s after the arrows exported as brain frames (every sample, 256 per second)
TRUSTED_UNTIL = 0.22  # s; the video shows estimated activity only up to here (see "where" in the trace for why)
LIMIT_WINDOW = (-0.15, 0.0)  # s before the press: the right-minus-left-hand map shown as the estimate's limitation
LRP_FRACTION = 0.5  # LRP onset = last time before its peak at which it is smaller than this fraction of the peak
REGIONS = {  # Desikan-Killiany atlas labels (aparc) grouped into the regions the video names
    "visual cortex (occipital)": ["lateraloccipital", "cuneus", "pericalcarine", "lingual"],
    "parietal cortex": ["superiorparietal", "inferiorparietal", "supramarginal", "precuneus"],
    "motor & premotor cortex": ["precentral", "caudalmiddlefrontal", "paracentral"],
    "frontal cortex": ["superiorfrontal", "rostralmiddlefrontal", "parsopercularis", "parstriangularis"],
}


def outlier_limit(values: np.ndarray) -> float:
    med = float(np.median(values))
    return med + REJECT_MADS * 1.4826 * float(np.median(np.abs(values - med)))


def load_raw() -> mne.io.BaseRaw:
    path = Path(mne.datasets.erp_core.data_path(path=str(ERP_DIR.resolve()), update_path=False, verbose=False))
    raw = mne.io.read_raw_fif(path / "ERP-CORE_Subject-001_Task-Flankers_eeg.fif", preload=True, verbose=False)
    raw.filter(*FILTER, verbose=False)
    raw.resample(RESAMPLE, verbose=False)
    return raw


def trials(raw: mne.io.BaseRaw):
    """Metadata with one row per arrow display: which side, the first response, its time, and whether it was correct."""
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    meta, ev, eid = mne.epochs.make_metadata(events=events, event_id=event_id, tmin=0.0, tmax=1.5,
                                             sfreq=raw.info["sfreq"], row_events=STIMULI, keep_first=["stimulus", "response"])
    meta["target"] = meta["event_name"].str.rsplit("_", n=1).str[-1]
    meta["compatible"] = meta["event_name"].str.contains("/compatible/")
    meta["rt"] = meta["response"]
    meta["correct"] = meta["first_response"] == meta["target"]
    meta["counted"] = meta["correct"] & meta["rt"].between(*RT_RANGE)
    return meta, ev, eid, events, event_id


def drop_outliers(epochs: mne.Epochs) -> mne.Epochs:
    swing = np.ptp(epochs.get_data(picks="eeg"), axis=2).max(axis=1)
    bad = np.flatnonzero(swing > outlier_limit(swing))
    return epochs.drop(bad, reason="peak-to-peak outlier", verbose=False)


def first_peak(evoked: mne.Evoked, channels, window, polarity: int) -> dict:
    """Latency and size of the largest positive (polarity=+1) or negative (-1) deflection in a window, mean of channels."""
    data = evoked.copy().pick(channels).data.mean(axis=0) * 1e6
    t = evoked.times
    inside = (t >= window[0]) & (t <= window[1])
    i = np.flatnonzero(inside)[np.argmax(polarity * data[inside])]
    return {"latency_s": round(float(t[i]), 4), "amplitude_uv": round(float(data[i]), 2)}


def source_model(info: mne.Info):
    fs = Path(mne.datasets.fetch_fsaverage(subjects_dir=SUBJECTS_DIR.resolve(), verbose=False))
    src = mne.read_source_spaces(fs / "bem" / "fsaverage-ico-5-src.fif", verbose=False)
    fwd = mne.make_forward_solution(info, trans="fsaverage", src=src, bem=fs / "bem" / "fsaverage-5120-5120-5120-bem-sol.fif",
                                    eeg=True, meg=False, verbose=False)
    return src, fwd


def region_courses(stc: mne.SourceEstimate) -> dict:
    """Mean estimated activity per named region (both hemispheres), over time."""
    labels = mne.read_labels_from_annot("fsaverage", "aparc", subjects_dir=SUBJECTS_DIR.resolve(), verbose=False)
    out = {}
    for region, names in REGIONS.items():
        chosen = [l for l in labels if l.name.rsplit("-", 1)[0] in names]
        label = chosen[0]
        for l in chosen[1:]:
            label = label + l
        out[region] = np.round(stc.in_label(label).data.mean(axis=0), 3).tolist()
    return out


def atlas_lookup(stc) -> dict:
    """Source index -> Desikan-Killiany label name ('unknown' for the medial wall, which has no label)."""
    labels = mne.read_labels_from_annot("fsaverage", "aparc", subjects_dir=SUBJECTS_DIR.resolve(), verbose=False)
    n_lh, lookup = len(stc.vertices[0]), {}
    for label in labels:
        h = 0 if label.hemi == "lh" else 1
        for i in np.flatnonzero(np.isin(stc.vertices[h], label.vertices)):
            lookup[int(i) + h * n_lh] = label.name
    return lookup


def where_strongest(stc_stim, contrast, limit_cols, n_top: int = 200) -> dict:
    """For every 20 ms after the arrows: the atlas labels holding most of the n_top strongest sources.
    Also, for the right-minus-left-hand map before the press, how evenly it is split between the hemispheres."""
    lookup = atlas_lookup(stc_stim)
    steps = []
    for t in np.arange(0.06, SOURCE_WINDOW[1] + 1e-9, 0.02):
        i = int(np.argmin(np.abs(stc_stim.times - t)))
        names = [lookup.get(int(j), "unknown") for j in np.argsort(stc_stim.data[:, i])[::-1][:n_top]]
        u, c = np.unique(names, return_counts=True)
        o = np.argsort(c)[::-1][:3]
        steps.append({"t": round(float(t), 3), "max": round(float(stc_stim.data[:, i].max()), 2),
                      "top_labels": [[str(u[k]), int(c[k])] for k in o]})
    m = contrast.data[:, limit_cols].mean(axis=1)
    n_lh = len(contrast.vertices[0])
    motor = [i for i, name in lookup.items() if name.split("-")[0] in ("precentral", "postcentral")]
    lh = float(np.mean([m[i] for i in motor if i < n_lh]))
    rh = float(np.mean([m[i] for i in motor if i >= n_lh]))
    return {"stim": steps, "hand_contrast_central_mean": {"lh": round(lh, 3), "rh": round(rh, 3)}}


def lrp_onset(times: np.ndarray, lrp: np.ndarray) -> float:
    """Fractional-peak onset: the LRP is negative; walk back from its most negative point until it is above
    LRP_FRACTION x peak. Restricted to before the press."""
    before = times <= 0
    i_peak = int(np.argmin(np.where(before, lrp, np.inf)))
    target = LRP_FRACTION * lrp[i_peak]
    i = i_peak
    while i > 0 and lrp[i] <= target:
        i -= 1
    return round(float(times[i + 1]), 4)


def electrode_layout(info: mne.Info) -> dict:
    """Flat positions of this recording's electrodes, in the same frame as the movement video (studio.eeg.HeadMap):
    computed on the full standard 10-05 set (which has the Fpz/T7/Oz/T8 ring) and then kept for these channels."""
    from compute.eeg_motor import MONTAGE, electrode_layout as flat

    montage = mne.channels.make_standard_montage(MONTAGE)
    full = mne.create_info(montage.ch_names, 100.0, "eeg")
    full.set_montage(montage, verbose=False)
    layout = {name.lower(): xy for name, xy in flat(full).items()}
    return {name: layout[name.lower()] for name in info.ch_names}


def build() -> tuple[dict, dict]:
    raw = load_raw()
    meta, ev, eid, events, event_id = trials(raw)
    raw.set_eeg_reference("average", projection=True, verbose=False)  # required for source estimation

    # stimulus-locked: what the visual system does after the arrows appear (correct, counted trials only)
    stim = mne.Epochs(raw, ev, eid, tmin=STIM_EPOCH[0], tmax=STIM_EPOCH[1], baseline=(None, 0), metadata=meta,
                      event_repeated="drop", preload=True, verbose=False, picks=["eeg", "eog"])
    stim = drop_outliers(stim["counted"])
    stim_evoked = stim.average(picks="eeg")

    # response-locked: what happens just before the button press
    resp_ids = {k: v for k, v in event_id.items() if k in RESPONSES}
    resp_events = events[np.isin(events[:, 2], list(resp_ids.values()))]
    resp = mne.Epochs(raw, resp_events, resp_ids, tmin=RESP_EPOCH[0], tmax=RESP_EPOCH[1], baseline=RESP_BASELINE,
                      preload=True, verbose=False, picks=["eeg", "eog"])
    resp = drop_outliers(resp)
    by_hand = {hand: resp[f"response/{hand}"].average(picks="eeg") for hand in ("left", "right")}
    # lateralized readiness potential: opposite-side minus same-side motor electrode, averaged over both hands
    c = lambda ev_, ch: ev_.copy().pick([ch]).data[0] * 1e6  # noqa: E731
    lrp = ((c(by_hand["right"], "C3") - c(by_hand["right"], "C4")) + (c(by_hand["left"], "C4") - c(by_hand["left"], "C3"))) / 2

    # where on the cortex: template forward model, noise from the pre-stimulus baseline
    src, fwd = source_model(stim_evoked.info)
    # noise = the EEG in the 0.2 s before the arrows; plain estimate plus MNE's standard 10 % regularisation
    eeg_only = stim.copy().pick("eeg")
    cov = mne.cov.regularize(mne.compute_covariance(eeg_only, tmax=0.0, method="empirical", verbose=False),
                             eeg_only.info, eeg=0.1, verbose=False)
    inv = mne.minimum_norm.make_inverse_operator(stim_evoked.info, fwd, cov, loose=INVERSE["loose"], depth=INVERSE["depth"], verbose=False)
    lam = 1.0 / INVERSE["snr"] ** 2
    stc_stim = mne.minimum_norm.apply_inverse(stim_evoked, inv, lam, method=INVERSE["method"], verbose=False)

    keep = (stc_stim.times >= SOURCE_WINDOW[0] - 1e-9) & (stc_stim.times <= SOURCE_WINDOW[1] + 1e-9)
    pre = stc_stim.data[:, stc_stim.times < 0]
    post = stc_stim.data[:, (stc_stim.times >= 0) & (stc_stim.times <= TRUSTED_UNTIL)]
    # right-hand minus left-hand presses: what differs between the hands, averaged over the 150 ms before the press
    contrast = mne.minimum_norm.apply_inverse(mne.combine_evoked([by_hand["right"], by_hand["left"]], weights=[1, -1]),
                                              inv, lam, method=INVERSE["method"], verbose=False)
    lw = (contrast.times > LIMIT_WINDOW[0]) & (contrast.times <= LIMIT_WINDOW[1])
    sources = {"lh_vertno": stc_stim.vertices[0], "rh_vertno": stc_stim.vertices[1],
               "stim_times": np.round(stc_stim.times[keep], 5), "stim_data": stc_stim.data[:, keep].astype(np.float32),
               # color scale: colored only above the largest pre-stimulus value; top = 99.5th percentile while shown
               "stim_limits": np.array([pre.max(), np.percentile(post, 99.5)], dtype=np.float32),
               "limit_map": contrast.data[:, lw].mean(axis=1).astype(np.float32),
               # like for like: the same 150 ms average taken from the baseline (-0.8 to -0.65 s) sets the threshold
               "limit_limits": np.array([contrast.data[:, contrast.times <= RESP_EPOCH[0] + 0.15].mean(axis=1).max(),
                                         np.percentile(contrast.data[:, lw].mean(axis=1), 99.5)], dtype=np.float32)}

    counted = meta[meta["counted"]]
    vis = stim_evoked.copy().pick(VISUAL_CHANNELS).data.mean(axis=0) * 1e6
    trace = {
        "dataset": {"name": "ERP CORE, Flankers task, subject 001", "citation": "Kappenman et al. (2021) NeuroImage 225:117465",
                    "doi": "10.1016/j.neuroimage.2020.117465", "license": "CC BY-SA 4.0",
                    "template": "fsaverage (FreeSurfer)", "inverse": INVERSE},
        "processing": {"filter_hz": FILTER, "resample_hz": RESAMPLE, "rt_range_s": RT_RANGE, "stim_epoch_s": STIM_EPOCH,
                       "resp_epoch_s": RESP_EPOCH, "resp_baseline_s": RESP_BASELINE, "reference": "average",
                       "reject": f"peak-to-peak > median + {REJECT_MADS:g} x 1.4826 x MAD", "visual_channels": VISUAL_CHANNELS},
        "trials": {"total": int(len(meta)), "correct": int(meta["correct"].sum()), "counted": int(len(counted)),
                   "stim_epochs_kept": int(len(stim)), "resp_epochs_kept": int(len(resp))},
        "rt": {"values_s": np.round(counted["rt"].to_numpy(), 4).tolist(), "median_s": round(float(counted["rt"].median()), 4),
               "compatible_median_s": round(float(counted.loc[counted["compatible"], "rt"].median()), 4),
               "incompatible_median_s": round(float(counted.loc[~counted["compatible"], "rt"].median()), 4)},
        "visual": {"times": np.round(stim_evoked.times, 4).tolist(), "uv": np.round(vis, 2).tolist(),
                   "p1": first_peak(stim_evoked, VISUAL_CHANNELS, P1_WINDOW, +1),
                   "n1": first_peak(stim_evoked, VISUAL_CHANNELS, N1_WINDOW, -1)},
        "lrp": {"times": np.round(by_hand["right"].times, 4).tolist(), "uv": np.round(lrp, 2).tolist()},
        "regions": {"stim_times": np.round(stc_stim.times, 4).tolist(), "stim": region_courses(stc_stim)},
        "where": where_strongest(stc_stim, contrast, lw),
        "lrp_onset_s": lrp_onset(by_hand["right"].times, lrp),
        "trusted_until_s": TRUSTED_UNTIL,
        "electrodes": electrode_layout(stim_evoked.info),
    }
    return trace, sources


def main() -> None:
    trace, sources = build()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TRACE_PATH.write_text(json.dumps(trace, separators=(",", ":")), encoding="utf-8")
    np.savez_compressed(SOURCES_PATH, **sources)
    for p in (TRACE_PATH, SOURCES_PATH):
        print(f"wrote {p.relative_to(ROOT)} ({p.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
