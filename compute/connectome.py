"""The adult fruit-fly brain connectome (FlyWire, FAFB v783), exported for the video "Inside a real brain's wiring".

Data (CC BY 4.0):
  Neurons, positions and cell types: Supplemental file 1 of Schlegel et al. (2024) Nature 634:139, release v2.1.0 of
    github.com/flyconnectome/flywire_annotations (the version reported in the papers), pinned by tag.
  Connections: connections.csv.gz of the FlyWire Codex v783 export (pairs with at least 5 synapses, one row per brain region),
    read from Codex's public bucket; generation and MD5 are checked below because that bucket is not a versioned release.
  Synapse positions: synapse_coordinates.csv.gz of the same export (317 MB), streamed once; only the circuit's synapses are kept.
  Dorkenwald et al. (2024) Nature 634:124 (the connectome); Zheng et al. (2018) Cell 174:730 (the electron-microscopy volume).
  One adult female Drosophila melanogaster brain. The ventral nerve cord (the fly's "spinal cord") is not in this volume.

Positions: FlyWire voxel coordinates (4 x 4 x 40 nm) converted to micrometres. x grows toward the fly's RIGHT,
y grows toward the BOTTOM of the brain (ventral), z grows toward the BACK (posterior).
Each neuron is placed at its cell body; neurons whose cell body lies outside the brain (mostly sensory neurons)
have no soma position and are placed at the annotation's anchor point on the neuron instead.

Run from the project root:  uv run python -m compute.connectome
  downloads two files once (27 MB + 50 MB) into assets/data/flywire/ (not tracked by Git) and writes
  assets/connectome/connectome_trace.json (the circuit) and assets/connectome/brain_points.npz (every neuron, compact).
"""
from __future__ import annotations

import base64
import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "assets" / "data" / "flywire"
OUT_DIR = ROOT / "assets" / "connectome"
TRACE_PATH = OUT_DIR / "connectome_trace.json"
POINTS_PATH = OUT_DIR / "brain_points.npz"
SYNAPSES_PATH = OUT_DIR / "circuit_synapses.npz"

ANNOTATIONS = {
    "url": "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/v2.1.0/supplemental_files/Supplemental_file1_neuron_annotations.tsv",
    "file": "Supplemental_file1_neuron_annotations_v2.1.0.tsv", "bytes": 27015208,
}
CONNECTIONS = {
    "url": "https://storage.googleapis.com/flywire-data/codex/data/fafb/783/connections.csv.gz",
    "file": "connections_783.csv.gz", "bytes": 50289304, "md5_base64": "QSBkQDGMd0GL/Hz/HLPg+g==", "gcs_generation": "1733428416885848",
}
SYNAPSES = {  # one row per synapse, x/y/z in nm; empty ids repeat the row above. Streamed once, never stored whole.
    "url": "https://storage.googleapis.com/flywire-data/codex/data/fafb/783/synapse_coordinates.csv.gz",
    "bytes": 316819225, "md5_base64": "nKST4jFMwDqviwPoDxZD6A==", "gcs_generation": "1733428450553759",
}
VOXEL_NM = np.array([4.0, 4.0, 40.0])

# ---- the story's choices ----------------------------------------------------------------------------
FOCUS_TYPE = "DNp01"  # the giant fiber; one per side
FOCUS_SIDE = "right"
FIRST_GROUP = 12  # beat 1 shows the giant fiber and its strongest inputs: this many neurons in total
LOOMING_TYPES = ["LPLC2", "LC4"]  # the two largest input types of the giant fiber (checked in build())
CLASS_ORDER = ["optic", "central", "sensory", "visual_projection", "ascending", "descending", "visual_centrifugal", "motor", "endocrine"]


def fetch(spec: dict) -> Path:
    path = DATA_DIR / spec["file"]
    if not path.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(spec["url"], timeout=300) as response:  # noqa: S310 (fixed https URLs)
            data = response.read()
        path.with_suffix(".part").write_bytes(data)
        path.with_suffix(".part").replace(path)
    data = path.read_bytes()
    assert len(data) == spec["bytes"], f"{path.name}: {len(data)} bytes, expected {spec['bytes']} (the source changed?)"
    if "md5_base64" in spec:
        assert base64.b64encode(hashlib.md5(data).digest()).decode() == spec["md5_base64"], f"{path.name}: checksum differs"
    return path


def load_neurons() -> pd.DataFrame:
    a = pd.read_csv(fetch(ANNOTATIONS), sep="\t", low_memory=False)
    # a FlyWire-specific type where one exists, otherwise the matched hemibrain type (how LC4 and LPLC2 are typed in v2.1.0)
    a["type"] = a["cell_type"].fillna(a["hemibrain_type"])
    has_soma = a["soma_x"].notna()
    xyz_vox = np.where(has_soma.to_numpy()[:, None], a[["soma_x", "soma_y", "soma_z"]].to_numpy(float), a[["pos_x", "pos_y", "pos_z"]].to_numpy(float))
    xyz = xyz_vox * VOXEL_NM / 1000.0  # micrometres
    a["x"], a["y"], a["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    a["at_soma"] = has_soma
    return a


def load_pairs() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = pd.read_csv(fetch(CONNECTIONS))
    pairs = rows.groupby(["pre_root_id", "post_root_id"], as_index=False)["syn_count"].sum()
    return rows, pairs


def stream_circuit_synapses(post_ids: set, pre_ids: set, chunk_rows: int = 2_000_000) -> pd.DataFrame:
    """Stream the synapse table and keep synapses onto post_ids or from pre_ids. Checks size and MD5 of the stream."""
    import gzip

    md5, size = hashlib.md5(), 0

    class Counting:  # passes bytes through while hashing them
        def __init__(self, raw):
            self.raw = raw

        def read(self, n=-1):
            nonlocal size
            b = self.raw.read(n)
            md5.update(b)
            size += len(b)
            return b

    kept, last = [], {"pre_root_id": None, "post_root_id": None}
    with urllib.request.urlopen(SYNAPSES["url"], timeout=600) as response:  # noqa: S310 (fixed https URL)
        with gzip.open(Counting(response), "rt") as text:
            for chunk in pd.read_csv(text, chunksize=chunk_rows, dtype={"pre_root_id": "Int64", "post_root_id": "Int64"}):
                for col in ("pre_root_id", "post_root_id"):  # blank = same as the row above, also across chunk borders
                    if pd.isna(chunk[col].iloc[0]):
                        chunk.loc[chunk.index[0], col] = last[col]
                    chunk[col] = chunk[col].ffill()
                    last[col] = chunk[col].iloc[-1]
                keep = chunk["post_root_id"].isin(post_ids) | chunk["pre_root_id"].isin(pre_ids)
                kept.append(chunk[keep])
    assert size == SYNAPSES["bytes"], f"synapse file: {size} bytes, expected {SYNAPSES['bytes']}"
    assert base64.b64encode(md5.digest()).decode() == SYNAPSES["md5_base64"], "synapse file: checksum differs"
    out = pd.concat(kept, ignore_index=True)
    out[["x", "y", "z"]] = out[["x", "y", "z"]] / 1000.0  # nm -> micrometres, the units of everything else
    return out


def circuit_synapses(a: pd.DataFrame, gf_ids: list, looming_ids: list) -> pd.DataFrame:
    """Cached extract: synapses onto/from the giant fibers, and from the looming detectors of the focus side."""
    if SYNAPSES_PATH.exists():
        z = np.load(SYNAPSES_PATH)
        cached = pd.DataFrame({"pre_root_id": z["pre"], "post_root_id": z["post"], "x": z["xyz"][:, 0], "y": z["xyz"][:, 1], "z": z["xyz"][:, 2]})
        if set(z["gf_ids"]) == set(gf_ids) and set(z["looming_ids"]) == set(looming_ids):
            return cached
    syn = stream_circuit_synapses(set(gf_ids), set(gf_ids) | set(looming_ids))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(SYNAPSES_PATH, pre=syn["pre_root_id"].to_numpy(np.int64), post=syn["post_root_id"].to_numpy(np.int64),
                        xyz=syn[["x", "y", "z"]].to_numpy(np.float32), gf_ids=np.array(gf_ids, np.int64), looming_ids=np.array(looming_ids, np.int64))
    return syn


def neuron_record(row) -> dict:
    return {"id": str(row.root_id), "type": None if pd.isna(row.type) else str(row.type), "class": str(row.super_class),
            "side": None if pd.isna(row.side) else str(row.side), "xyz": [round(float(row.x), 2), round(float(row.y), 2), round(float(row.z), 2)],
            "at_soma": bool(row.at_soma),
            "predicted_nt": None if pd.isna(row.top_nt) else str(row.top_nt),
            "predicted_nt_confidence": None if pd.isna(row.top_nt_conf) else round(float(row.top_nt_conf), 3)}


def build() -> tuple[dict, dict]:
    a = load_neurons()
    rows, pairs = load_pairs()
    by_id = a.set_index("root_id")
    focus = a[(a["type"] == FOCUS_TYPE) & (a["side"] == FOCUS_SIDE)]
    assert len(focus) == 1
    gf = focus.iloc[0]
    inputs = pairs[pairs.post_root_id == gf.root_id].join(by_id[["type", "super_class", "side"]], on="pre_root_id")
    outputs = pairs[pairs.pre_root_id == gf.root_id]
    by_type = (inputs.groupby("type").agg(neurons=("pre_root_id", "size"), synapses=("syn_count", "sum"))
               .sort_values("synapses", ascending=False))
    assert set(by_type.index[:2]) == set(LOOMING_TYPES), f"the two largest input types are now {list(by_type.index[:2])}"
    regions = rows[rows.post_root_id == gf.root_id].groupby("neuropil")["syn_count"].sum().sort_values(ascending=False)

    # beat 1: the giant fiber and its strongest individual inputs, with every >=5-synapse connection among them
    strongest = inputs.sort_values("syn_count", ascending=False).head(FIRST_GROUP - 1)
    group_ids = [gf.root_id] + strongest.pre_root_id.tolist()
    group_edges = pairs[pairs.pre_root_id.isin(group_ids) & pairs.post_root_id.isin(group_ids)]

    # the looming detectors on the focus side, and which of them reach the giant fiber
    looming = {}
    for t in LOOMING_TYPES:
        pop = a[(a["type"] == t) & (a["side"] == FOCUS_SIDE)]
        syn = inputs.set_index("pre_root_id")["syn_count"]
        looming[t] = [{**neuron_record(r), "syn_to_focus": int(syn.get(r.root_id, 0))} for r in pop.itertuples()]

    # real synapse positions for the circuit (streamed once, then cached in assets/connectome/circuit_synapses.npz)
    gf_ids = a[a["type"] == FOCUS_TYPE].root_id.tolist()
    loom_ids = [int(n["id"]) for t in LOOMING_TYPES for n in looming[t]]
    syn = circuit_synapses(a, gf_ids, loom_ids)
    onto_focus = syn[syn.post_root_id == gf.root_id]
    loom_type = {int(n["id"]): t for t in LOOMING_TYPES for n in looming[t]}
    synapse_extract = {
        "focus_inputs": int(len(onto_focus)),
        "focus_inputs_by_type": {t: int(onto_focus.pre_root_id.map(loom_type).eq(t).sum()) for t in LOOMING_TYPES},
        "looming_outputs": {t: int(syn.pre_root_id.map(loom_type).eq(t).sum()) for t in LOOMING_TYPES},
        "other_side_inputs": int((syn.post_root_id == gf_ids[1 - gf_ids.index(gf.root_id)]).sum()),
    }
    row_of = {r.root_id: r for r in a[a.root_id.isin(group_ids)].itertuples()}
    counts = a["super_class"].value_counts()
    trace = {
        "dataset": {
            "name": "FlyWire whole-brain connectome, FAFB v783", "license": "CC BY 4.0",
            "citations": ["Dorkenwald et al. (2024) Nature 634:124", "Schlegel et al. (2024) Nature 634:139", "Zheng et al. (2018) Cell 174:730"],
            "annotations": ANNOTATIONS["url"], "connections": CONNECTIONS["url"], "connections_generation": CONNECTIONS["gcs_generation"],
            "animal": "one adult female Drosophila melanogaster (brain only; no ventral nerve cord)",
        },
        "totals": {"neurons": int(len(a)), "with_soma_position": int(a["at_soma"].sum()),
                   "connections_ge5": int(len(pairs)), "synapses_in_those": int(pairs["syn_count"].sum()),
                   "classes": {k: int(counts.get(k, 0)) for k in CLASS_ORDER}},
        "extent_um": {ax: [round(float(a[ax].min()), 1), round(float(a[ax].max()), 1)] for ax in ("x", "y", "z")},
        "focus": {**neuron_record(gf), "other_side": neuron_record(a[(a["type"] == FOCUS_TYPE) & (a["side"] != FOCUS_SIDE)].iloc[0]),
                  "n_inputs": int(len(inputs)), "input_synapses": int(inputs["syn_count"].sum()),
                  "n_outputs_in_brain": int(len(outputs)), "output_synapses_in_brain": int(outputs["syn_count"].sum()),
                  "input_types": [{"type": str(t), "neurons": int(r.neurons), "synapses": int(r.synapses)} for t, r in by_type.head(8).iterrows()],
                  "input_regions": {str(k): int(v) for k, v in regions.head(5).items()}},
        "first_group": {"neurons": [{**neuron_record(row_of[i]), "syn_to_focus": int(strongest.set_index("pre_root_id")["syn_count"].get(i, 0))}
                                    for i in group_ids],
                        "edges": [{"pre": str(r.pre_root_id), "post": str(r.post_root_id), "syn": int(r.syn_count)} for r in group_edges.itertuples()]},
        "looming": looming,
        "synapse_extract": synapse_extract,
    }
    points = {"xyz": a[["x", "y", "z"]].to_numpy(np.float32), "cls": a["super_class"].map({k: i for i, k in enumerate(CLASS_ORDER)}).to_numpy(np.int8),
              "class_names": np.array(CLASS_ORDER)}
    return trace, points


def main() -> None:
    trace, points = build()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TRACE_PATH.write_text(json.dumps(trace, separators=(",", ":")), encoding="utf-8")
    np.savez_compressed(POINTS_PATH, **points)
    for p in (TRACE_PATH, POINTS_PATH):
        print(f"wrote {p.relative_to(ROOT)} ({p.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
