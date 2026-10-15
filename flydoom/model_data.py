"""Connectome data for the fly brain model, exactly as used by Shiu et al. (2024).

Source: github.com/philshiu/Drosophila_brain_model, pinned at commit 91bdd1e (MIT licence, (c) 2023 Philip Shiu and
Nico Spiller). Paper: Shiu et al. (2024) "A Drosophila computational brain model reveals sensorimotor processing",
Nature 634:210-219, doi:10.1038/s41586-024-07763-9. The connectivity itself is FlyWire data (CC BY 4.0;
Dorkenwald et al. 2024, Schlegel et al. 2024); neurotransmitter signs are machine-learning predictions (Eckstein et al. 2024).

Two releases, kept apart because neuron IDs differ between them:
  "630"  the paper's version; its example outputs (results/example/*.parquet, Brian2 spikes) are our port's reference
  "783"  the public release used by our anatomy videos and by the game experiment
Each file is verified against its git blob hash from the pinned commit (a checksum GitHub itself uses).

    uv run --group flydoom python -m flydoom.model_data     # download and verify (~195 MB into assets/data/shiu/)
"""
from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "assets" / "data" / "shiu"
COMMIT = "91bdd1e"
BASE = f"https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/{COMMIT}/"
FILES = {  # path in the repo: (bytes, git blob sha1)
    "2023_03_23_completeness_630_final.csv": (3057611, "be745f0ce054308df21accc5c4b3883aa38498f9"),
    "2023_03_23_connectivity_630_final.parquet": (86630944, "8b4d9531bc0acbda7c2074ae577e6ac9e2fca166"),
    "Completeness_783.csv": (3327347, "b5a26b82b69a3c2e7fd99cd6810c36fc8f0e492b"),
    "Connectivity_783.parquet": (100804642, "d386555d1a5f40ebfa1380bcb05b1fab044855fd"),
    "results/example/sugarR.parquet": (948526, "1204cb3ed453ad03b0141028fa511d924f99d599"),
    "results/example/sugarR_100Hz.parquet": (557849, "bd76730d5e991d64b756f354a749f94ffb874c64"),
}
VERSIONS = {
    "630": ("2023_03_23_completeness_630_final.csv", "2023_03_23_connectivity_630_final.parquet"),
    "783": ("Completeness_783.csv", "Connectivity_783.parquet"),
}


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def fetch(name: str) -> Path:
    path = DATA / name
    size, sha = FILES[name]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(BASE + name, timeout=600) as response:  # noqa: S310 (fixed https URL)
            data = response.read()
        path.with_suffix(".part").write_bytes(data)
        path.with_suffix(".part").replace(path)
    data = path.read_bytes()
    assert len(data) == size and git_blob_sha1(data) == sha, f"{name}: size or checksum differs from commit {COMMIT}"
    return path


@dataclass
class Connectome:
    version: str
    flywire_ids: np.ndarray  # model index -> FlyWire root id
    indptr: np.ndarray  # CSR over PRESYNAPTIC neurons: targets of neuron i are indices[indptr[i]:indptr[i+1]]
    indices: np.ndarray
    weights: np.ndarray  # signed synapse count ("Excitatory x Connectivity"), not yet multiplied by w_syn

    @property
    def n(self) -> int:
        return len(self.flywire_ids)

    def index_of(self, flywire_ids) -> np.ndarray:
        lookup = {int(f): i for i, f in enumerate(self.flywire_ids)}
        missing = [int(f) for f in flywire_ids if int(f) not in lookup]
        assert not missing, f"{len(missing)} FlyWire ids are not in release {self.version}: {missing[:3]}"
        return np.array([lookup[int(f)] for f in flywire_ids], dtype=np.int64)


def load(version: str = "783") -> Connectome:
    comp_name, con_name = VERSIONS[version]
    comp = pd.read_csv(fetch(comp_name), index_col=0)
    con = pd.read_parquet(fetch(con_name), columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"])
    pre = con["Presynaptic_Index"].to_numpy(np.int64)
    post = con["Postsynaptic_Index"].to_numpy(np.int64)
    w = con["Excitatory x Connectivity"].to_numpy(np.float64)
    order = np.lexsort((post, pre))
    pre, post, w = pre[order], post[order], w[order]
    indptr = np.zeros(len(comp) + 1, dtype=np.int64)
    np.add.at(indptr, pre + 1, 1)
    return Connectome(version, comp.index.to_numpy(np.int64), np.cumsum(indptr), post, w)


def reference_rates(name: str = "sugarR.parquet", n_run: int = 30, t_run: float = 1.0) -> pd.Series:
    """Mean firing rate (Hz) per FlyWire id in one of Shiu's Brian2 reference runs (v630)."""
    df = pd.read_parquet(fetch(f"results/example/{name}"))
    return df.groupby("flywire_id").size() / (n_run * t_run)


if __name__ == "__main__":
    for name in FILES:
        print(f"{name}: ok ({fetch(name).stat().st_size / 1e6:.1f} MB)")
