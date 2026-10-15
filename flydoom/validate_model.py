"""Milestone 2: validate the whole-brain LIF port BEFORE it is ever connected to the game.

    uv run --group flydoom python -m flydoom.validate_model      # ~10 minutes; writes assets/flydoom/model_validation.json

Part A, the port reproduces the published model (FlyWire v630, the paper's release):
    stimulate Shiu's 21 right-side sugar neurons at 200 Hz (their example), 30 trials of 1 s, compare every neuron's rate
    with Shiu's own Brian2 output (results/example/sugarR.parquet at commit 91bdd1e).
    (A one-off cross-check that ran Shiu's unmodified model.py in Brian2 next to this port is in flydoom/brian_check.py.)
Part B, behaviour on v783 (the release the game uses), each with the same run on a SHUFFLED connectome
(every connection keeps its presynaptic neuron and weight, the postsynaptic partners are permuted: each neuron keeps
exactly its number of inputs and outputs, but who-connects-to-whom is random):
    B1 sugar -> feeding:   the same 21 sugar neurons -> MN9 (the proboscis motor neuron). Published effect (Shiu et al. 2024).
    B2 looming -> escape:  right LC4 + LPLC2 -> giant fibers (DNp01) and steering neurons (DNa01, DNa02), at several rates.
                           A MODEL PREDICTION that matches anatomy and experiments (Ache et al. 2019; von Reyn et al. 2014),
                           not a published validation of this model. It is the pathway Milestone 3 will use.
Nothing here touches the game.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import lif
from . import model_data as md
from .cells import MN9, SUGAR

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets" / "flydoom" / "model_validation.json"
ANNOTATIONS = ROOT / "assets" / "data" / "flywire" / "Supplemental_file1_neuron_annotations_v2.1.0.tsv"

TRIALS_A, TRIALS_B = 30, 10
LOOM_RATES = [50.0, 100.0, 200.0]
SHUFFLE_SEED = 7


def shuffled(con: md.Connectome, seed: int = SHUFFLE_SEED) -> md.Connectome:
    rng = np.random.default_rng(seed)
    return md.Connectome(con.version + "-shuffled", con.flywire_ids, con.indptr, rng.permutation(con.indices), con.weights)


def cells(types: list[str], side: str | None = None) -> list[int]:
    a = pd.read_csv(ANNOTATIONS, sep="\t", low_memory=False, usecols=["root_id", "cell_type", "hemibrain_type", "side"])
    a["type"] = a["cell_type"].fillna(a["hemibrain_type"])
    m = a["type"].isin(types) & ((a["side"] == side) if side else True)
    return a.loc[m, "root_id"].astype(np.int64).tolist()


def part_a() -> dict:
    con = md.load("630")
    run, _ = lif.simulate(con, {200.0: con.index_of(SUGAR)}, trials=TRIALS_A, seed=0)
    ours = pd.Series(run.rates, index=con.flywire_ids)
    ref = md.reference_rates("sugarR.parquet")
    both = pd.concat([ours[ours > 0].rename("ours"), ref.rename("ref")], axis=1).fillna(0)
    i = con.index_of([MN9["right"]])[0]
    return {"release": "630", "stimulus": "21 right sugar GRNs, 200 Hz Poisson, 30 x 1 s", "reference": "Shiu et al. Brian2 sugarR.parquet",
            "active_ours": int((ours > 0).sum()), "active_reference": int(len(ref)),
            "pearson_rates": round(float(np.corrcoef(both.ours, both.ref)[0, 1]), 5),
            "mn9_hz": {"ours": round(float(run.rates[i]), 2), "ours_se": round(float(run.rate_se[i]), 2), "reference": round(float(ref[MN9["right"]]), 2)}}


def rates_of(con, stim_ids, rate, read: dict, seed=0) -> dict:
    run, _ = lif.simulate(con, {rate: con.index_of(stim_ids)}, trials=TRIALS_B, seed=seed)
    out = {}
    for name, ids in read.items():
        idx = con.index_of(ids)
        out[name] = {"mean_hz": round(float(run.rates[idx].mean()), 2), "per_cell_hz": [round(float(x), 2) for x in run.rates[idx]],
                     "se_hz": [round(float(x), 2) for x in run.rate_se[idx]]}
    out["active_neurons"] = int((run.rates > 0).sum())
    return out


def part_b() -> dict:
    real = md.load("783")
    null = shuffled(real)
    read_feed = {"MN9_right": [MN9["right"]], "MN9_left": [MN9["left"]]}
    loom_r = cells(["LC4", "LPLC2"], "right")
    read_loom = {f"{t}_{s}": cells([t], s) for t in ("DNp01", "DNa01", "DNa02") for s in ("right", "left")}
    present = set(real.flywire_ids.tolist())
    sugar = [i for i in SUGAR if i in present]  # proofreading changes some ids between releases; the rest are not replaced by guesses
    b = {"release": "783", "trials": TRIALS_B, "shuffle": "post-synaptic partners permuted (degree-preserving), seed %d" % SHUFFLE_SEED,
         "sugar_cells": len(sugar), "sugar_ids_missing_in_783": [str(i) for i in SUGAR if i not in present],
         "looming_cells_right": len(loom_r), "read_cells": {k: len(v) for k, v in read_loom.items()}}
    b["sugar_200hz"] = {"real": rates_of(real, sugar, 200.0, read_feed), "shuffled": rates_of(null, sugar, 200.0, read_feed)}
    b["looming_right"] = {}
    for r in LOOM_RATES:
        b["looming_right"][str(int(r))] = {"real": rates_of(real, loom_r, r, read_loom), "shuffled": rates_of(null, loom_r, r, read_loom)}
        print(f"  looming {r:.0f} Hz done", flush=True)
    b["no_stimulus"] = rates_of(real, [], 0.0, {**read_feed, **read_loom})
    return b


def main() -> None:
    t0 = time.time()
    print("part A (v630 vs Shiu's Brian2 output)...", flush=True)
    a = part_a()
    print(" ", a, flush=True)
    print("part B (v783 behaviour, real vs shuffled wiring)...", flush=True)
    b = part_b()
    out = {"model": "Shiu et al. 2024 LIF, numba port (flydoom/lif.py)", "params": lif.PARAMS, "data_commit": md.COMMIT,
           "part_a": a, "part_b": b, "runtime_s": round(time.time() - t0)}
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} in {out['runtime_s']} s")


if __name__ == "__main__":
    main()
