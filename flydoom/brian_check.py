"""Cross-check of the numba port against Shiu's UNMODIFIED model.py running in Brian2 (optional; Brian2 is not a project dependency).

Step 1 runs in a throwaway environment with Brian2 (numpy code generation, no compiler needed), step 2 in the project:
    uv run --no-project --python 3.11 --with brian2==2.9.0 --with "numpy<2.4" --with pandas --with pyarrow --with joblib python -m flydoom.brian_check brian
    (numpy<2.4: Brian2 2.9.0 uses ndarray.ptp, which numpy 2.4 removed)
    uv run --group flydoom python -m flydoom.brian_check compare     # writes assets/flydoom/brian_check.json

What it checks:
  replay   10 Poisson-driven neurons -> 1 target (5 s). Brian2's exact input spike times are replayed into the port;
           the target's spike count and membrane trace must match. (This is how the refractory-input rule in lif.py was found.)
  sugar    the full v630 brain, Shiu's sugarR condition, 30 trials in each simulator; per-neuron rates compared.
Step 1 takes about 7 minutes. Outputs of step 1 go to assets/data/shiu/brian_check/ (not tracked by Git).
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "assets" / "data" / "shiu" / "brian_check"
OUT = ROOT / "assets" / "flydoom" / "brian_check.json"
SECONDS_REPLAY = 5.0


def brian() -> None:
    """Runs under Brian2. Imports Shiu's model.py from the pinned commit (downloaded next to the outputs)."""
    import numpy as np
    import pandas as pd
    from brian2 import Hz, Network, NeuronGroup, PoissonInput, SpikeMonitor, StateMonitor, Synapses, mV, ms, prefs, second, seed

    WORK.mkdir(parents=True, exist_ok=True)
    model_py = WORK / "model.py"
    if not model_py.exists():
        url = "https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/91bdd1e/model.py"
        model_py.write_bytes(urllib.request.urlopen(url, timeout=60).read())  # noqa: S310 (fixed https URL)
    sys.path.insert(0, str(WORK))
    import model  # Shiu et al., unmodified

    prefs.codegen.target = "numpy"
    p = model.default_params
    # replay network
    seed(3)
    neu = NeuronGroup(11, p["eqs"], method="linear", threshold=p["eq_th"], reset=p["eq_rst"], refractory="rfc", namespace=p)
    neu.v, neu.g, neu.rfc = p["v_0"], 0, p["t_rfc"]
    pois = [PoissonInput(neu[i], "v", 1, 200 * Hz, p["w_syn"] * p["f_poi"]) for i in range(10)]
    neu.rfc[:10] = 0 * ms
    syn = Synapses(neu, neu, "w : volt", on_pre="g += w", delay=p["t_dly"])
    syn.connect(i=list(range(10)), j=[10] * 10)
    syn.w = 6 * p["w_syn"]
    spk, mon = SpikeMonitor(neu), StateMonitor(neu, "v", record=[10])
    Network(neu, syn, spk, mon, *pois).run(SECONDS_REPLAY * second)
    steps = [[int(round(t / (0.1 * ms))), int(i)] for i, t in zip(spk.i[:], spk.t[:])]
    json.dump(steps, open(WORK / "replay_spikes.json", "w"))
    np.save(WORK / "replay_v.npy", mon.v[0] / mV)
    # full brain, sugarR, 30 trials
    from .model_data import DATA, VERSIONS
    from .cells import SUGAR

    comp_p, con_p = (str(DATA / f) for f in VERSIONS["630"])
    comp = pd.read_csv(comp_p, index_col=0)
    exc = [{f: i for i, f in enumerate(comp.index)}[f] for f in SUGAR]
    params = dict(p, r_poi=200 * Hz)
    counts = []
    for k in range(30):
        res = model.run_trial(exc, [], [], comp_p, con_p, params)
        c = np.zeros(len(comp), dtype=np.int64)
        for i, ts in res.items():
            c[i] = len(ts)
        counts.append(c)
        print(f"trial {k + 1}/30", flush=True)
    np.save(WORK / "sugar_counts.npy", np.array(counts))


def compare() -> None:
    import numpy as np

    from . import lif
    from . import model_data as md
    from .cells import MN9, SUGAR

    steps = json.load(open(WORK / "replay_spikes.json"))
    inputs = sorted(s for s in steps if s[1] < 10)
    target_brian = sorted(s[0] for s in steps if s[1] == 10)
    toy = md.Connectome("toy", np.arange(11), np.r_[np.arange(11), 10], np.full(10, 10), np.full(10, 6.0))
    run, _ = lif.simulate(toy, {0.0: list(range(10))}, t_run_s=SECONDS_REPLAY, trials=1, forced=inputs, v_record=[10])
    v_brian = np.load(WORK / "replay_v.npy")
    v_ours = run.v[0][0]
    max_dv = float(np.abs(v_ours[:-1] - v_brian[1:]).max())  # ours = end of step t, Brian's monitor = start of step t+1
    con = md.load("630")
    brian_counts = np.load(WORK / "sugar_counts.npy")
    ours, _ = lif.simulate(con, {200.0: con.index_of(SUGAR)}, trials=brian_counts.shape[0], seed=0)
    rb, ro = brian_counts.mean(0), ours.counts.mean(0)
    se = np.sqrt(brian_counts.var(0, ddof=1) / len(brian_counts) + ours.counts.var(0, ddof=1) / len(ours.counts)) + 1e-12
    big = np.maximum(rb, ro) > 5
    i = con.index_of([MN9["right"]])[0]
    out = {"replay": {"target_spikes_brian": len(target_brian), "target_spikes_port": int(run.counts[0][10]),
                      "inputs_replayed": len(inputs), "max_abs_dv_mV": round(max_dv, 6)},
           "sugar": {"trials_each": int(brian_counts.shape[0]), "pearson_rates": round(float(np.corrcoef(ro, rb)[0, 1]), 5),
                     "neurons_over_5hz": int(big.sum()), "neurons_z_over_3": int((np.abs((ro - rb) / se)[big] > 3).sum()),
                     "mn9_hz": {"brian2": round(float(rb[i]), 2), "port": round(float(ro[i]), 2)}}}
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"brian": brian, "compare": compare}[sys.argv[1]]()
