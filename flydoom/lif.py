"""Leaky integrate-and-fire simulation of the whole fly brain: a numba port of Shiu et al. (2024), model.py at commit 91bdd1e.

Same model, same constants (model.py "default_params"), same order of operations as Brian2's default schedule:
    every 0.1 ms step:
      1. state update, exact for this linear system (Brian2 method='linear'), skipped while a neuron is refractory:
             dv/dt = (v_0 - v + g) / t_mbr        dg/dt = -g / tau
      2. threshold: spike if v > v_th and the neuron is not refractory
      3. synapses: spikes from 1.8 ms ago add w = w_syn x (signed synapse count) to g of their targets, EXCEPT targets that
         are refractory: there the input is lost, because g is "(unless refractory)" and Brian2 keeps it unchanged
         (found by replaying identical input spikes through both simulators: before this rule our rates were ~20 % too high);
         stimulated neurons get Poisson events that add w_syn x f_poi to v
      4. reset of spiking neurons: v = v_rst, g = 0; refractory for t_rfc (stimulated neurons: no refractory period)
Silencing a neuron sets its OUTGOING weights to zero (what Shiu's code does; its README says "to and from").

Why numba and not Brian2: one dependency (numba) instead of Brian2 plus a C++ toolchain on Windows, and a loop we can step
in lock-step with the game later. tests/test_fly_model.py and flydoom/validate_model.py check it against Shiu's Brian2 output.
"""
from __future__ import annotations

from dataclasses import dataclass

import numba as nb
import numpy as np

from .model_data import Connectome

# Shiu et al. 2024, model.py default_params (values in mV and ms)
PARAMS = dict(v_0=-52.0, v_rst=-52.0, v_th=-45.0, t_mbr=20.0, tau=5.0, t_rfc=2.2, t_dly=1.8, w_syn=0.275, f_poi=250.0, dt=0.1)


@dataclass
class Run:
    counts: np.ndarray  # spikes per neuron per trial: (trials, n)
    t_run_s: float

    @property
    def rates(self) -> np.ndarray:
        """Mean rate (Hz) per neuron over trials."""
        return self.counts.mean(axis=0) / self.t_run_s

    @property
    def rate_se(self) -> np.ndarray:
        n = self.counts.shape[0]
        return self.counts.std(axis=0, ddof=1) / self.t_run_s / np.sqrt(n) if n > 1 else np.zeros(self.counts.shape[1])


@nb.njit(cache=True)
def _trial(indptr, indices, weights, stim_idx, stim_rate_hz, n_steps, seed, p, record_idx, rec_times, forced, v_rec_idx, v_rec):
    v_0, v_rst, v_th, t_mbr, tau, t_rfc, t_dly, w_syn, f_poi, dt = p
    n = indptr.shape[0] - 1
    a_m = np.exp(-dt / t_mbr)
    a_g = np.exp(-dt / tau)
    b = (tau / (tau - t_mbr)) * (a_g - a_m)  # contribution of g to v over one step (exact solution)
    d = int(round(t_dly / dt))
    rfc_steps = int(round(t_rfc / dt))
    np.random.seed(seed)
    v = np.full(n, v_0)
    g = np.zeros(n)
    last = np.full(n, -10_000_000, dtype=np.int64)
    rfc = np.full(n, rfc_steps, dtype=np.int64)
    for k in range(stim_idx.shape[0]):
        rfc[stim_idx[k]] = 0
    ring = np.zeros((d, n))
    counts = np.zeros(n, dtype=np.int64)
    spiking = np.zeros(n, dtype=np.int64)
    p_poi = stim_rate_hz * dt / 1000.0
    w_poi = w_syn * f_poi
    n_rec = record_idx.shape[0]
    rec_n = np.zeros(n_rec, dtype=np.int64)
    for t in range(n_steps):
        # 1. state update (not while refractory)
        for i in range(n):
            if t - last[i] >= rfc[i]:
                v[i] = v_0 + (v[i] - v_0) * a_m + g[i] * b
                g[i] = g[i] * a_g
        # forced spikes (tests, and later the game coupling): (step, neuron) pairs, sorted by step
        for f in range(forced.shape[0]):
            if forced[f, 0] == t and t - last[forced[f, 1]] >= rfc[forced[f, 1]]:
                v[forced[f, 1]] = 1e9
        # 2. threshold
        ns = 0
        for i in range(n):
            if v[i] > v_th and t - last[i] >= rfc[i]:
                spiking[ns] = i
                ns += 1
        # 3. synapses: delayed spikes arrive, new spikes leave; Poisson input to the stimulated neurons
        slot = t % d
        row = ring[slot]
        for i in range(n):
            if row[i] != 0.0:
                if t - last[i] >= rfc[i]:  # input reaching a refractory neuron is lost (as in Brian2, see the docstring)
                    g[i] += row[i]
                row[i] = 0.0
        for s in range(ns):
            i = spiking[s]
            for k in range(indptr[i], indptr[i + 1]):
                row[indices[k]] += weights[k] * w_syn
        for k in range(stim_idx.shape[0]):
            if np.random.random() < p_poi:
                v[stim_idx[k]] += w_poi
        # 4. reset
        for s in range(ns):
            i = spiking[s]
            v[i] = v_rst
            g[i] = 0.0
            last[i] = t
            counts[i] += 1
        for r in range(v_rec_idx.shape[0]):  # membrane potential at the end of the step (tests)
            v_rec[r, t] = v[v_rec_idx[r]]
        # optional spike times of a few neurons (for plots)
        for r in range(n_rec):
            if last[record_idx[r]] == t and rec_n[r] < rec_times.shape[1]:
                rec_times[r, rec_n[r]] = t
                rec_n[r] += 1
    return counts, rec_n


def simulate(con: Connectome, stim: dict, *, t_run_s: float = 1.0, trials: int = 30, seed: int = 0,
             silence=(), record=(), params: dict | None = None, forced=None, v_record=()) -> tuple[Run, list]:
    """stim: {rate_hz: model indices}. Returns spike counts, and spike times (s) of the `record` neurons per trial."""
    p = dict(PARAMS, **(params or {}))
    weights = con.weights.copy()
    for i in silence:
        weights[con.indptr[i]:con.indptr[i + 1]] = 0.0
    p_tuple = tuple(float(p[k]) for k in ("v_0", "v_rst", "v_th", "t_mbr", "tau", "t_rfc", "t_dly", "w_syn", "f_poi", "dt"))
    n_steps = int(round(t_run_s * 1000 / p["dt"]))
    record = np.asarray(record, dtype=np.int64)
    counts, times = [], []
    rates = sorted(stim) or [0.0]
    assert len(rates) == 1, "one stimulation rate per run is implemented (Shiu's r_poi); r_poi2 is not needed yet"
    stim_idx = np.asarray(stim.get(rates[0], []), dtype=np.int64)
    forced = np.zeros((0, 2), np.int64) if forced is None else np.asarray(forced, np.int64)
    v_record = np.asarray(v_record, dtype=np.int64)
    self_v = []
    for trial in range(trials):
        rec = np.zeros((len(record), int(t_run_s * 1000)), dtype=np.int64)
        v_rec = np.zeros((len(v_record), n_steps))
        c, rec_n = _trial(con.indptr, con.indices, weights, stim_idx, float(rates[0]), n_steps, seed + trial, p_tuple, record, rec,
                          forced, v_record, v_rec)
        self_v.append(v_rec)
        counts.append(c)
        times.append([rec[r, :rec_n[r]] * p["dt"] / 1000.0 for r in range(len(record))])
    run = Run(np.array(counts), t_run_s)
    run.v = self_v  # membrane potential traces of the v_record neurons, per trial (mV)
    return run, times


# ---- stepping version, for the closed loop -------------------------------------------------------------------------
# Same model, same order of operations as _trial, but the state lives in a Brain object so the simulation can advance
# one game tic at a time, with a separate Poisson rate per stimulated neuron. tests/test_closed_loop.py checks that
# advancing in chunks with constant rates gives exactly the same spikes as _trial.

@nb.njit(cache=True)
def _seed(seed):
    np.random.seed(seed)


@nb.njit(cache=True)
def _advance(indptr, indices, weights, v, g, last, rfc, ring, t0, n_steps, stim_idx, stim_p, p, counts_out, spiked_flag):
    v_0, v_rst, v_th, t_mbr, tau, t_rfc, t_dly, w_syn, f_poi, dt = p
    n = indptr.shape[0] - 1
    a_m = np.exp(-dt / t_mbr)
    a_g = np.exp(-dt / tau)
    b = (tau / (tau - t_mbr)) * (a_g - a_m)
    d = ring.shape[0]
    w_poi = w_syn * f_poi
    spiking = np.zeros(n, dtype=np.int64)
    for t in range(t0, t0 + n_steps):
        for i in range(n):
            if t - last[i] >= rfc[i]:
                v[i] = v_0 + (v[i] - v_0) * a_m + g[i] * b
                g[i] = g[i] * a_g
        ns = 0
        for i in range(n):
            if v[i] > v_th and t - last[i] >= rfc[i]:
                spiking[ns] = i
                ns += 1
        row = ring[t % d]
        for i in range(n):
            if row[i] != 0.0:
                if t - last[i] >= rfc[i]:
                    g[i] += row[i]
                row[i] = 0.0
        for s in range(ns):
            i = spiking[s]
            for k in range(indptr[i], indptr[i + 1]):
                row[indices[k]] += weights[k] * w_syn
        for k in range(stim_idx.shape[0]):
            if np.random.random() < stim_p[k]:
                v[stim_idx[k]] += w_poi
        for s in range(ns):
            i = spiking[s]
            v[i] = v_rst
            g[i] = 0.0
            last[i] = t
            counts_out[i] += 1
            spiked_flag[i] = 1


class Brain:
    """The whole-brain model as a stepping simulator. stim_idx: neurons that can receive Poisson input (no refractory
    period, like Shiu's Poisson targets). advance(seconds, rates_hz) runs the model with one rate per stim neuron and
    returns the spike count of every neuron during that time."""

    def __init__(self, con: Connectome, stim_idx, *, seed: int = 0, silence=(), params: dict | None = None):
        p = dict(PARAMS, **(params or {}))
        self.p = tuple(float(p[k]) for k in ("v_0", "v_rst", "v_th", "t_mbr", "tau", "t_rfc", "t_dly", "w_syn", "f_poi", "dt"))
        self.dt_ms = p["dt"]
        self.con = con
        self.weights = con.weights.copy()
        for i in silence:
            self.weights[con.indptr[i]:con.indptr[i + 1]] = 0.0
        n = con.n
        self.stim_idx = np.asarray(stim_idx, dtype=np.int64)
        self.v = np.full(n, p["v_0"])
        self.g = np.zeros(n)
        self.last = np.full(n, -10_000_000, dtype=np.int64)
        self.rfc = np.full(n, int(round(p["t_rfc"] / p["dt"])), dtype=np.int64)
        self.rfc[self.stim_idx] = 0
        self.ring = np.zeros((int(round(p["t_dly"] / p["dt"])), n))
        self.t = 0  # steps done
        self.elapsed_ms = 0.0
        _seed(seed)  # numba's own generator; one Brain at a time (the closed loop never interleaves two)

    def advance(self, ms: float, rates_hz) -> tuple[np.ndarray, np.ndarray]:
        """Run until `ms` more milliseconds have elapsed (steps aligned to the total time, so no drift).
        Returns (spike counts of every neuron, 0/1 flag of neurons that spiked) for this interval."""
        self.elapsed_ms += ms
        n_steps = int(round(self.elapsed_ms / self.dt_ms)) - self.t
        stim_p = np.asarray(rates_hz, dtype=np.float64) * self.dt_ms / 1000.0
        counts = np.zeros(self.con.n, dtype=np.int64)
        flag = np.zeros(self.con.n, dtype=np.int8)
        _advance(self.con.indptr, self.con.indices, self.weights, self.v, self.g, self.last, self.rfc, self.ring,
                 self.t, n_steps, self.stim_idx, stim_p, self.p, counts, flag)
        self.t += n_steps
        return counts, flag
