"""Checks for the whole-brain LIF port (Milestone 2). Needs the optional group:  uv run --group flydoom pytest -q tests/test_fly_model.py"""
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("numba")

from flydoom import lif  # noqa: E402
from flydoom.model_data import Connectome  # noqa: E402
from flydoom.validate_model import shuffled  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
VALIDATION = ROOT / "assets" / "flydoom" / "model_validation.json"
BRIAN = ROOT / "assets" / "flydoom" / "brian_check.json"
P = lif.PARAMS


def chain(weights=(30.0,)):
    """Neuron 0 = input (forced spikes); neuron 1 = target with the given synapse count from 0."""
    return Connectome("toy", np.arange(2), np.array([0, 1, 1]), np.array([1]), np.array(weights, float))


def run(con, forced, t_ms=30.0, record=(1,)):
    r, _ = lif.simulate(con, {0.0: [0]}, t_run_s=t_ms / 1000, trials=1, forced=forced, v_record=record)
    return r


# ---- the equations ---------------------------------------------------------------------------------

def test_one_input_spike_follows_the_exact_solution():
    """v(t) - v_0 = w * tau / (tau - t_mbr) * (exp(-t/tau) - exp(-t/t_mbr)) after the synaptic input arrives."""
    r = run(chain(), [[10, 0]])
    v = r.v[0][0]
    arrive = 10 + int(round(P["t_dly"] / P["dt"]))  # the input reaches g at the end of this step
    w = 30 * P["w_syn"]
    k = np.arange(1, 100)
    t = k * P["dt"]
    expected = P["v_0"] + w * P["tau"] / (P["tau"] - P["t_mbr"]) * (np.exp(-t / P["tau"]) - np.exp(-t / P["t_mbr"]))
    assert np.allclose(v[arrive + k], expected, atol=1e-9)
    assert np.all(v[: arrive + 1] == P["v_0"])


def test_synaptic_delay_is_1_8_ms():
    r = run(chain(), [[50, 0]])
    first_change = int(np.argmax(r.v[0][0] != P["v_0"]))
    assert first_change == 50 + 18 + 1


def test_input_to_a_refractory_neuron_is_lost():
    """The rule found by replaying Brian2 spikes: g is '(unless refractory)', so input during refractoriness has no effect.
    Input 0 makes neuron 2 spike; input 1 arrives inside (must change nothing) or just after (must change v) the 2.2 ms."""
    con = Connectome("toy", np.arange(3), np.array([0, 1, 2, 2]), np.array([2, 2]), np.array([200.0, 10.0]))
    sim = lambda forced: lif.simulate(con, {0.0: [0, 1]}, t_run_s=0.03, trials=1, forced=forced, v_record=[2])[0]  # noqa: E731
    alone = sim([[10, 0]])
    v0 = alone.v[0][0]
    assert alone.counts[0][2] == 1
    spike = int(np.argmax((v0[:-1] != P["v_0"]) & (v0[1:] == P["v_rst"]))) + 1  # the step v was reset
    delay, rfc = int(round(P["t_dly"] / P["dt"])), int(round(P["t_rfc"] / P["dt"]))
    inside = sim([[10, 0], [spike + 5 - delay, 1]])
    after = sim([[10, 0], [spike + rfc + 2 - delay, 1]])
    assert np.array_equal(inside.v[0][0], v0)
    assert not np.array_equal(after.v[0][0], v0)


def test_silencing_removes_outgoing_effect():
    r, _ = lif.simulate(chain(), {0.0: [0]}, t_run_s=0.03, trials=1, forced=[[10, 0]], silence=[0], v_record=[1])
    assert np.all(r.v[0][0] == P["v_0"])


def test_same_seed_same_spikes_different_seed_differs():
    con = Connectome("toy", np.arange(2), np.array([0, 1, 1]), np.array([1]), np.array([30.0]))
    a, _ = lif.simulate(con, {200.0: [0]}, t_run_s=0.5, trials=2, seed=4)
    b, _ = lif.simulate(con, {200.0: [0]}, t_run_s=0.5, trials=2, seed=4)
    c, _ = lif.simulate(con, {200.0: [0]}, t_run_s=0.5, trials=2, seed=5)
    assert np.array_equal(a.counts, b.counts) and not np.array_equal(a.counts, c.counts)


def test_shuffle_keeps_every_neurons_degree():
    rng = np.random.default_rng(0)
    pre = np.sort(rng.integers(0, 50, 400))
    post = rng.integers(0, 50, 400)
    indptr = np.searchsorted(pre, np.arange(51))
    con = Connectome("toy", np.arange(50), indptr, post, rng.normal(size=400))
    s = shuffled(con, seed=1)
    assert np.array_equal(np.bincount(s.indices, minlength=50), np.bincount(con.indices, minlength=50))  # in-degree
    assert np.array_equal(s.indptr, con.indptr) and np.array_equal(s.weights, con.weights)  # out-degree and weights
    assert not np.array_equal(s.indices, con.indices)


# ---- the recorded validation --------------------------------------------------------------------------

@pytest.mark.skipif(not VALIDATION.exists(), reason="run uv run --group flydoom python -m flydoom.validate_model")
def test_port_reproduces_shius_published_output():
    a = json.loads(VALIDATION.read_text(encoding="utf-8"))["part_a"]
    assert a["pearson_rates"] > 0.999
    assert abs(a["active_ours"] - a["active_reference"]) <= 10
    assert abs(a["mn9_hz"]["ours"] - a["mn9_hz"]["reference"]) / a["mn9_hz"]["reference"] < 0.05


@pytest.mark.skipif(not VALIDATION.exists(), reason="run uv run --group flydoom python -m flydoom.validate_model")
def test_v783_effects_need_the_real_wiring():
    b = json.loads(VALIDATION.read_text(encoding="utf-8"))["part_b"]
    sug = b["sugar_200hz"]
    assert sug["real"]["MN9_right"]["mean_hz"] > 50 and sug["shuffled"]["MN9_right"]["mean_hz"] == 0
    for rate, v in b["looming_right"].items():
        real, null = v["real"], v["shuffled"]
        assert real["DNp01_right"]["mean_hz"] > real["DNp01_left"]["mean_hz"] > 0  # right looming: right giant fiber strongest
        assert all(null[k]["mean_hz"] == 0 for k in null if k != "active_neurons")
        # recorded finding (to be interpreted with care in Milestone 3): right looming drives the LEFT steering neurons only
        assert real["DNa01_right"]["mean_hz"] == 0 and real["DNa02_right"]["mean_hz"] == 0
        assert real["DNa01_left"]["mean_hz"] > 0 and real["DNa02_left"]["mean_hz"] > 0
    giant = [b["looming_right"][r]["real"]["DNp01_right"]["mean_hz"] for r in sorted(b["looming_right"], key=float)]
    assert giant == sorted(giant)  # stronger looming input -> faster giant fiber
    assert b["no_stimulus"]["active_neurons"] == 0


@pytest.mark.skipif(not BRIAN.exists(), reason="optional: see flydoom/brian_check.py")
def test_brian2_cross_check():
    c = json.loads(BRIAN.read_text(encoding="utf-8"))
    assert c["replay"]["target_spikes_brian"] == c["replay"]["target_spikes_port"] and c["replay"]["max_abs_dv_mV"] < 1e-6
    assert c["sugar"]["pearson_rates"] > 0.999
    assert c["sugar"]["neurons_z_over_3"] <= 0.03 * c["sugar"]["neurons_over_5hz"]
