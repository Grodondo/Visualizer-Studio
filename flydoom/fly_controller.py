"""Controllers A, B, D: the validated whole-brain fly model (Milestone 2) closing the loop with the game.

Every choice below was written down BEFORE any test seed was played. The chain, per game tic (1/35 s):

  pixels --(1)--> looming eye --(2)--> Poisson input to LC4 + LPLC2 --(3)--> whole-brain LIF model, 1/35 s
         --(4)--> spikes of the steering neurons DNa01 + DNa02 --(5)--> key

(1) MEASURED-NOTHING / ENGINEERED: flydoom.eye.LoomingEye, the SAME object class and the same 4 numbers the reflex (C) gets.
(2) ENGINEERED encoder. The game's left half is the fly's left eye: it drives the LEFT LC4 and LPLC2 cells (FlyWire "side"
    label), the right half the right ones. Both cell types get the same rate:  rate = min(RATE_MAX, GAIN x expansion (deg/s)).
    Simplification, said out loud: in flies LC4 signals approach speed and LPLC2 size (Ache et al. 2019); here both get the
    expansion signal. RATE_MAX = 200 Hz keeps input inside the range validated in Milestone 2 (50-200 Hz).
    GAIN is the ONLY tuned number, chosen on the calibration seeds (as the reflex threshold was).
(3) SIMULATED: Shiu et al. (2024) model on FlyWire v783, unchanged (flydoom/lif.py, validated against Brian2).
    Poisson targets have no refractory period, as in Shiu's code. The model's random numbers are seeded from the episode seed.
(4)+(5) ENGINEERED decoder, direction from the literature: DNa01/DNa02 depolarize before steering IPSILATERALLY, and steering
    is predicted by the right-left difference in DNa02 rate (Rayshubskiy et al. 2025, eLife 13:RP102230). Rates are spike counts
    of DNa01+DNa02 per side, low-pass filtered (time constant TAU_MS). Key = LEFT if left - right > THRESHOLD_HZ, RIGHT if
    right - left > THRESHOLD_HZ, else none. In take_cover the player can only step sideways, so "steer left" becomes
    "step left": an engineering mapping (a walking fly turns; this player cannot).
    The giant fibers (DNp01) are recorded but do NOT drive any key: take_cover has no jump.
Controls with the identical encoder, decoder and GAIN:
    B  shuffled wiring: postsynaptic partners permuted (degree-preserving, flydoom.validate_model.shuffled)
    D  giant fibers silenced: tests whether the escape neurons shape the steering output through the network
"""
from __future__ import annotations

import numpy as np

from . import lif
from . import model_data as md
from .controllers import Controller
from .eye import LoomingEye
from .game import TICRATE
from .validate_model import cells, shuffled

RATE_MAX = 200.0  # Hz
TAU_MS = 100.0  # decoder low-pass
THRESHOLD_HZ = 10.0  # decoder dead band
SEED_OFFSET = 77_000  # model randomness seed = episode seed + this
STEER = {"DNa01", "DNa02"}
RECORD = ("LC4", "LPLC2", "DNa01", "DNa02", "DNp01")

_CACHE: dict = {}


def connectome(kind: str) -> md.Connectome:
    if kind not in _CACHE:
        real = _CACHE.get("real") or md.load("783")
        _CACHE["real"] = real
        _CACHE["shuffled"] = shuffled(real)
    return _CACHE[kind]


def groups(con: md.Connectome) -> dict:
    """Model indices of every recorded cell type, per side."""
    out = {}
    for t in RECORD:
        for side in ("left", "right"):
            out[(t, side)] = con.index_of(cells([t], side))
    return out


class FlyBrain(Controller):
    """A (wiring="real"), B (wiring="shuffled"), D (wiring="real", silence_giant_fibers=True)."""

    uses = "the looming eye's 4 numbers -> whole-brain model -> steering neurons"

    def __init__(self, gain: float = 2.0, wiring: str = "real", silence_giant_fibers: bool = False, record_all: bool = False):
        self.gain, self.wiring, self.silence_gf, self.record_all = gain, wiring, silence_giant_fibers, record_all
        self.name = {("real", False): "fly", ("shuffled", False): "fly_shuffled", ("real", True): "fly_no_giant_fiber"}[(wiring, silence_giant_fibers)]
        self.eye = LoomingEye()
        con = connectome(wiring)
        self.con = con
        self.g = groups(connectome("real"))  # neuron indices are the same in the shuffled copy
        self.inputs = {s: np.concatenate([self.g[("LC4", s)], self.g[("LPLC2", s)]]) for s in ("left", "right")}
        self.stim_idx = np.concatenate([self.inputs["left"], self.inputs["right"]])
        self.steer = {s: np.concatenate([self.g[(t, s)] for t in STEER]) for s in ("left", "right")}

    def reset(self, seed: int = 0):
        self.eye.reset()
        silence = np.concatenate([self.g[("DNp01", "left")], self.g[("DNp01", "right")]]) if self.silence_gf else ()
        self.brain = lif.Brain(self.con, self.stim_idx, seed=seed + SEED_OFFSET, silence=silence)
        self.rate = {"left": 0.0, "right": 0.0}
        self.current = "none"
        self.log = {"rates_in": [], "counts": {f"{t}_{s}": [] for t, s in self.g}, "steer_hz": [], "active": []}

    def act(self, rgb):
        out = self.eye.see(rgb)
        self.last_eye = out.as_list()
        exp_l, exp_r = out.expansion_dps
        r_l, r_r = (min(RATE_MAX, self.gain * e) for e in (exp_l, exp_r))
        rates = np.r_[np.full(len(self.inputs["left"]), r_l), np.full(len(self.inputs["right"]), r_r)]
        ms = 1000.0 / TICRATE
        counts, flag = self.brain.advance(ms, rates)
        a = np.exp(-ms / TAU_MS)
        for s in ("left", "right"):
            hz = counts[self.steer[s]].sum() / (ms / 1000.0)
            self.rate[s] = a * self.rate[s] + (1 - a) * hz
        diff = self.rate["left"] - self.rate["right"]
        self.current = "left" if diff > THRESHOLD_HZ else ("right" if diff < -THRESHOLD_HZ else "none")
        self.log["rates_in"].append([round(r_l, 2), round(r_r, 2)])
        for (t, s), idx in self.g.items():
            self.log["counts"][f"{t}_{s}"].append(int(counts[idx].sum()))
        self.log["steer_hz"].append([round(self.rate["left"], 2), round(self.rate["right"], 2)])
        self.log["active"].append(np.flatnonzero(flag).astype(np.int32) if self.record_all else int(flag.sum()))
        return self.current
