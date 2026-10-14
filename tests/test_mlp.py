import json

import numpy as np
import pytest

from compute.export import ASSETS, build_golden, build_mlp_json
from compute.mlp import forward, softmax
from compute.mlp_presets import xor_preset


def test_forward_matches_hand_calculation():
    # 2 -> 2 -> 1 network small enough to check by hand. Output layer is softmax of one value = 1.0.
    weights = [[[1.0, -1.0], [0.5, 0.5]], [[2.0, 1.0]]]
    biases = [[0.0, 0.1], [0.0]]
    r = forward(weights, biases, [1.0, 0.5], "relu")
    assert r["pre_activations"][0] == pytest.approx([0.5, 0.85])
    assert r["activations"][1] == pytest.approx([0.5, 0.85])
    assert r["pre_activations"][1] == pytest.approx([1.85])
    assert r["activations"][2] == pytest.approx([1.0])
    assert r["contributions"][0][0] == pytest.approx([1.0, -0.5])  # w_ij * a_j for neuron 0


def test_softmax_is_a_distribution_and_stable():
    p = softmax(np.array([1000.0, 1001.0, 999.0]))
    assert p.sum() == pytest.approx(1.0)
    assert np.all(np.isfinite(p))


def test_xor_preset_computes_xor():
    p = xor_preset()
    for x, expect_different in [([0, 0], False), ([1, 0], True), ([0, 1], True), ([1, 1], False)]:
        out = forward(p["weights"], p["biases"], x, "tanh")["activations"][-1]
        assert (out[0] > 0.5) == expect_different, (x, out)


def test_committed_golden_is_up_to_date():
    # Fails if someone edits the model or presets without re-running `python -m compute.export`.
    expected = json.loads(json.dumps(build_golden(build_mlp_json())))
    assert json.loads((ASSETS / "golden.json").read_text(encoding="utf-8")) == expected
