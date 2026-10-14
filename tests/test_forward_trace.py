"""The video's numbers: the exported trace must equal the model, and the story's numbers must be real."""
import json
import math

import pytest

from compute.export import ASSETS, VIDEO_INPUT, build_forward_trace, build_mlp_json

TRACE_PATH = ASSETS / "forward_trace.json"


def load() -> dict:
    return json.loads(TRACE_PATH.read_text(encoding="utf-8"))


def test_committed_trace_is_up_to_date():
    # Fails if the model or presets change without re-running `uv run python -m compute.export`.
    assert load() == json.loads(json.dumps(build_forward_trace(build_mlp_json())))


def test_trace_matches_an_independent_scalar_computation():
    # Plain Python loops with math.tanh/exp share no code with compute.mlp (numpy).
    # Both are IEEE doubles, so they may differ only by libm rounding: 1e-12 is generous.
    t = load()
    a = list(t["input"])
    n_layers = len(t["weights"])
    for layer, (weights, biases) in enumerate(zip(t["weights"], t["biases"]), start=1):
        z = [sum(w * x for w, x in zip(row, a)) + b for row, b in zip(weights, biases)]
        if layer == n_layers:
            e = [math.exp(v - max(z)) for v in z]
            a = [v / sum(e) for v in e]
        else:
            a = [math.tanh(v) for v in z]
        assert t["pre_activations"][layer - 1] == pytest.approx(z, abs=1e-12)
        assert t["activations"][layer] == pytest.approx(a, abs=1e-12)


def test_contributions_are_weight_times_incoming_activation():
    t = load()
    for layer, matrix in enumerate(t["contributions"], start=1):
        for to, row in enumerate(matrix):
            for src, value in enumerate(row):
                assert value == pytest.approx(t["weights"][layer - 1][to][src] * t["activations"][layer - 1][src], abs=1e-12)


def test_input_and_preset_are_the_documented_ones():
    t = load()
    assert t["input"] == VIDEO_INPUT == [0.8, 0.3]
    assert t["preset"] == "xor" and t["activation"] == "tanh"


def test_numbers_quoted_in_the_storyboard():
    # The OR neuron of hidden layer 1 is the one the video zooms into: 4.0*0.8 + 4.0*0.3 - 2.0 = 2.4.
    t = load()
    assert t["neuron_labels"][1][0] == "OR"
    assert t["weights"][0][0] == [4, 4] and t["biases"][0][0] == -2
    assert t["contributions"][0][0] == pytest.approx([3.2, 1.2])
    assert t["pre_activations"][0][0] == pytest.approx(2.4)
    assert t["activations"][1][0] == pytest.approx(math.tanh(2.4))


def test_prediction_is_derived_from_the_output():
    t = load()
    out = t["activations"][-1]
    assert sum(out) == pytest.approx(1.0)
    assert t["prediction"]["index"] == out.index(max(out))
    assert t["prediction"]["label"] == t["neuron_labels"][-1][t["prediction"]["index"]]
    assert t["prediction"]["probability"] == max(out)
    # Saturated but not certain: the reason the video prints p to five decimals instead of 1.00.
    assert 0.99998 <= t["prediction"]["probability"] < 1.0
