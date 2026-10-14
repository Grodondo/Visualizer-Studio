"""A small neural network trained on real handwritten digits, exported for the Manim video.

Dataset: Optical Recognition of Handwritten Digits (UCI Machine Learning Repository)
  E. Alpaydin and C. Kaynak (1998), https://doi.org/10.24432/C50P49, licence CC BY 4.0.
  5,620 digits written by 43 people (30 wrote the training set, 13 different people the test set).
  Each digit is an 8x8 image; every value is the amount of ink in a 4x4 block of a 32x32 scan, an integer from 0 to 16.

Run from the project root:  uv run python -m compute.digits
  downloads the two data files once into assets/data/ (about 0.8 MB, not tracked by Git),
  trains a 64 -> 16 -> 10 network (ReLU hidden layer, softmax output) with a fixed seed,
  and writes assets/digits/digits_trace.json, the only file the video reads.

The forward pass of the exported examples uses compute.mlp.forward, the same model code as the other videos.
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np

from compute.mlp import ACTIVATIONS, forward

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "assets" / "data"
TRACE_PATH = ROOT / "assets" / "digits" / "digits_trace.json"
BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/"
FILES = {"train": "optdigits.tra", "test": "optdigits.tes"}
EXPECTED_ROWS = {"train": 3823, "test": 1797}

# Training choices (all fixed, so the export is reproducible).
LAYER_SIZES = (64, 16, 10)
SEED = 0
EPOCHS = 30
BATCH = 32
LEARNING_RATE = 2e-3
SHOWCASE_CLASS = 3  # the video follows the first test digit of this class that the network reads confidently
SHOWCASE_MIN_PROBABILITY = 0.99
ACTIVATION_SAMPLES = 49  # points of the sigmoid / tanh / ReLU curves, x from -6 to 6
CURVE_LIMIT = 6.0


def download(split: str) -> Path:
    path = DATA_DIR / FILES[split]
    if not path.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(BASE_URL + FILES[split], timeout=60) as response:  # noqa: S310 (fixed https URL)
            path.write_bytes(response.read())
    return path


def load_split(split: str) -> tuple[np.ndarray, np.ndarray]:
    """Pixels as raw integers 0..16 (n x 64) and labels 0..9."""
    data = np.loadtxt(download(split), delimiter=",", dtype=int)
    assert data.shape == (EXPECTED_ROWS[split], 65), data.shape
    return data[:, :64], data[:, 64]


def softmax_rows(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def predict(params, x: np.ndarray) -> np.ndarray:
    w1, b1, w2, b2 = params
    return softmax_rows(np.maximum(x @ w1.T + b1, 0.0) @ w2.T + b2)


def train(x_train, y_train, x_test, y_test):
    """Plain numpy training: minibatch gradient descent with Adam on cross-entropy."""
    rng = np.random.default_rng(SEED)
    n_in, n_hidden, n_out = LAYER_SIZES
    params = [
        rng.normal(0.0, np.sqrt(2.0 / n_in), (n_hidden, n_in)),
        np.zeros(n_hidden),
        rng.normal(0.0, np.sqrt(2.0 / n_hidden), (n_out, n_hidden)),
        np.zeros(n_out),
    ]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    onehot = np.eye(n_out)[y_train]
    history, step = [], 0
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(len(x_train))
        for start in range(0, len(order), BATCH):
            idx = order[start : start + BATCH]
            x, t = x_train[idx], onehot[idx]
            w1, b1, w2, b2 = params
            z1 = x @ w1.T + b1
            a1 = np.maximum(z1, 0.0)
            p = softmax_rows(a1 @ w2.T + b2)
            dz2 = (p - t) / len(idx)
            dz1 = (dz2 @ w2) * (z1 > 0)
            grads = [dz1.T @ x, dz1.sum(axis=0), dz2.T @ a1, dz2.sum(axis=0)]
            step += 1
            for i, g in enumerate(grads):
                m[i] = 0.9 * m[i] + 0.1 * g
                v[i] = 0.999 * v[i] + 0.001 * g * g
                params[i] -= LEARNING_RATE * (m[i] / (1 - 0.9**step)) / (np.sqrt(v[i] / (1 - 0.999**step)) + 1e-8)
        train_p = predict(params, x_train)
        loss = float(-np.log(train_p[np.arange(len(y_train)), y_train] + 1e-12).mean())
        accuracy = float((predict(params, x_test).argmax(axis=1) == y_test).mean())
        history.append({"epoch": epoch, "train_loss": loss, "test_accuracy": accuracy})
    return params, history


def example_record(role: str, index: int, pixels_raw, label: int, weights, biases) -> dict:
    trace = forward(weights, biases, (pixels_raw / 16.0).tolist(), "relu")
    probs = trace["activations"][2]
    return {
        "role": role,
        "test_index": int(index),
        "label": int(label),
        "pixels": [int(v) for v in pixels_raw],  # raw ink amounts 0..16
        "z1": trace["pre_activations"][0],
        "a1": trace["activations"][1],
        "z2": trace["pre_activations"][1],
        "probabilities": probs,
        "predicted": int(np.argmax(probs)),
    }


def build_trace() -> dict:
    x_train_raw, y_train = load_split("train")
    x_test_raw, y_test = load_split("test")
    params, history = train(x_train_raw / 16.0, y_train, x_test_raw / 16.0, y_test)
    w1, b1, w2, b2 = params
    weights = [w1.tolist(), w2.tolist()]
    biases = [b1.tolist(), b2.tolist()]

    test_probs = predict(params, x_test_raw / 16.0)
    predicted = test_probs.argmax(axis=1)
    confident = np.where((y_test == SHOWCASE_CLASS) & (predicted == y_test) & (test_probs.max(axis=1) > SHOWCASE_MIN_PROBABILITY))[0]
    wrong = np.where(predicted != y_test)[0]
    showcase = example_record("showcase", confident[0], x_test_raw[confident[0]], y_test[confident[0]], weights, biases)
    mistake = example_record("mistake", wrong[0], x_test_raw[wrong[0]], y_test[wrong[0]], weights, biases)
    focus = int(np.argmax(showcase["a1"]))  # the hidden neuron most excited by the showcase digit
    quiet = int(np.argmin(showcase["z1"]))  # the one pushed furthest below zero: ReLU switches it off

    def neuron_record(index: int) -> dict:
        z = showcase["z1"][index]
        return {"neuron": index, "z": z, "values": {name: float(f(z)) for name, f in ACTIVATIONS.items()}}  # what each function would output

    grid = np.linspace(-CURVE_LIMIT, CURVE_LIMIT, ACTIVATION_SAMPLES)
    return {
        "schema": 1,
        "dataset": {
            "name": "Optical Recognition of Handwritten Digits",
            "source": "UCI Machine Learning Repository",
            "citation": "Alpaydin, E. & Kaynak, C. (1998). doi:10.24432/C50P49",
            "license": "CC BY 4.0",
            "image": "8x8; each value is the amount of ink (0..16) in a 4x4 block of a 32x32 scan",
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
        },
        "provenance": {"data": "measured", "weights": "learned (simulated training)", "forward_pass": "computed"},
        "layer_sizes": list(LAYER_SIZES),
        "activation": "relu",
        "training": {
            "seed": SEED,
            "epochs": EPOCHS,
            "batch_size": BATCH,
            "learning_rate": LEARNING_RATE,
            "optimizer": "Adam",
            "history": history,
            "test_accuracy": float((predicted == y_test).mean()),
            "test_errors": int((predicted != y_test).sum()),
        },
        "weights": weights,
        "biases": biases,
        "focus": neuron_record(focus),
        "quiet": neuron_record(quiet),
        "examples": [showcase, mistake],
        "activation_curves": {"x": grid.tolist(), **{name: [float(f(v)) for v in grid] for name, f in ACTIVATIONS.items()}},
    }


def write_trace() -> None:
    trace = build_trace()
    TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRACE_PATH.write_text(json.dumps(trace, separators=(",", ":")), encoding="utf-8")
    t = trace["training"]
    print(f"wrote {TRACE_PATH.relative_to(ROOT)} ({TRACE_PATH.stat().st_size // 1024} KB); test accuracy {t['test_accuracy']:.4f}, {t['test_errors']} errors")
    for ex in trace["examples"]:
        print(f"  {ex['role']}: test digit #{ex['test_index']}, label {ex['label']}, predicted {ex['predicted']}, p = {max(ex['probabilities']):.5f}")
    print(f"  focus hidden neuron: {trace['focus']['neuron']} (z = {trace['focus']['z']:.3f}); quiet neuron: {trace['quiet']['neuron']} (z = {trace['quiet']['z']:.3f})")


if __name__ == "__main__":
    write_trace()
