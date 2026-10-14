import { describe, expect, it } from "vitest";
// The golden trace is written by `uv run python -m compute.export` (see compute/mlp.py).
import golden from "../../../../assets/mlp/golden.json";
import mlp from "../../../../assets/mlp/mlp.json";
import { forward, softmax, type ActivationName } from "./model";

const TOL: number = golden.tolerance_abs;

function expectClose(actual: number[], expected: number[], label: string): void {
  expect(actual.length, label).toBe(expected.length);
  actual.forEach((v, i) => expect(Math.abs(v - expected[i]), `${label}[${i}]`).toBeLessThanOrEqual(TOL));
}

describe("TypeScript forward pass agrees with the Python golden trace", () => {
  it("has golden cases to compare against", () => {
    expect(golden.cases.length).toBeGreaterThan(0);
  });

  for (const c of golden.cases) {
    it(`${c.preset} / ${c.activation} / input ${JSON.stringify(c.input)}`, () => {
      const preset = mlp.presets.find((p) => p.id === c.preset)!;
      const r = forward(preset.weights, preset.biases, c.input, c.activation as ActivationName);

      // Intermediate values first, so a failure points at the layer that diverged.
      r.preActivations.forEach((z, l) => expectClose(z, c.pre_activations[l], `z layer ${l + 1}`));
      r.activations.forEach((a, l) => expectClose(a, c.activations[l], `a layer ${l}`));
      r.contributions.forEach((m, l) => m.forEach((row, i) => expectClose(row, c.contributions[l][i], `w*a layer ${l + 1} neuron ${i}`)));
    });
  }
});

describe("softmax", () => {
  it("sums to 1 and survives large inputs", () => {
    const p = softmax([1000, 1001, 999]);
    expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
    expect(p.every(Number.isFinite)).toBe(true);
  });
});
