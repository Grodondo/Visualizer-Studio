/**
 * Forward pass of a small MLP. TypeScript port of compute/mlp.py: keep the two in step.
 * Index conventions are documented there; agreement is checked by model.test.ts against assets/mlp/golden.json.
 */
export type ActivationName = "tanh" | "relu" | "sigmoid";

export const ACTIVATION_LABELS: Record<ActivationName, string> = {
  tanh: "tanh",
  relu: "ReLU",
  sigmoid: "sigmoid",
};

const ACTIVATIONS: Record<ActivationName, (z: number) => number> = {
  tanh: Math.tanh,
  relu: (z) => Math.max(z, 0),
  sigmoid: (z) => 1 / (1 + Math.exp(-z)),
};

export interface ForwardResult {
  /** preActivations[l-1][i] is z of neuron i in layer l. */
  preActivations: number[][];
  /** activations[l][i] is a of neuron i in layer l; activations[0] is the input. */
  activations: number[][];
  /** contributions[l-1][i][j] = weights[l-1][i][j] * activations[l-1][j]. */
  contributions: number[][][];
}

export function softmax(z: number[]): number[] {
  const max = Math.max(...z);
  const e = z.map((v) => Math.exp(v - max));
  const sum = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / sum);
}

export function forward(weights: number[][][], biases: number[][], x: number[], activation: ActivationName): ForwardResult {
  const act = ACTIVATIONS[activation];
  let a = x.slice();
  const preActivations: number[][] = [];
  const activations: number[][] = [a];
  const contributions: number[][][] = [];

  weights.forEach((w, idx) => {
    const isOutput = idx === weights.length - 1;
    const c = w.map((row) => row.map((wij, j) => wij * a[j]));
    const z = c.map((row, i) => row.reduce((sum, v) => sum + v, 0) + biases[idx][i]);
    a = isOutput ? softmax(z) : z.map(act);
    contributions.push(c);
    preActivations.push(z);
    activations.push(a);
  });

  return { preActivations, activations, contributions };
}
