import type { ActivationName } from "./model";

/** Shape of web/public/data/mlp.json, written by `uv run python -m compute.export`. */
export interface Preset {
  id: string;
  title: string;
  description: string;
  activation: ActivationName;
  layer_sizes: number[];
  weights: number[][][];
  biases: number[][];
  input_labels: string[];
  neuron_labels: (string[] | null)[];
  input_range: [number, number];
  default_input: number[];
}

export interface MlpCtx {
  presets: Preset[];
}

export async function loadMlpData(): Promise<MlpCtx> {
  const res = await fetch(`${import.meta.env.BASE_URL}data/mlp.json`);
  if (!res.ok) throw new Error(`Could not load mlp.json (${res.status}). Run: uv run python -m compute.export`);
  return (await res.json()) as MlpCtx;
}
