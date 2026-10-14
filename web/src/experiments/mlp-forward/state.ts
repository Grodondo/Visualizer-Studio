import type { ActivationName } from "./model";
import type { Preset } from "./data";

export type Selection =
  | { kind: "neuron"; layer: number; index: number }
  // weights[layer-1][to][from]: the connection into neuron `to` of `layer`.
  | { kind: "edge"; layer: number; to: number; from: number };

export type EdgeEncoding = "signal" | "weight";

export interface MlpState {
  presetId: string;
  activation: ActivationName;
  weights: number[][][];
  biases: number[][];
  input: number[];
  /** How far the signal has travelled: 0 = only the input, L = all layers shown. */
  stage: number;
  playing: boolean;
  edgeEncoding: EdgeEncoding;
  selection: Selection | null;
}

export function stateFromPreset(p: Preset, edgeEncoding: EdgeEncoding = "signal"): MlpState {
  return {
    presetId: p.id,
    activation: p.activation,
    weights: structuredClone(p.weights),
    biases: structuredClone(p.biases),
    input: p.default_input.slice(),
    stage: p.layer_sizes.length - 1, // start fully revealed so the idea is visible without interaction
    playing: false,
    edgeEncoding,
    selection: null,
  };
}

export function sameSelection(a: Selection | null, b: Selection | null): boolean {
  if (!a || !b) return a === b;
  if (a.kind === "neuron" && b.kind === "neuron") return a.layer === b.layer && a.index === b.index;
  if (a.kind === "edge" && b.kind === "edge") return a.layer === b.layer && a.to === b.to && a.from === b.from;
  return false;
}

/** Click-to-select, click-again-to-clear; shared by both views. */
export function toggleSelection(s: MlpState, sel: Selection): MlpState {
  return { ...s, selection: sameSelection(s.selection, sel) ? null : sel };
}
