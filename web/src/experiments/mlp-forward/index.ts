import { defineExperiment, type ControlSpec } from "../../app/types";
import { clamp, fmt, prefersReducedMotion } from "../../lib/format";
import { loadMlpData, type MlpCtx } from "./data";
import { PARAM_SLIDER, REVEAL_SPEED } from "./encoding";
import { explain } from "./explain";
import { ACTIVATION_LABELS, forward, type ActivationName, type ForwardResult } from "./model";
import { stateFromPreset, type EdgeEncoding, type MlpState } from "./state";
import { matricesView } from "./views/matrices";
import { networkView } from "./views/network";

const presetOf = (state: MlpState, ctx: MlpCtx) => ctx.presets.find((p) => p.id === state.presetId)!;

function controls(state: MlpState, _result: ForwardResult, ctx: MlpCtx): ControlSpec<MlpState>[] {
  const p = presetOf(state, ctx);
  const L = p.layer_sizes.length - 1;
  const reduced = prefersReducedMotion();
  const specs: ControlSpec<MlpState>[] = [
    {
      kind: "select",
      id: "preset",
      group: "Network",
      label: "Weights",
      options: ctx.presets.map((x) => ({ value: x.id, label: x.title })),
      value: state.presetId,
      set: (s, v) => stateFromPreset(ctx.presets.find((x) => x.id === v)!, s.edgeEncoding),
    },
    {
      kind: "select",
      id: "activation",
      group: "Network",
      label: "Hidden activation function",
      options: (Object.keys(ACTIVATION_LABELS) as ActivationName[]).map((k) => ({ value: k, label: ACTIVATION_LABELS[k] })),
      value: state.activation,
      set: (s, v) => ({ ...s, activation: v as ActivationName }),
    },
    ...p.input_labels.map((label, i): ControlSpec<MlpState> => ({
      kind: "slider",
      id: `input-${i}`,
      group: "Input",
      label: `Input ${label}`,
      min: p.input_range[0],
      max: p.input_range[1],
      step: 0.01,
      value: state.input[i],
      set: (s, v) => ({ ...s, input: s.input.map((x, k) => (k === i ? v : x)) }),
    })),
    {
      kind: "slider",
      id: "stage",
      group: "Propagation",
      label: "Signal has reached layer",
      min: 0,
      max: L,
      step: 0.01,
      value: state.stage,
      format: (v) => `${fmt(v, 1)} of ${L}`,
      set: (s, v) => ({ ...s, stage: v, playing: false }),
    },
    {
      kind: "button",
      id: "play",
      group: "Propagation",
      // With reduced motion, step one layer per click instead of animating.
      label: reduced ? "Next layer" : state.playing ? "Playing\u2026" : "Replay propagation",
      onClick: (s) => {
        if (reduced) return { ...s, stage: s.stage >= L ? 0 : Math.floor(s.stage) + 1, playing: false };
        return { ...s, stage: 0, playing: true };
      },
    },
    {
      kind: "button",
      id: "reset",
      group: "Propagation",
      label: "Reset all",
      onClick: (s) => stateFromPreset(p, s.edgeEncoding),
    },
    {
      kind: "select",
      id: "edge-encoding",
      group: "Display",
      label: "Line thickness shows",
      options: [
        { value: "signal", label: "Signal (weight \u00D7 activation)" },
        { value: "weight", label: "Weight only" },
      ],
      value: state.edgeEncoding,
      set: (s, v) => ({ ...s, edgeEncoding: v as EdgeEncoding }),
    },
  ];

  const sel = state.selection;
  if (sel?.kind === "edge") {
    specs.push({
      kind: "slider",
      id: "sel-weight",
      group: "Edit selected connection",
      label: "Weight",
      ...PARAM_SLIDER,
      value: state.weights[sel.layer - 1][sel.to][sel.from],
      set: (s, v) => ({ ...s, weights: s.weights.map((m, l) => (l === sel.layer - 1 ? m.map((row, i) => (i === sel.to ? row.map((w, j) => (j === sel.from ? v : w)) : row)) : m)) }),
    });
  } else if (sel?.kind === "neuron" && sel.layer > 0) {
    specs.push({
      kind: "slider",
      id: "sel-bias",
      group: "Edit selected neuron",
      label: "Bias",
      ...PARAM_SLIDER,
      value: state.biases[sel.layer - 1][sel.index],
      set: (s, v) => ({ ...s, biases: s.biases.map((b, l) => (l === sel.layer - 1 ? b.map((x, i) => (i === sel.index ? v : x)) : b)) }),
    });
  }
  return specs;
}

export const mlpForward = defineExperiment<MlpState, ForwardResult, MlpCtx>({
  meta: {
    id: "mlp-forward",
    title: "Forward pass of a neural network",
    topic: "Artificial neural networks",
    summary: "Watch numbers flow through a small multilayer network. Change an input or a weight and see exactly what changes.",
    provenance: [
      { aspect: "Network and weights", kind: "schematic", note: "Hand-designed or random, not learned from data (training comes later)." },
      { aspect: "Activations", kind: "computed", note: "Calculated exactly from the weights and inputs shown; checked against a Python reference." },
    ],
  },
  load: loadMlpData,
  init: (ctx) => stateFromPreset(ctx.presets[0]),
  compute: (state) => forward(state.weights, state.biases, state.input, state.activation),
  controls,
  explain,
  views: [networkView, matricesView],
  animation: {
    active: (s) => s.playing,
    step(s, dtMs) {
      const L = s.weights.length;
      const stage = clamp(s.stage + (dtMs / 1000) * REVEAL_SPEED, 0, L);
      return { ...s, stage, playing: stage < L };
    },
  },
});
