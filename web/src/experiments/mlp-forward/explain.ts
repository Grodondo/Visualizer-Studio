import { h } from "../../lib/dom";
import { fmt } from "../../lib/format";
import type { MlpCtx, Preset } from "./data";
import { ACTIVATION_LABELS, type ForwardResult } from "./model";
import type { MlpState } from "./state";

const nameOf = (p: Preset, l: number, i: number): string => (l === 0 ? p.input_labels[i] : p.neuron_labels[l]?.[i]) ?? `neuron ${i + 1}`;

function selectionDetail(state: MlpState, result: ForwardResult, p: Preset): HTMLElement {
  const sel = state.selection;
  const L = p.layer_sizes.length - 1;
  if (!sel) return h("p", { class: "hint" }, "Click a neuron or a connection (in either view) to see its arithmetic and edit its value.");

  if (sel.kind === "edge") {
    const w = state.weights[sel.layer - 1][sel.to][sel.from];
    const aFrom = result.activations[sel.layer - 1][sel.from];
    return h(
      "div",
      {},
      h("h3", {}, `Connection ${nameOf(p, sel.layer - 1, sel.from)} \u2192 ${nameOf(p, sel.layer, sel.to)}`),
      h("p", { class: "mono" }, `weight ${fmt(w)} \u00D7 activation ${fmt(aFrom)} = ${fmt(result.contributions[sel.layer - 1][sel.to][sel.from])}`),
      h("p", {}, "That product is what this connection adds to the receiving neuron\u2019s total z. Change the weight with the slider and watch everything downstream."),
    );
  }

  if (sel.layer === 0) {
    return h("div", {}, h("h3", {}, `Input ${nameOf(p, 0, sel.index)}`), h("p", {}, "Inputs are set with the sliders under Input; they are not computed."));
  }

  const l = sel.layer;
  const i = sel.index;
  const fn = l === L ? "softmax" : ACTIVATION_LABELS[state.activation];
  const rows = state.weights[l - 1][i].map((w, j) =>
    h("tr", {}, h("td", {}, nameOf(p, l - 1, j)), h("td", { class: "num" }, fmt(w)), h("td", { class: "num" }, fmt(result.activations[l - 1][j])), h("td", { class: "num" }, fmt(result.contributions[l - 1][i][j]))),
  );
  return h(
    "div",
    {},
    h("h3", {}, `Neuron ${nameOf(p, l, i)} (layer ${l})`),
    h("table", { class: "mx mx--small" }, h("thead", {}, h("tr", {}, h("th", {}, "from"), h("th", {}, "w"), h("th", {}, "a"), h("th", {}, "w \u00D7 a"))), h("tbody", {}, ...rows)),
    h("p", { class: "mono" }, `z = sum of the products + bias ${fmt(state.biases[l - 1][i])} = ${fmt(result.preActivations[l - 1][i])}`),
    h("p", { class: "mono" }, `a = ${fn}(z${l === L ? ", all outputs" : ""}) = ${fmt(result.activations[l][i])}`),
  );
}

export function explain(state: MlpState, result: ForwardResult, ctx: MlpCtx): HTMLElement {
  const p = ctx.presets.find((x) => x.id === state.presetId)!;
  const L = p.layer_sizes.length - 1;
  const out = result.activations[L];
  const best = out.indexOf(Math.max(...out));
  return h(
    "div",
    { class: "explain" },
    h("p", {}, p.description),
    h("p", { class: "readout" }, `Output: ${nameOf(p, L, best)}, ${fmt(out[best] * 100, 1)}%`),
    h("p", { class: "hint" }, "Each neuron computes z = (weights \u00D7 incoming activations) + bias, then applies its activation function. The output layer uses softmax so its values add up to 1."),
    selectionDetail(state, result, p),
  );
}
