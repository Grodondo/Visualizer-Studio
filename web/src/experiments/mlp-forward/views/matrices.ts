import { h } from "../../../lib/dom";
import { fmt } from "../../../lib/format";
import type { View, ViewDef, ViewHost } from "../../../app/types";
import type { MlpCtx, Preset } from "../data";
import { ACTIVATION_LABELS, type ForwardResult } from "../model";
import { WEIGHT_MAX, magnitude, reveal, signColor } from "../encoding";
import { sameSelection, toggleSelection, type MlpState } from "../state";

/** Cell tint: sign picks the hue, magnitude picks how strong. */
function tint(v: number, max: number): string {
  return `color-mix(in srgb, ${signColor(v)} ${Math.round(70 * magnitude(v, max))}%, transparent)`;
}

interface LayerRefs {
  section: HTMLElement;
  title: HTMLElement;
  rows: HTMLTableRowElement[];
  weightCells: HTMLButtonElement[][];
  prevValues: HTMLElement[];
  bias: HTMLElement[];
  z: HTMLElement[];
  a: HTMLElement[];
}

function createMatricesView(container: HTMLElement, host: ViewHost<MlpState>, ctx: MlpCtx): View<MlpState, ForwardResult> {
  const root = h("div", { class: "view view--matrices" });
  container.append(root);
  let key = "";
  let inputCells: HTMLElement[] = [];
  let layers: LayerRefs[] = [];

  const preset = (state: MlpState): Preset => ctx.presets.find((p) => p.id === state.presetId)!;
  const nameOf = (p: Preset, l: number, i: number) => (l === 0 ? p.input_labels[i] : p.neuron_labels[l]?.[i]) ?? `#${i + 1}`;

  function build(p: Preset, activation: string): void {
    const sizes = p.layer_sizes;
    const L = sizes.length - 1;
    root.replaceChildren();

    inputCells = Array.from({ length: sizes[0] }, () => h("td", { class: "num" }));
    const inputHead = h("tr", {}, h("th", {}), ...inputCells.map((_, i) => h("th", { class: "col-name" }, nameOf(p, 0, i))));
    root.append(
      h(
        "section",
        { class: "mx-layer" },
        h("h4", {}, "Layer 0 \u00B7 Input"),
        h("div", { class: "mx-scroll" }, h("table", { class: "mx" }, h("thead", {}, inputHead), h("tbody", {}, h("tr", {}, h("th", {}, "a\u2070"), ...inputCells)))),
      ),
    );

    layers = [];
    for (let l = 1; l <= L; l++) {
      const isOut = l === L;
      const fn = isOut ? "softmax" : ACTIVATION_LABELS[activation as keyof typeof ACTIVATION_LABELS];
      const title = h("h4", {}, `Layer ${l} \u00B7 ${isOut ? "Output" : `Hidden ${l}`}`);
      const prevValues = Array.from({ length: sizes[l - 1] }, () => h("td", { class: "num num--prev" }));
      const rows: HTMLTableRowElement[] = [];
      const weightCells: HTMLButtonElement[][] = [];
      const bias: HTMLElement[] = [];
      const z: HTMLElement[] = [];
      const a: HTMLElement[] = [];

      for (let i = 0; i < sizes[l]; i++) {
        const cells: HTMLButtonElement[] = [];
        const tds: HTMLElement[] = [];
        for (let j = 0; j < sizes[l - 1]; j++) {
          const btn = h("button", { type: "button", class: "wcell", "aria-label": `weight from ${nameOf(p, l - 1, j)} to ${nameOf(p, l, i)}`, onclick: () => host.update((st) => toggleSelection(st, { kind: "edge", layer: l, to: i, from: j })) });
          cells.push(btn);
          tds.push(h("td", {}, btn));
        }
        const neuronBtn = h("button", { type: "button", class: "name-btn", onclick: () => host.update((st) => toggleSelection(st, { kind: "neuron", layer: l, index: i })) }, nameOf(p, l, i));
        const b = h("td", { class: "num" });
        const zc = h("td", { class: "num" });
        const ac = h("td", { class: "num num--out" });
        bias.push(b);
        z.push(zc);
        a.push(ac);
        weightCells.push(cells);
        rows.push(h("tr", {}, h("th", { scope: "row" }, neuronBtn), ...tds, b, zc, ac));
      }

      const head = h(
        "tr",
        {},
        h("th", {}, "neuron"),
        ...Array.from({ length: sizes[l - 1] }, (_, j) => h("th", { class: "col-name" }, nameOf(p, l - 1, j))),
        h("th", {}, "b"),
        h("th", {}, "z"),
        h("th", { class: "col-out" }, fn),
      );
      const prevRow = h("tr", { class: "mx-prev" }, h("th", {}, `a\u207D${l - 1}\u207E`), ...prevValues, h("td", { colspan: 3 }));
      const section = h(
        "section",
        { class: "mx-layer" },
        title,
        h("p", { class: "mx-eq" }, `z = W \u00B7 a + b,   a = ${fn}(z)`),
        h("div", { class: "mx-scroll" }, h("table", { class: "mx" }, h("thead", {}, head), h("tbody", {}, prevRow, ...rows))),
      );
      root.append(section);
      layers.push({ section, title, rows, weightCells, prevValues, bias, z, a });
    }
  }

  return {
    update(state, result) {
      const p = preset(state);
      const k = `${state.presetId}|${state.activation}`;
      if (k !== key) {
        key = k;
        build(p, state.activation);
      }
      result.activations[0].forEach((v, i) => {
        inputCells[i].textContent = fmt(v, 2);
        inputCells[i].style.background = tint(v, 1);
      });
      layers.forEach((ly, idx) => {
        const l = idx + 1;
        ly.section.style.opacity = String(0.35 + 0.65 * reveal(state.stage, l));
        ly.prevValues.forEach((cell, j) => {
          cell.textContent = fmt(result.activations[l - 1][j], 2);
        });
        ly.rows.forEach((row, i) => {
          row.classList.toggle("is-selected", sameSelection(state.selection, { kind: "neuron", layer: l, index: i }));
          ly.weightCells[i].forEach((btn, j) => {
            const w = state.weights[l - 1][i][j];
            btn.textContent = fmt(w, 2);
            btn.style.background = tint(w, WEIGHT_MAX);
            btn.classList.toggle("is-selected", sameSelection(state.selection, { kind: "edge", layer: l, to: i, from: j }));
          });
          ly.bias[i].textContent = fmt(state.biases[l - 1][i], 2);
          ly.z[i].textContent = fmt(result.preActivations[l - 1][i], 2);
          const act = result.activations[l][i];
          ly.a[i].textContent = fmt(act, 2);
          // Layered over an opaque background so the sticky column hides cells scrolling beneath it.
          const t = tint(act, 1);
          ly.a[i].style.backgroundImage = `linear-gradient(${t}, ${t})`;
        });
      });
    },
    destroy() {
      root.remove();
    },
  };
}

export const matricesView: ViewDef<MlpState, ForwardResult, MlpCtx> = {
  id: "matrices",
  label: "Matrices",
  tradeoff: "Technical: every number in the calculation, row by row; the signal's spatial flow is lost.",
  create: createMatricesView,
};
