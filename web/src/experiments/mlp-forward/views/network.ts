import { h, s } from "../../../lib/dom";
import { fmt } from "../../../lib/format";
import type { View, ViewDef, ViewHost } from "../../../app/types";
import type { MlpCtx } from "../data";
import type { ForwardResult } from "../model";
import { SIGNAL_MAX, WEIGHT_MAX, magnitude, reveal, signColor } from "../encoding";
import { sameSelection, toggleSelection, type MlpState, type Selection } from "../state";

const NARROW_PX = 520;
const MIN_WIDTH_PX = 300;

interface Layout {
  width: number;
  height: number;
  radius: number;
  /** Narrow screens put every label under its node; wide screens put input/output names beside the columns. */
  narrow: boolean;
  x(layer: number): number;
  y(layer: number, index: number): number;
}

function computeLayout(sizes: number[], width: number): Layout {
  const narrow = width < NARROW_PX;
  const radius = narrow ? 17 : 26;
  const padLeft = narrow ? 30 : 70;
  const padRight = narrow ? 36 : 112;
  const top = 34;
  const rowH = radius * 2 + 22; // room for a role label under each node
  const maxN = Math.max(...sizes);
  const height = top + maxN * rowH + 8;
  const span = width - padLeft - padRight;
  const lastLayer = sizes.length - 1;
  return {
    width,
    height,
    radius,
    narrow,
    x: (l) => padLeft + (span * l) / lastLayer,
    y: (l, i) => top + (height - top - 8 - sizes[l] * rowH) / 2 + rowH * i + radius + 2,
  };
}

interface NodeRefs {
  g: SVGGElement;
  circle: SVGCircleElement;
  value: SVGTextElement;
  title: SVGTitleElement;
}
interface EdgeRefs {
  g: SVGGElement;
  halo: SVGLineElement;
  line: SVGLineElement;
  title: SVGTitleElement;
  layer: number;
  to: number;
  from: number;
}

function createNetworkView(container: HTMLElement, host: ViewHost<MlpState>, ctx: MlpCtx): View<MlpState, ForwardResult> {
  const root = h("div", { class: "view view--network" });
  const svgHost = h("div", { class: "net-host" });
  root.append(svgHost, encodingKey());
  container.append(root);

  let last: { state: MlpState; result: ForwardResult } | null = null;
  let built: { key: string; nodes: NodeRefs[][]; edges: EdgeRefs[] } | null = null;
  let width = 0;

  const preset = (state: MlpState) => ctx.presets.find((p) => p.id === state.presetId)!;

  function build(state: MlpState, w: number): void {
    const p = preset(state);
    const sizes = p.layer_sizes;
    const L = sizes.length - 1;
    const lay = computeLayout(sizes, w);
    const svg = s("svg", { class: `net${lay.narrow ? " net--narrow" : ""}`, viewBox: `0 0 ${lay.width} ${lay.height}`, width: lay.width, height: lay.height, role: "group", "aria-label": "Neural network diagram. Click a neuron or connection to inspect it." });
    svg.addEventListener("click", (e) => {
      if (e.target === svg) host.update((st) => ({ ...st, selection: null }));
    });

    // Layer names label the columns directly (no separate legend needed).
    sizes.forEach((_, l) => {
      const name = l === 0 ? "Input" : l === L ? "Output" : `Hidden ${l}`;
      svg.append(s("text", { class: "layer-name", x: lay.x(l), y: 16, "text-anchor": "middle" }, name));
    });

    const edgeLayer = s("g");
    const nodeLayer = s("g");
    svg.append(edgeLayer, nodeLayer);

    const edges: EdgeRefs[] = [];
    for (let l = 1; l <= L; l++) {
      for (let to = 0; to < sizes[l]; to++) {
        for (let from = 0; from < sizes[l - 1]; from++) {
          const coords = { x1: lay.x(l - 1), y1: lay.y(l - 1, from), x2: lay.x(l), y2: lay.y(l, to), pathLength: 1 };
          const sel: Selection = { kind: "edge", layer: l, to, from };
          const hit = s("line", { ...coords, class: "edge-hit" });
          const halo = s("line", { ...coords, class: "edge-halo" });
          const line = s("line", { ...coords, class: "edge-line" });
          const title = s("title");
          const g = s("g", { class: "edge", onclick: () => host.update((st) => toggleSelection(st, sel)) }, title, hit, halo, line);
          edgeLayer.append(g);
          edges.push({ g, halo, line, title, layer: l, to, from });
        }
      }
    }

    const nodes: NodeRefs[][] = sizes.map((n, l) => {
      const labels = p.neuron_labels[l] ?? (l === 0 ? p.input_labels : null);
      return Array.from({ length: n }, (_, i) => {
        const sel: Selection = { kind: "neuron", layer: l, index: i };
        const cx = lay.x(l);
        const cy = lay.y(l, i);
        const circle = s("circle", { class: "node-circle", cx, cy, r: lay.radius });
        const value = s("text", { class: "node-value", x: cx, y: cy, "text-anchor": "middle", "dominant-baseline": "central" });
        const title = s("title");
        const activate = () => host.update((st) => toggleSelection(st, sel));
        const g = s(
          "g",
          {
            class: "node",
            tabindex: 0,
            role: "button",
            "aria-label": `${labels?.[i] ?? `neuron ${i + 1}`}, layer ${l}`,
            onclick: activate,
            onkeydown: (e) => {
              const key = (e as KeyboardEvent).key;
              if (key === "Enter" || key === " ") {
                e.preventDefault();
                activate();
              }
            },
          },
          title,
          circle,
          value,
        );
        if (labels?.[i]) {
          // Input and output names sit beside the column; hidden-unit roles sit under the node.
          const side = lay.narrow ? null : l === 0 ? "start" : l === L ? "end" : null;
          const text = side
            ? s("text", { class: "node-label", x: l === 0 ? cx - lay.radius - 6 : cx + lay.radius + 6, y: cy, "text-anchor": l === 0 ? "end" : "start", "dominant-baseline": "central" }, labels[i])
            : s("text", { class: "node-label node-label--under", x: cx, y: cy + lay.radius + 12, "text-anchor": "middle" }, labels[i]);
          nodeLayer.append(text);
        }
        nodeLayer.append(g);
        return { g, circle, value, title };
      });
    });

    svgHost.replaceChildren(svg);
    built = { key: `${state.presetId}|${w}`, nodes, edges };
  }

  function paint(state: MlpState, result: ForwardResult): void {
    const p = preset(state);
    const L = p.layer_sizes.length - 1;
    const b = built!;

    for (const e of b.edges) {
      const w = state.weights[e.layer - 1][e.to][e.from];
      const c = result.contributions[e.layer - 1][e.to][e.from];
      const showSignal = state.edgeEncoding === "signal";
      const v = showSignal ? c : w;
      const m = magnitude(v, showSignal ? SIGNAL_MAX : WEIGHT_MAX);
      const r = reveal(state.stage, e.layer);
      const opacity = v === 0 ? 0.04 : showSignal ? 0.08 + 0.92 * m : 0.2 + 0.8 * m;
      e.line.style.stroke = signColor(v);
      e.line.style.strokeWidth = String(0.6 + 3.4 * m);
      e.line.style.strokeOpacity = String(opacity * r);
      // pathLength=1 lets the dash length equal the fraction of the line drawn so far.
      e.line.style.strokeDasharray = r < 1 ? `${r} 1` : "none";
      const selected = state.selection?.kind === "edge" && sameSelection(state.selection, { kind: "edge", layer: e.layer, to: e.to, from: e.from });
      e.halo.style.display = selected ? "block" : "none";
      const fromName = p.neuron_labels[e.layer - 1]?.[e.from] ?? `neuron ${e.from + 1}`;
      e.title.textContent = `weight ${fmt(w, 2)} \u00D7 ${fromName} ${fmt(result.activations[e.layer - 1][e.from], 2)} = ${fmt(c, 2)}`;
    }

    b.nodes.forEach((layerNodes, l) => {
      const r = reveal(state.stage, l);
      layerNodes.forEach((n, i) => {
        const a = result.activations[l][i];
        n.circle.style.fill = signColor(a);
        n.circle.style.fillOpacity = String(0.1 + 0.85 * Math.min(1, Math.abs(a)));
        n.g.style.opacity = String(0.25 + 0.75 * r);
        n.value.textContent = fmt(a, 2);
        const selected = sameSelection(state.selection, { kind: "neuron", layer: l, index: i });
        n.g.classList.toggle("is-selected", selected);
        const label = (l === 0 ? p.input_labels[i] : p.neuron_labels[l]?.[i]) ?? `neuron ${i + 1}`;
        n.title.textContent = l === 0 ? `${label} = ${fmt(a, 2)}` : `${label}: z = ${fmt(result.preActivations[l - 1][i], 2)}, ${l === L ? "softmax" : state.activation}(z) = ${fmt(a, 2)}`;
      });
    });
  }

  function render(): void {
    if (!last) return;
    const w = Math.max(MIN_WIDTH_PX, Math.floor(svgHost.clientWidth || container.clientWidth));
    if (!built || built.key !== `${last.state.presetId}|${w}`) {
      width = w;
      build(last.state, w);
    }
    paint(last.state, last.result);
  }

  // Rebuild the SVG when the available width changes so text stays at a readable pixel size.
  // Deferred to the next frame: rebuilding inside the callback resizes the observed box and triggers a loop warning.
  let frameId = 0;
  const observer = new ResizeObserver(() => {
    if (frameId) return;
    frameId = requestAnimationFrame(() => {
      frameId = 0;
      const w = Math.max(MIN_WIDTH_PX, Math.floor(svgHost.clientWidth));
      if (w !== width) render();
    });
  });
  observer.observe(svgHost);

  return {
    update(state, result) {
      last = { state, result };
      render();
    },
    destroy() {
      observer.disconnect();
      if (frameId) cancelAnimationFrame(frameId);
      root.remove();
    },
  };
}

/** The one place the encodings are explained; the lines themselves carry no extra legend. */
function encodingKey(): HTMLElement {
  const swatch = (cls: string) => h("span", { class: `swatch swatch--${cls}`, "aria-hidden": "true" });
  return h(
    "div",
    { class: "key", "aria-label": "How to read the diagram" },
    h("span", {}, swatch("pos"), " positive"),
    h("span", {}, swatch("neg"), " negative"),
    h("span", {}, "Circle: brighter = larger |activation|"),
    h("span", {}, "Line: thicker and more opaque = larger |weight| or |signal|"),
  );
}

export const networkView: ViewDef<MlpState, ForwardResult, MlpCtx> = {
  id: "network",
  label: "Network",
  tradeoff: "Explanatory: shows how the signal spreads layer by layer; exact numbers are secondary (hover or select to read them).",
  create: createNetworkView,
};
