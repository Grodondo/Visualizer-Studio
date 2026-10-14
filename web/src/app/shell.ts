import { h } from "../lib/dom";
import { Store } from "../lib/store";
import { PROVENANCE, provenanceBadge, provenanceList } from "../ui/badge";
import { createControlPanel } from "../ui/controls";
import type { AnyExperiment, ProvenanceKind, View } from "./types";

const COMPARE = "__compare__";

function galleryPage(experiments: AnyExperiment[]): HTMLElement {
  const byTopic = new Map<string, AnyExperiment[]>();
  for (const e of experiments) byTopic.set(e.meta.topic, [...(byTopic.get(e.meta.topic) ?? []), e]);

  return h(
    "div",
    { class: "page" },
    h("h1", {}, "Visualizer Studio"),
    h("p", { class: "lead" }, "A playground for seeing how neural networks, brains and other complex systems work. Pick an experiment, change something, and watch the result."),
    h(
      "section",
      { class: "prov-key", "aria-label": "What the labels mean" },
      h("h2", {}, "Every experiment is labeled"),
      h("ul", { class: "prov-list" }, ...(Object.keys(PROVENANCE) as ProvenanceKind[]).map((k) => h("li", {}, provenanceBadge(k), ` ${PROVENANCE[k].meaning}`))),
    ),
    ...[...byTopic].map(([topic, list]) =>
      h(
        "section",
        { class: "topic" },
        h("h2", {}, topic),
        h(
          "div",
          { class: "cards" },
          ...list.map((e) =>
            h(
              "a",
              { class: "card", href: `#/e/${e.meta.id}` },
              h("h3", {}, e.meta.title),
              h("p", {}, e.meta.summary),
              h("div", { class: "badges" }, ...[...new Set(e.meta.provenance.map((n) => n.kind))].map(provenanceBadge)),
            ),
          ),
        ),
      ),
    ),
  );
}

/** Mounts one experiment page. Returns a disposer that stops animation and removes listeners. */
function experimentPage(main: HTMLElement, exp: AnyExperiment): () => void {
  let disposed = false;
  let cleanup: (() => void) | null = null;
  main.replaceChildren(h("div", { class: "page" }, h("p", { class: "hint" }, "Loading\u2026")));

  exp.load().then(
    (ctx) => {
      if (!disposed) cleanup = mountLoaded(main, exp, ctx);
    },
    (err: unknown) => {
      if (!disposed) main.replaceChildren(h("div", { class: "page" }, h("p", { class: "error" }, `Could not load "${exp.meta.title}": ${err instanceof Error ? err.message : String(err)}`)));
    },
  );
  return () => {
    disposed = true;
    cleanup?.();
  };
}

// AnyExperiment erases the state type, so state and result are `any` inside the shell.
function mountLoaded(main: HTMLElement, exp: AnyExperiment, ctx: unknown): () => void {
  const store = new Store(exp.init(ctx));
  const host = { getState: () => store.get(), update: store.update.bind(store) };

  const tabs = h("div", { class: "tabs", role: "group", "aria-label": "Representation" });
  const tradeoff = h("p", { class: "hint tradeoff" });
  const viewArea = h("div", { class: "view-area" });
  const panel = createControlPanel(host);
  const explainEl = h("div", { class: "panel explain-panel" });

  main.replaceChildren(
    h(
      "div",
      { class: "page page--experiment" },
      h("a", { class: "back", href: "#/" }, "\u2190 All experiments"),
      h("h1", {}, exp.meta.title),
      h("p", { class: "lead" }, exp.meta.summary),
      provenanceList(exp.meta.provenance),
      h(
        "div",
        { class: "layout" },
        h("section", { class: "stage", "aria-label": "Visualization" }, tabs, tradeoff, viewArea),
        h("div", { class: "panel controls-panel" }, panel.el),
        explainEl,
      ),
    ),
  );

  let mode: string = exp.views[0].id;
  let mounted: View<any, any>[] = [];

  function mountViews(): void {
    mounted.forEach((v) => v.destroy());
    viewArea.replaceChildren();
    const chosen = mode === COMPARE ? exp.views : exp.views.filter((v) => v.id === mode);
    viewArea.className = `view-area view-area--${chosen.length > 1 ? "compare" : "single"}`;
    mounted = chosen.map((def) => {
      const slot = h("div", { class: "view-slot" }, chosen.length > 1 ? h("h2", { class: "slot-title" }, def.label) : null);
      viewArea.append(slot);
      return def.create(slot, host, ctx);
    });
    tradeoff.textContent = mode === COMPARE ? "Both views read the same numbers from the same computation: only the representation differs." : exp.views.find((v) => v.id === mode)!.tradeoff;
    const choices: [string, string][] = [...exp.views.map((v): [string, string] => [v.id, v.label]), [COMPARE, "Compare"]];
    tabs.replaceChildren(
      ...choices.map(([id, label]) =>
        h("button", { type: "button", class: "tab", "aria-pressed": String(id === mode), onclick: () => { mode = id; mountViews(); render(); } }, label),
      ),
    );
  }

  function render(): void {
    const state = store.get();
    const result = exp.compute(state, ctx); // computed once; every mounted view reads this same result
    mounted.forEach((v) => v.update(state, result));
    panel.render(exp.controls(state, result, ctx));
    explainEl.replaceChildren(exp.explain(state, result, ctx));
  }

  // Playback: advance state on animation frames only while the experiment says it is playing.
  let raf = 0;
  let lastT = 0;
  function frame(t: number): void {
    raf = 0;
    const dt = lastT ? Math.min(64, t - lastT) : 16; // clamp so a background tab does not jump the animation
    lastT = t;
    store.update((s) => exp.animation!.step(s, dt));
  }
  const unsubscribe = store.subscribe((state) => {
    render();
    if (exp.animation?.active(state)) {
      if (!raf) raf = requestAnimationFrame(frame);
    } else {
      lastT = 0;
    }
  });

  mountViews();
  render();

  return () => {
    unsubscribe();
    if (raf) cancelAnimationFrame(raf);
    mounted.forEach((v) => v.destroy());
  };
}

export function startApp(root: HTMLElement, experiments: AnyExperiment[]): void {
  const main = h("main", { id: "main" });
  root.replaceChildren(
    h("header", { class: "app-header" }, h("a", { class: "brand", href: "#/" }, "Visualizer Studio"), h("span", { class: "tagline" }, "neural networks \u00B7 brains \u00B7 complex systems")),
    main,
  );

  let dispose: (() => void) | null = null;
  const route = (): void => {
    dispose?.();
    dispose = null;
    const id = location.hash.match(/^#\/e\/([\w-]+)/)?.[1];
    const exp = experiments.find((e) => e.meta.id === id);
    if (exp) dispose = experimentPage(main, exp);
    else main.replaceChildren(galleryPage(experiments));
    window.scrollTo(0, 0);
  };
  window.addEventListener("hashchange", route);
  route();
}
