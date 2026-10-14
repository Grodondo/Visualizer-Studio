import { h } from "../lib/dom";
import { fmt } from "../lib/format";
import type { ButtonSpec, ControlSpec, SelectSpec, SliderSpec, ViewHost } from "../app/types";

interface Entry<S> {
  spec: ControlSpec<S>;
  refresh(): void;
}

/**
 * Renders declarative control specs. Controls are matched by id and updated in place,
 * so a slider keeps focus and drag state while its value changes. The DOM is rebuilt
 * only when the set of controls changes (e.g. a weight slider appears on selection).
 */
export function createControlPanel<S>(host: ViewHost<S>): { el: HTMLElement; render(specs: ControlSpec<S>[]): void } {
  const el = h("div", { class: "controls" });
  let signature = "";
  const entries = new Map<string, Entry<S>>();

  function buildSlider(spec: SliderSpec<S>): { node: HTMLElement; entry: Entry<S> } {
    const input = h("input", { type: "range", id: `ctl-${spec.id}`, min: spec.min, max: spec.max, step: spec.step });
    const out = h("output", { for: `ctl-${spec.id}` });
    const entry: Entry<S> = {
      spec,
      refresh() {
        const sp = entry.spec as SliderSpec<S>;
        input.value = String(sp.value);
        out.textContent = (sp.format ?? ((v: number) => fmt(v)))(sp.value);
      },
    };
    input.addEventListener("input", () => {
      const sp = entry.spec as SliderSpec<S>;
      host.update((s) => sp.set(s, Number(input.value)));
    });
    const node = h("div", { class: "control" }, h("div", { class: "control-head" }, h("label", { for: `ctl-${spec.id}` }, spec.label), out), input);
    return { node, entry };
  }

  function buildSelect(spec: SelectSpec<S>): { node: HTMLElement; entry: Entry<S> } {
    const select = h("select", { id: `ctl-${spec.id}` }, ...spec.options.map((o) => h("option", { value: o.value }, o.label)));
    const entry: Entry<S> = {
      spec,
      refresh() {
        select.value = (entry.spec as SelectSpec<S>).value;
      },
    };
    select.addEventListener("change", () => {
      const sp = entry.spec as SelectSpec<S>;
      host.update((s) => sp.set(s, select.value));
    });
    const node = h("div", { class: "control" }, h("div", { class: "control-head" }, h("label", { for: `ctl-${spec.id}` }, spec.label)), select);
    return { node, entry };
  }

  function buildButton(spec: ButtonSpec<S>): { node: HTMLElement; entry: Entry<S> } {
    const button = h("button", { type: "button", class: "btn" });
    const entry: Entry<S> = {
      spec,
      refresh() {
        button.textContent = entry.spec.label;
      },
    };
    button.addEventListener("click", () => {
      const sp = entry.spec as ButtonSpec<S>;
      host.update((s) => sp.onClick(s));
    });
    return { node: h("div", { class: "control control--button" }, button), entry };
  }

  function rebuild(specs: ControlSpec<S>[]): void {
    entries.clear();
    el.replaceChildren();
    let group: HTMLElement | null = null;
    let lastGroup: string | undefined;
    for (const spec of specs) {
      if (!group || spec.group !== lastGroup) {
        lastGroup = spec.group;
        group = h("fieldset", { class: "control-group" }, spec.group ? h("legend", {}, spec.group) : null);
        el.append(group);
      }
      const built = spec.kind === "slider" ? buildSlider(spec) : spec.kind === "select" ? buildSelect(spec) : buildButton(spec);
      entries.set(spec.id, built.entry);
      group.append(built.node);
    }
  }

  return {
    el,
    render(specs) {
      const next = specs.map((s) => `${s.group ?? ""}|${s.kind}|${s.id}|${s.kind === "select" ? s.options.map((o) => o.value).join(",") : ""}|${s.kind === "slider" ? `${s.min},${s.max},${s.step}` : ""}`).join(";");
      if (next !== signature) {
        signature = next;
        rebuild(specs);
      }
      for (const spec of specs) {
        const entry = entries.get(spec.id)!;
        entry.spec = spec;
        entry.refresh();
      }
    },
  };
}
