type AttrValue = string | number | boolean | undefined | null | ((e: Event) => void);
export type Child = Node | string | number | null | undefined | false;

function apply(el: Element, attrs: Record<string, AttrValue> | undefined, children: Child[]): void {
  for (const [key, value] of Object.entries(attrs ?? {})) {
    if (value === undefined || value === null || value === false) continue;
    if (typeof value === "function") el.addEventListener(key.slice(2), value); // onclick -> "click"
    else if (value === true) el.setAttribute(key, "");
    else el.setAttribute(key, String(value));
  }
  for (const child of children) {
    if (child === null || child === undefined || child === false) continue;
    el.append(child instanceof Node ? child : String(child));
  }
}

/** Create an HTML element: h("button", { class: "x", onclick: fn }, "label"). */
export function h<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  attrs?: Record<string, AttrValue>,
  ...children: Child[]
): HTMLElementTagNameMap[K] {
  const el = document.createElement(tag);
  apply(el, attrs, children);
  return el;
}

/** Create an SVG element (needs the SVG namespace, so it cannot share h()). */
export function s<K extends keyof SVGElementTagNameMap>(
  tag: K,
  attrs?: Record<string, AttrValue>,
  ...children: Child[]
): SVGElementTagNameMap[K] {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  apply(el, attrs, children);
  return el;
}
