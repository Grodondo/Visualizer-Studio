---
paths:
  - "web/**"
---

# Interactive web visualizations

## Stack

- **Vite + TypeScript** (`npm create vite@latest web -- --template vanilla-ts`). Stay framework-free
  unless the page needs real UI state; ask before adding React or Vue.
- **2D data-driven**: D3 (scales, axes, joins) on SVG. Switch to Canvas when you draw more than
  roughly 5,000 elements (rule of thumb).
- **3D**: Three.js. **Quick charts**: Observable Plot or Plotly.
- Pin exact versions in `package.json`. Check the current version with `npm view <pkg> version`
  before adding a dependency; do not rely on memory.

## Three.js pitfalls

- Releases ship about monthly and can break things. Pin the version and read the release notes before upgrading.
- Two entry points exist: `three` (WebGLRenderer) and `three/webgpu` (WebGPURenderer, TSL shaders).
  They are **separate builds with different class identities**. Never mix classes between them
  (for example lights created from `three` silently do nothing under the WebGPU renderer).
- Default to `WebGLRenderer` (widest support). Use `three/webgpu` only for compute shaders or TSL,
  and check availability before creating the renderer.
- Use `renderer.setAnimationLoop(...)`. Cap pixel ratio: `renderer.setPixelRatio(Math.min(devicePixelRatio, 2))`.
  Handle window resize. Call `.dispose()` on geometries, materials, and textures you remove.
- Camera controls come from `three/addons/controls/OrbitControls.js`.

## Interaction design

- One state object is the single source of truth. Controls update the state; one `render()` redraws.
- The initial state must already show the key idea without any interaction. Controls come second.
- Label objects directly instead of using a legend. Put units on sliders.
- Respect `prefers-reduced-motion`, keep controls keyboard-accessible, and check color contrast.
- Neural networks in the browser: run the forward pass in plain TypeScript on small arrays, with
  weights exported from Python as JSON (`assets/` -> `web/public/data/`). Do not add TensorFlow.js
  or ONNX unless the model truly needs it.
- Keep data files small (roughly under 5 MB). Downsample on the Python side, not in the browser.
- Brain meshes: export from PyVista to glTF/GLB and load them with Three.js `GLTFLoader`.

## Verify before reporting done

- `npm run build` passes with zero TypeScript errors.
- Run `npm run dev`. If a headless browser is available, take a screenshot and look at it.
- Check a narrow phone width (360 px) and a desktop width.
