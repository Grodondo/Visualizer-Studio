---
paths:
  - "scenes/**/*.py"
---

# Manim CE rules (0.22)

## If you see ManimGL code, translate it, never paste it

| ManimGL | Manim CE |
|---|---|
| `from manimlib import *` | `from manim import *` |
| `ShowCreation` | `Create` |
| `Tex` (math mode) | `MathTex` |
| `TexText` | `Tex` |
| `self.frame` | `self.camera.frame`, in a `MovingCameraScene` |
| `axes.get_graph(f)` | `axes.plot(f)` |
| `CONFIG = {...}` | constructor keyword arguments |
| `InteractiveScene`, `self.embed()` | no direct equivalent |

## Equations: choose one system per formula, never mix syntax

- **LaTeX** `MathTex(r"\sum_{k=0}^n k = \frac{n(n+1)}{2}")`. Always raw strings. Split into parts to
  animate pieces: `MathTex("a^2", "+", "b^2", "=", "c^2")`, then `TransformMatchingTex`.
  Default choice when LaTeX is installed (longest track record in docs and examples).
- **Typst** `MathTypst("sum_(k=0)^n k = (n(n+1)) / 2")`. Group parts with `{{ a^2 + b^2 : lhs }} = {{ c^2 }}`
  and fetch them with `select()` (read its docstring). Needs `manim[typst]`. Use when LaTeX is missing.
- **Typst text is black by default.** On Manim's black background it is invisible: always pass `color=`.
- Plain labels: `Text` (Pango). Prefer it for non-math words.
- `Code` (changed in 0.21): colors come from the Pygments `formatter_style`; `paragraph_config={"color": ...}` no longer works.
- `MathTex` compiles LaTeX per unique string. Do not create hundreds in a loop.

## Layout

- Default frame: 8 units high, about 14.22 wide. Keep content within |x| <= 6.5 and |y| <= 3.5.
- Position relatively (`next_to`, `arrange`, `to_edge`, `align_to`). Use absolute coordinates only for plotted data.
- Wrap related objects in `VGroup` before moving or fading them.

## Pacing and style (3Blue1Brown-like)

- One idea per scene. Reveal progressively: `Write`, `Create`, `FadeIn`, `LaggedStart`.
- `self.wait(1)` after every beat; longer after text or equations. Never stack unreadable text.
- Define semantic colors at the top (`COLOR_WEIGHT = BLUE`) and keep them identical in the entire video.
  Do not rely on red vs green alone (color-blind viewers).
- Use motion to show relationships: `ReplacementTransform`, `TransformFromCopy`, `TransformMatchingTex`.
  Do not just fade objects in and out.
- Continuous change: `ValueTracker` plus `always_redraw` or updaters. Keep updaters cheap.

## Neural-network diagrams (build natively; no ManimML)

- A `layer(n)` helper returns a `VGroup` of `Circle`s (`arrange(DOWN)`); layers go `arrange(RIGHT)`.
- Never draw every edge of a large layer (784 -> 16 is 12,544 lines). Show a subset plus a vertical
  ellipsis `MathTex(r"\vdots")`, as 3b1b does. Rule of thumb: under about 500 individual edges per scene.
- Encode values visually: activation -> fill opacity; weight sign -> color, magnitude -> stroke opacity or width.
- Forward pass: `LaggedStart` of `ShowPassingFlash` on edges, or `Indicate` on neurons.
- "Real" behavior: compute with numpy/PyTorch first, save arrays under `assets/`, load them in the scene.
  Never train or compute inside `construct()`.

## Graphs

- Compute with networkx; animate with `Graph` / `DiGraph` (`Graph.from_networkx`). Fix a seed in
  `layout_config` so layouts are reproducible (verify the argument in the docstring).
- Algorithm animations: precompute the list of states, then one `self.play` per step. Color = state
  (unvisited / frontier / done), with a small legend.

## 3D

- `ThreeDScene` + `set_camera_orientation(phi=..., theta=...)`; spin with `begin_ambient_camera_rotation`.
  Keep text readable with `add_fixed_in_frame_mobjects`.
- Cairo 3D is slow and has depth-sorting artifacts. Use coarse surfaces for drafts (`Surface(resolution=(24, 24))`).
- Heavy meshes (brains) -> render in PyVista and import frames, or make it interactive in Three.js.

## Debugging in VS Code

Render from Python so the debugger works, and construct the scene inside the `with` block
(0.22 captures output settings at construction):

```python
with tempconfig({"quality": "low_quality", "preview": True}):
    MyScene().render()
```
