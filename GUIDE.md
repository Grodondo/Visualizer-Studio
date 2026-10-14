# Animation Studio Guide

A beginner's guide to this repository: what each file does, how to change the neural-network video
safely, and where to go next. Everything here was run on Windows PowerShell with Manim Community 0.22.0;
commands are meant to be pasted at the repository root (`c:\dev\Visualizers\Test1`).

## 1. The big picture

```
compute/mlp.py + mlp_presets.py   the model: weights, biases, the forward-pass arithmetic (pure numbers)
            |  uv run python -m compute.export
            v
assets/mlp/forward_trace.json     every number of ONE forward pass (generated file)
            |  loaded by scenes/studio/trace.py
            v
scenes/studio/*.py                reusable drawing pieces: network, equations, plot, theme, beats
            |  used by
            v
scenes/mlp_forward.py             the video itself: what appears, when, how it moves
            |  uv run manim ...
            v
media/videos/.../MlpForwardPass.mp4   the rendered video (generated, never edited by hand)
```

The rule that keeps this safe: **numbers are computed in `compute/`, never inside a scene.**
A scene only *shows* numbers it reads from the trace, so a video can never disagree with the model.

## 2. Which file does what, and how freely you can edit it

| File | Role | Editing |
|---|---|---|
| `scenes/mlp_forward.py` | The video. The `KNOBS` block at the top and the `beat_*` methods. | **Edit freely.** Start here. |
| `scenes/example_new_scene.py` | A tiny starter scene to copy for new ideas. | **Edit freely.** |
| `scenes/saddle_point.py` | The second video, a 3D scene (section 8). | **Edit freely** (its `KNOBS` block, like the first video). |
| `compute/saddle.py` | The mathematics behind it: the surface, its Hessian, the classification. | Edit the `HESSIAN`, then export (section 8). |
| `scenes/studio/theme.py` | Colors, font, text sizes. One value changes the whole video. | **Edit freely** (keep blue = positive, orange = negative). |
| `compute/mlp_presets.py` | The network's weights, biases and neuron names (`xor_preset()`). | Edit, then run the export (section 4). |
| `scenes/studio/network.py` | Draws the network (`NetworkDiagram`, `Node`). Layout and encoding constants at the top. | Constants are safe; code: with care. |
| `scenes/studio/equations.py` | Builds the neuron equations from the trace. | With care. |
| `scenes/studio/trace.py` | Loads the trace; decides how numbers are rounded for display. | With care: tests depend on it. |
| `scenes/studio/beats.py` | `BeatScene`: named sections, `BEATS` filter, camera helpers. | Only when you understand it. |
| `scenes/studio/plots.py` | The small tanh graph. | With care. |
| `scenes/studio/inspect_video.py` | Prints video info and extracts frames. | Rarely. |
| `compute/mlp.py` | The forward-pass math. The web app has a copy in TypeScript that must agree. | **Change only on purpose.** |
| `compute/export.py` | Writes the generated files; holds the video's input (`VIDEO_INPUT`). | Only to change the input. |
| `tests/`, `web/src/**/*.test.ts` | Checks that numbers and drawings agree with the model. | Update when you change the story's numbers. |
| `assets/mlp/*.json`, `web/public/data/mlp.json` | **Generated** by `compute.export`. | Never edit by hand. |
| `media/` | **Generated** renders, frames, caches (not tracked by Git). | Never edit; safe to delete. |
| `.venv/`, `uv.lock`, `web/node_modules/` | **Generated** environments. | Do not touch. |
| `scenes/spike_a1.py`, `scenes/smoke_typst.py` | Small proofs that Typst equations, camera and beats work. | Read-only examples. |

## 3. The everyday loop: edit, render, look

Render a quick draft (480p, 15 fps, about 20 seconds for the six beats, which make a 55-second video):

```powershell
uv run manim -ql scenes/mlp_forward.py MlpForwardPass
```

The video is written to `media\videos\mlp_forward\480p15\MlpForwardPass.mp4`.
Open it in any player. (`-pql` also opens it automatically; that flag is the standard Manim one but I did not run it here.)
Use `-qm` for 720p at 30 fps when you want to judge motion; it lands in `...\720p30\`.

Work on one section only. Each `self.beat("name")` in the scene starts a section; list the ones you want
(the neural-network video has `question`, `build-network`, `zoom-neuron`, `back-out`, `forward-pass` and `result`):

```powershell
$env:BEATS = "zoom-neuron"
uv run manim -ql scenes/mlp_forward.py MlpForwardPass
Remove-Item Env:BEATS          # back to rendering everything
```

Skipped sections are fast-forwarded (their objects are still placed correctly) and left out of the video.
A one-section render takes about 8 seconds here.

Look at the result without watching it:

```powershell
uv run python scenes/studio/inspect_video.py media/videos/mlp_forward/480p15/MlpForwardPass.mp4
```

This prints resolution, frame rate and duration and writes `media\inspect\MlpForwardPass\contact_sheet.png`
(frames from inside every section). For specific moments: `... .mp4 --times 5 12.5 20`.

One still of the finished network with every value shown:

```powershell
uv run manim -qm -s scenes/mlp_forward.py NetworkReference
```

Checks (run these after any change that touches numbers or layout):

```powershell
uv run pytest -q
npm --prefix web test
```

Final delivery (1920x1080 at 30 fps) is rendered only when you decide the draft is done:

```powershell
uv run manim -r 1920,1080 --fps 30 scenes/mlp_forward.py MlpForwardPass
```

If a render ever looks stale or wrong, delete its cached pieces (use the same quality flag you rendered with, here `-ql`):
`uv run manim cache clear -ql scenes/mlp_forward.py MlpForwardPass`.

### Reading a scene in plain English

```python
self.play(Write(title), run_time=1.2)       # "play" one animation for 1.2 seconds: the title is written in
self.wait(1.2)                              # hold still for 1.2 seconds
self.play(FadeOut(box), FadeIn(foot))       # two animations at the same time
```

* An **object** (`Text`, `Circle`, `Line`, a whole `NetworkDiagram`) is something you can draw. Creating it does not show it.
* `self.play(animation)` shows or changes things over time. `self.add(object)` shows something instantly.
* `run_time=` is the duration in seconds. `self.wait(n)` is a pause. These two are your timing controls.
* A **VGroup** is a bag of objects you can move or fade together.
* Positions are numbers: `(0, 0)` is the screen center, x grows to the right, y grows **up**, the screen is 14.22 wide and 8 high.
  `.move_to([x, y, 0])` places; `.next_to(other, UP, buff=0.2)` places relative to another object.
* A **beat** (`self.beat("name")`) is a labelled section. It exists so you can render and inspect one part at a time.

## 4. Everyday changes

Each recipe says what to open, what to change, how to render, and how to know it worked.
Make one change at a time, render, look, and keep going only when it looks right.

### 4.1 Change a title, a caption, or an equation

**Title.** Open `scenes/mlp_forward.py`, method `beat_question`:

```python
title = Text("Two numbers in. One decision out.", font=FONT, font_size=40, color=TEXT).to_edge(UP, buff=0.7)
```

Change the words inside the quotes (`font_size` makes it bigger or smaller). Render only that beat:

```powershell
$env:BEATS = "question"; uv run manim -ql scenes/mlp_forward.py MlpForwardPass; Remove-Item Env:BEATS
```

Check: open the video, or run `inspect_video.py` and look at the contact sheet. (I tested this exact edit.)

**Caption.** `caption("Every neuron repeats this recipe.")` at the end of `beat_back_out`. Same idea.

**Equation.** The *numbers* in an equation always come from the trace; you change *which* neuron is explained, or the *notation*.

* Which neuron: in the `KNOBS` block set `FOCUS_LAYER, FOCUS_INDEX = 1, 1` (hidden layer 1, second neuron, "AND").
  The whole zoom beat now explains it: `z = 4.00*0.80 + 4.00*0.30 - 6.00 = -1.60`, activation -0.92, in orange because it is negative.
  (I rendered this. The graph labels overlap a little for negative values: see the exercise at the end.)
* Notation: in `scenes/studio/equations.py` the word `dot` inside the `substituted` string draws a middle dot;
  replace it with `times` for a cross. Typst writes math with words like `dot`, `times`, `approx`. Run `pytest` afterwards.

### 4.2 Make something disappear

```python
self.play(FadeOut(box_label), run_time=0.4)   # fade it away (tested: works)
self.remove(box_label)                        # vanish instantly
```

To remove a message for good, delete the line that creates it *and* every later line that uses it
(Python will tell you with a `NameError` if you miss one). To make it vanish at a chosen moment, put the
`FadeOut` line right before the `self.wait(...)` where it should disappear. Other exits: `Uncreate(obj)` (undraws a line), `ShrinkToCenter(obj)`.

### 4.3 Colors, positions, sizes, timing, transitions

| Want to change | Where | How |
|---|---|---|
| A color everywhere | `scenes/studio/theme.py` | e.g. `POS = "#4ea8ff"`. I changed it to teal and every blue item followed. |
| Text size | the `font_size=` of that `Text(...)` | bigger number = bigger text |
| Where something sits | `.move_to([x, y, 0])`, `.shift(UP * 0.5)`, `.next_to(...)` | y grows upward |
| The zoomed view | `KNOBS`: `ZOOM_CENTER`, `ZOOM_WIDTH`, `EQUATION_CENTER`, `PLOT_CENTER` | see 4.6 |
| Network spacing and node size | `scenes/studio/network.py`: `NODE_SPACING`, `NODE_RADIUS`, `LAST_LAYER_X`, ... | constants at the top |
| How long a step takes | `run_time=` on that `self.play(...)` | seconds |
| A pause | `self.wait(0.5)` | seconds |
| Speed profile of a move | `rate_func=linear` (or `smooth`, `there_and_back`) on `self.play` | default is smooth |
| One thing after another | separate `self.play(...)` lines | |
| Several things at once | several animations in one `self.play(a, b, c)` | |
| Staggered | `LaggedStart(*[Create(e) for e in edges], lag_ratio=0.1)` | smaller ratio = more overlap |

Typical transitions in this video: `FadeIn`, `FadeOut`, `Create` (draw a line), `GrowFromCenter`, `Write` (text), `TransformFromCopy`
(a copy flies from one place to another and becomes something else), `Indicate` (a pulse).

### 4.4 Change one weight and regenerate the trace

The weights live in `compute/mlp_presets.py`, in `xor_preset()`:

```python
weights = [
    # layer 1: OR, AND, NAND, NOR of (x1, x2)
    [[4, 4], [4, 4], [-4, -4], [-4, -4]],
```

Each inner list is one neuron: its weights from `[x1, x2]`. To weaken the OR neuron's second input change the first row to `[4, 1]`.

```powershell
uv run python -m compute.export     # recompute and rewrite the generated files
uv run pytest -q                    # check
npm --prefix web test               # the web app reads the same files
uv run manim -ql scenes/mlp_forward.py MlpForwardPass
```

What I saw when I did exactly this (then undid it):

* **Before** running the export, two tests fail: "committed trace/golden is up to date". That is the alarm that says *you changed the model but forgot to export*.
* **After** the export, the OR neuron's `z` becomes 1.50 and the probability changes slightly. Two tests still fail on purpose:
  `test_numbers_quoted_in_the_storyboard` (the video quotes `4.00 x 0.80 + 4.00 x 0.30 - 2.00 = 2.40`) and
  `test_prediction_is_derived_from_the_output` (it pins the probability near 0.99998). They guard the story. When you
  *intend* a new story, update those expected numbers; otherwise undo your change.
* The video then shows the new numbers by itself; no scene code changes.

Undo an experiment: `git checkout -- compute/mlp_presets.py assets web/public` (this throws away uncommitted changes in those paths).
The input numbers (0.8 and 0.3) are `VIDEO_INPUT` in `compute/export.py`.

### 4.5 Add neurons or layers

A network is a list of weight matrices. `weights[l]` has one row per neuron of layer `l+1` and one column per neuron of layer `l`.
Network `[2, 4, 3, 2]` has matrices of shape 4x2, 3x4 and 2x3.

**Add a neuron to hidden layer 1** (4 -> 5 neurons), five edits in `xor_preset()`:

1. `layer_sizes`: `[2, 5, 3, 2]`
2. layer-1 weights: add a row, e.g. `[2, -2]`
3. layer-1 biases: add a value, e.g. `0`
4. layer-2 weights: **every row gets one more column** (`[1.5, 0, 1.5, 0]` becomes `[1.5, 0, 1.5, 0, 0]`)
5. `neuron_labels`: add a name to the hidden-layer-1 list

**Add a layer** (I tried a third hidden layer of 2 neurons): add a new weight matrix of shape (2 x 3) in the right place,
change the *next* matrix's width to 2 (the output matrix becomes 2x2), add a bias list `[0, 0]`, add `2` to `layer_sizes`, and add
`None` to `neuron_labels` for the unnamed layer.

Then export and test as in 4.4. What happened in my experiments:

* The model, trace, web files and diagram all updated by themselves; layer count and x positions are computed from `layer_sizes`.
* The extra neuron crowded the picture: the top node touched the layer title and the bottom row came close to the caption.
  Fix by lowering `NODE_SPACING` or moving `TITLE_Y` in `scenes/studio/network.py`, then look at `NetworkReference`.
* Tests failed where they pin the old shape (`test_trace_loader_matches_the_file` expects `(2, 4, 3, 2)`; the "text rows stay
  clear of the network" layout check failed with five neurons; the saturation bound failed with the new layer). Update the pinned
  values on purpose, or revert.
* Scene code that names a neuron by position (`FOCUS_INDEX`) keeps pointing at *position*, not at the neuron's name.

### 4.6 Change the camera

In the `KNOBS` block of `scenes/mlp_forward.py`:

```python
ZOOM_CENTER = (-1.7, 0.2)   # where the camera looks
ZOOM_WIDTH = 10.4           # how much fits across the screen; smaller = closer
```

The scene uses two helpers from `scenes/studio/beats.py`:

```python
self.play(self.zoom_to(ZOOM_CENTER, ZOOM_WIDTH), other_animations)   # move the camera
self.play(self.zoom_reset())                                         # back to the full view
```

Check: render `BEATS="zoom-neuron"`. I tried `ZOOM_WIDTH = 7.0`: too close, the input nodes and equation were cut off.
Zoom too far in and objects leave the frame; zoom out and text gets small. Move `EQUATION_CENTER` and `PLOT_CENTER` along with the camera.

### 4.7 Create a new scene from the existing parts

Copy the starter and render it:

```powershell
Copy-Item scenes\example_new_scene.py scenes\my_scene.py
uv run manim -ql scenes/my_scene.py MyFirstScene
```

(Rename the class if you like; the last name on the command is the class.) The starter loads the trace, builds a
`NetworkDiagram`, and lights up the input layer and hidden layer 1 with their real values. Things you can import:

```python
from studio.network import NetworkDiagram          # the network; .nodes[layer][i], .edges, .weight_labels(...)
from studio.equations import neuron_equations      # the four equation stages for one neuron
from studio.plots import ActivationPlot            # the tanh graph
from studio.trace import ForwardTrace              # the numbers
from studio.theme import POS, NEG, TEXT, FONT      # the look
from studio.beats import BeatScene                 # named sections + zoom helpers
```

Always look at the first result with `inspect_video.py` before changing more.

## 5. The same studio for other kinds of visualization

### A. Scientific simulations (a model that changes over time)

Keep the same split as the neural network: **a model computes numbers; a scene draws them.**

1. `compute/lif.py` (pure Python, no Manim): a function that simulates and returns arrays.
2. A small script or export step saves the arrays (JSON or `.npz`) under `assets/`.
3. The scene loads the arrays and draws them. It never contains the equations of the model.

A leaky integrate-and-fire neuron is a good example (textbook model with illustrative parameters; label it "simulated"):

```python
def simulate_lif(current, dt=0.1, tau=10.0, v_rest=-65.0, v_thresh=-50.0, v_reset=-65.0, resistance=10.0):
    v = v_rest
    voltages, spikes = [], []
    for step, i in enumerate(current):
        v += dt * (-(v - v_rest) + resistance * i) / tau   # voltage leaks toward rest, input current pushes it up
        if v >= v_thresh:                                  # threshold crossed: spike, then reset
            spikes.append(step * dt)
            v = v_reset
        voltages.append(v)
    return np.array(voltages), spikes
```

and a scene that draws the *computed* voltage (I rendered this; no LaTeX needed):

```python
axes = Axes(x_range=[0, T, 20], y_range=[0, 20, 5], x_length=10, y_length=4, axis_config={"include_numbers": False})
trace = VMobject(color=BLUE).set_points_as_corners([axes.c2p(t, v - V_REST) for t, v in zip(times[::5], volts[::5])])
self.play(Create(trace), run_time=4, rate_func=linear)      # the line draws itself in the simulated order
```

The same numbers can drive other pictures: a dot riding the curve (`MoveAlongPath`), a circle whose color follows the voltage
(`sign_color`-style mapping), flashes at the stored `spikes` times (`Flash`, `ShowPassingFlash` on an edge), or a network whose nodes
light up according to a stored activity array. Rule of thumb: if you would need to know the model to understand the scene code, the model belongs in `compute/`.
Label simulated results as simulated (see `.claude/rules/neuro-data.md`).

### B. 3D graphs and spatial pictures

Manim Community draws 3D with `ThreeDScene`: the screen is a camera looking at a 3D world, and you control where it looks.

```python
class SurfaceDemo(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes(x_range=[-3, 3, 1], y_range=[-3, 3, 1], z_range=[-1.5, 1.5, 1])
        surface = Surface(                                       # a surface = a function (u, v) -> a 3D point
            lambda u, v: axes.c2p(u, v, np.sin(u) * np.cos(v)),
            u_range=[-3, 3], v_range=[-3, 3], resolution=(24, 24), fill_opacity=0.8,
        )
        self.set_camera_orientation(phi=65 * DEGREES, theta=-45 * DEGREES)   # phi: tilt from the top, theta: turn around
        self.add(axes, surface)
        self.begin_ambient_camera_rotation(rate=0.3)             # slow spin while time passes
        self.wait(2)
```

I rendered this as a still and a short video. Notes: the surface function is just maths (`sin(u)*cos(v)`), `resolution` trades smoothness for
render time (draft with small numbers), `self.move_camera(phi=..., theta=...)` animates the camera to a new angle, and
`add_fixed_in_frame_mobjects(text)` keeps captions flat on screen. Cairo's 3D can sort overlapping surfaces badly; keep draft meshes coarse.

**3D video vs interactive 3D.** A Manim 3D scene is a *video*: you choose every camera move; the viewer watches. An interactive 3D
visualization lets the viewer rotate and inspect it; that is a browser app (Three.js in `web/`, see `.claude/rules/web-interactive.md`), fed by
exported meshes or arrays. Use video for an explanation with a chosen story; use the browser for exploration. Real brain meshes belong in the browser or in PyVista, not Manim.

### C. Editing and refining finished videos

An MP4 is a flat sequence of pictures: the words in it are just pixels. The source scene is the recipe that made it.

* **Prefer editing the source and re-rendering** for anything the scene controls: wording, numbers, colors, timing, camera, order.
  It stays sharp at any resolution, the numbers stay tied to the model, and re-rendering one beat takes seconds.
  Changing a caption in an MP4 means covering pixels with new text and it would be wrong the moment the model changes.
* **Use a separate video editor** for things that are not animation: recording and syncing narration, cutting between several clips,
  background music, color grading, intro/outro cards, final YouTube export. The project makes this easy: Manim output is silent, and
  `uv run manim --save_sections ...` is documented to write one clip per beat as well (an option of the installed version; not tried here), so an editor can move whole sections around. Manim can also write
  subtitles from `self.add_subcaption(...)` calls (present in the installed version; not tried here).

## 6. Learning path

| Stage | First exercise | Expected result | Status in this project |
|---|---|---|---|
| 1. Edit the video | Change the title text and the `INPUT_VALUE_SCALE` knob; render `BEATS="question"`. | A new title and larger or smaller input numbers in the 5-second clip. | **Supported now.** |
| 2. Change the network | Weaken one weight (4.4), then add a neuron (4.5); export, test, render `NetworkReference`. | New numbers in the picture; a crowded layout you then fix with `NODE_SPACING`. | **Supported now.** Structure changes need test updates. |
| 3. Simulation to animation | Put `simulate_lif` in `compute/lif.py`, save voltages to `assets/lif/trace.json`, draw them in `scenes/lif.py`. | A graph that draws itself with 4 spikes, each reaching the dashed threshold line. | **Needs a small amount of new code** (model, export, scene, a test); the pattern and the drawing code are shown above. |
| 4. 3D surface | Copy the `SurfaceDemo` above to `scenes/surface.py`; change the formula and `phi`/`theta`. | A surface that spins; a different formula gives a different shape. | **Needs a new scene file only**; example verified. |
| 5. Compose longer videos | Make two scenes with the starter, then join them. | One video of several parts. | **Partly supported:** one long scene with beats works today. Joining separate scene files needs a decision (a video editor, or a small concatenation script); not built. |

## 7. Safe habits

* Commit before experimenting: `git status`, then `git add ...` and `git commit -m "..."` when it looks right. `git diff` shows what you changed;
  `git checkout -- <file>` undoes uncommitted edits to that file (it cannot be undone).
* Change one thing, render a draft, look. Draft quality (`-ql`) only; final quality when you decide.
* Never hand-edit generated files. If a number looks wrong, fix the model or the preset and re-export.
* Use `uv run ...` for everything; do not `pip install`.
* Equations use Typst (no LaTeX). Some Manim classes default to LaTeX: I checked `DecimalNumber` and the axis numbers of `Axes`, and did not test `Matrix`,
  `Table` or `BarChart`; treat those as unavailable until tested. This project uses `MathTypst`, `DecimalNumber(..., mob_class=MathTypst)` and `Axes(include_numbers=False)`.
* Error messages: the last lines of the traceback name the file and line. `NameError` = a name you removed or misspelled; `KeyError`/`IndexError` after a
  structure change = something still expects the old shape (section 4.5).

## 8. A second video: "Why is a saddle point not a minimum?" (3D)

A 35-second 3D explainer built from the same studio, with the same split: **`compute/saddle.py` does the mathematics, `scenes/saddle_point.py` only draws it.**

```powershell
uv run python -m compute.saddle                              # write assets/saddle/saddle_trace.json (generated)
uv run manim -ql scenes/saddle_point.py SaddlePoint          # draft; 3D is slower: about a minute at 480p15
uv run manim -qm scenes/saddle_point.py SaddlePoint          # 720p30, about two minutes
$env:BEATS = "along-x"; uv run manim -ql scenes/saddle_point.py SaddlePoint; Remove-Item Env:BEATS   # one beat: about 10 seconds
uv run python scenes/studio/inspect_video.py media/videos/saddle_point/480p15/SaddlePoint.mp4
```

The five beats are `surface`, `flat-spot`, `along-x`, `along-y`, `conclusion`. What each file owns:

* `compute/saddle.py`: the surface is `f(x, y) = 1/2 (x, y) H (x, y)^T`; `HESSIAN` is the matrix `H`. It computes the gradient at the center (0, 0), the
  curvature along x and along y, the eigenvalues, and the classification (all positive = minimum, all negative = maximum, mixed = saddle).
* `scenes/saddle_point.py`: rebuilds the surface from `H` in the trace, draws the exported slice curves, and prints the traced gradient and curvatures.
  The video asserts `classification == "saddle"`; with a matrix that is not a saddle it stops instead of telling a false story.
* `scenes/studio/beats.py`: `BeatScene3D` is the 3D twin of `BeatScene` (same beats, same `BEATS` filter, same `beats.json`).

3D camera knobs, at the top of the scene: `phi` tilts the camera (0 = straight down, 90 = level with the surface), `theta` turns it around the vertical axis.
`self.move_camera(phi=..., theta=..., run_time=...)` flies to a new view; `begin_ambient_camera_rotation(rate=...)` turns it slowly until `stop_ambient_camera_rotation()`.
Text that must stay flat on screen goes through the scene's `fixed(...)` helper (`add_fixed_in_frame_mobjects`). I tested the whole video, a single-beat render, and the tests.

Exercise: in `compute/saddle.py` change `HESSIAN` to `[[2.0, 0.0], [0.0, 2.0]]` (a bowl), run `uv run python -m compute.saddle` and `uv run pytest -q`.
Expected (I ran it, then undid it): before the export, `test_committed_trace_is_up_to_date` and three more tests fail ("you changed the model but did not export");
after the export only the three tests that pin the story's numbers still fail on purpose (curvature 2 and -2, "rises along x and falls along y"), the
classification in the trace becomes "minimum", and rendering the scene stops with the message "this video explains a saddle point", because a bowl is not a saddle.
A minimum would need a new story; this is the place to make it. Undo by changing the matrix back and running the export again.

## 9. A third video: "How does a network read a handwritten digit?" (real data)

A 95-second explainer about a real network that reads real handwritten digits. Same split as before: **`compute/digits.py` trains and exports, `scenes/digit_network.py` only draws.**

```powershell
uv run python -m compute.digits                                  # first time: downloads two small files (about 0.8 MB) into assets/data/, trains (a few seconds), writes assets/digits/digits_trace.json
uv run manim -ql scenes/digit_network.py DigitRecognition        # draft, 480p15, about 45 seconds to render
uv run manim -qm scenes/digit_network.py DigitRecognition        # 720p30
$env:BEATS = "activation-functions"; uv run manim -ql scenes/digit_network.py DigitRecognition; Remove-Item Env:BEATS   # one beat
uv run python scenes/studio/inspect_video.py media/videos/digit_network/480p15/DigitRecognition.mp4 --per-beat 4
```

The nine beats are `digit`, `network`, `one-neuron`, `what-neurons-learn`, `activation-functions`, `forward-pass`, `answer`, `mistake`, `learning`.

**Where the data comes from.** Pixels: the UCI "Optical Recognition of Handwritten Digits" dataset (Alpaydin & Kaynak, 1998, CC BY 4.0; 8 x 8 images, ink 0 to 16),
downloaded on first use and not committed (`assets/data/` is gitignored). Weights: learned by `compute/digits.py` (plain numpy, fixed seed). Forward pass: computed
from those weights. The video shows one test digit the network reads correctly and the first test digit it gets wrong. Nothing on screen is drawn by hand;
the only schematic is the thin lines from the picture to the hidden neurons, each standing for 64 weights (the video says so).

What each file owns:

* `compute/digits.py`: download, training, the choice of the example digits, the activation curves. Knobs at its top: `EPOCHS`, `LAYER_SIZES`, `SHOWCASE_CLASS`.
* `scenes/studio/digits_trace.py`: reads the JSON (`DigitsTrace`); the scene imports this and never `compute`.
* `scenes/studio/digit_network.py`: the drawing pieces: `PixelGrid` (a picture or a weight map), `weight_gallery`, `DigitNetwork`, `ProbabilityBars`.
* `scenes/digit_network.py`: the nine beats. Layout numbers sit in the knob block at its top (`PANEL_Y`, `ROW_Y`, `TABLE_X`, ...).
* `tests/test_digits.py`: recomputes the forward pass from the exported weights with numpy, checks the sums, the colors, and that text fits the frame.
  One test also re-evaluates the weights on the real test file and is skipped if you have not downloaded it.

Exercise (not run; try it and look at the still): in `scenes/digit_network.py` change `DIM_AMOUNT` from 0.88 to 0.5, then render only `one-neuron`
with the `BEATS` command above. The other neurons stay visible while one is studied. Undo by restoring 0.88.
