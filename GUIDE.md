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

## 10. A fourth video: "Can we see the brain preparing to move?" (real EEG)

A two-minute explainer built on real EEG from 20 volunteers who opened and closed a fist, or only imagined it.
Same split as before: **`compute/eeg_motor.py` downloads and analyses the data, `scenes/eeg_motor.py` only draws it.**

```powershell
uv run python -m compute.eeg_motor                               # first time: ~300 MB into assets/data/eegbci/ (about 3 minutes), then writes assets/eeg_motor/eeg_motor_trace.json
uv run python -m compute.eeg_motor 1 2 3                         # quick prototype on volunteers 1-3 only (overwrites the trace; run without numbers to restore all 20)
uv run manim -ql scenes/eeg_motor.py EegMotor                    # draft, 480p15, about 40 seconds to render
$env:BEATS = "average"; uv run manim -ql scenes/eeg_motor.py EegMotor; Remove-Item Env:BEATS   # one beat
uv run python scenes/studio/inspect_video.py media/videos/eeg_motor/480p15/EegMotor.mp4 --per-beat 3
uv run pytest -q tests/test_eeg_motor.py
```

Beats: `question`, `task`, `raw`, `rhythm`, `average`, `map`, `limits`, `recap`.

**Where the data comes from.** PhysioNet EEG Motor Movement/Imagery Dataset v1.0.0 (Schalk et al. 2004; doi 10.13026/C28G6P; ODC-By 1.0),
volunteers 1-20, runs 3/7/11 (real left/right fist) and 4/8/12 (imagined). The files are fetched from PhysioNet's AWS open-data mirror
(`physionet-open`), because physionet.org served them at about 3 minutes per file; I checked one file byte-for-byte against physionet.org
and against PhysioNet's SHA256SUMS, and MNE checks every file against its own SHA-256 list before using it.

**Processing choices** (all constants at the top of `compute/eeg_motor.py`, all copied into the trace under `"processing"`):
standard 10-05 positions, average reference, 1-40 Hz filter, epochs -1.5 to 4.5 s around each target, wavelet power 4-35 Hz,
% change from the second before the target, mu = 8-13 Hz, beta = 13-30 Hz, numbers averaged 0.5-3.5 s after the target.
Rejection: an epoch is dropped if its largest swing on the 15 central electrodes is an outlier for that volunteer (median + 3 robust SD).
A first attempt with a fixed 150 µV limit dropped every trial of volunteer 3; this dataset's amplitudes are much larger than such limits assume.
That change was made before looking at any result. Nobody is excluded for "not showing the effect". 1,655 of 1,800 trials (92%) were kept; the lowest for any volunteer, hand and condition was 17 of 24.

**What the data shows, and the caveats the video states:**

* mu power over motor cortex falls about 27% while moving and about 17% while only imagining (18 and 15 of 20 volunteers show a fall).
* For real movement the fall is about as large on both sides of the head; for imagery the opposite side falls more in 14 of 20.
  The textbook "opposite side only" pattern is not what these untrained volunteers show; McFarland et al. (2000) found the same.
* The markers are when the target appeared, not when the hand moved (no movement sensor), so this data cannot say whether
  the change came *before* the movement. The video cites self-paced studies for that (about 2 s before; Magnani et al. 1998 measured 2.15 and 1.78 s).
* A brief rise right after the target is strongest beside the eyes (F7/F8/FT7): most likely eye movements toward the target.
  Slow waves (4-7 Hz) during real movement are strongest above the eyes (Fp1/Fpz/Fp2). Neither is a motor rhythm. Eye artifacts add power,
  so they can hide a fall in mu power but not create one.
* EEG sums synchronized currents of very many neurons over centimetres of cortex; it cannot show single neurons or spikes (Buzsáki et al. 2012).

What each file owns:

* `compute/eeg_motor.py`: download, analysis, export. Knobs: `SUBJECTS`, `BANDS`, `ERD_WINDOW`, `REJECT_MADS`, `EXAMPLE_SUBJECT`, `EXAMPLE_WINDOW`.
* `scenes/studio/eeg.py`: `EegTrace` (reads the JSON), `HeadMap` (electrodes at their real positions), `power_color`, `color_bar`, plot helpers.
* `scenes/eeg_motor.py`: the eight beats. Layout knobs at the top (`RAW_WINDOW`, `PLOT_CENTER`, `MAP_X`, ...). `BAND = "beta"` tells the same story with the beta rhythm
  (captions that quote numbers follow the data; the "mu" wording in captions does not, so edit those too).
* `tests/test_eeg_motor.py`: checks the export, re-runs volunteer 1 against it (skipped if not downloaded), and checks that every sentence the
  video says about the data is still true (for example "imagined is smaller than real", "within half a second", "beside the eyes").
  If you change the analysis and one of those fails, the video would now say something false: change the sentence, not the test.

Exercise (not run): set `RAW_WINDOW = (20.0, 35.0)` at the top of the scene to show a different stretch of the same recording
(a left-fist and a right-fist target), then render only `raw` and `rhythm`. The gold labels follow the recorded markers by themselves.

## 11. A fifth video: "What happens in your brain when you react to a punch?" (real EEG on a real brain)

A 2-minute explainer: an illustrated jab, the visual-to-motor pathway on a real (template) human cortex, then one person's real EEG,
including estimated cortical activity played in slow motion in sync with the measured brain wave.
Three steps, each its own module: **`compute/punch_reaction.py` analyses, `compute/brain_render.py` draws the brain images, `scenes/punch_reaction.py` composes the video.**

```powershell
uv run python -m compute.punch_reaction       # first time: ERP CORE (92 MB) + fsaverage template (~430 MB) into assets/data/; then ~15 s
uv run python -m compute.brain_render         # 190 brain images + manifest.json into media/brain_frames/punch_reaction/ (~25 s)
uv run python -m compute.brain_render --check # a few test stills into media/check/ (to try a new view or colormap)
uv run manim -ql scenes/punch_reaction.py PunchReaction
$env:BEATS = "seeing"; uv run manim -ql scenes/punch_reaction.py PunchReaction; Remove-Item Env:BEATS
uv run python scenes/studio/inspect_video.py media/videos/punch_reaction/480p15/PunchReaction.mp4 --per-beat 3
uv run pytest -q tests/test_punch_reaction.py
```

Beats: `punch`, `pathway`, `experiment`, `seeing`, `preparing`, `fighters`, `recap`. A corner tag always says what is on screen:

| Tag | What it is |
|---|---|
| ILLUSTRATION | the boxers: a drawing, not a recording |
| ANATOMY | the fsaverage template cortex (FreeSurfer) with Desikan-Killiany atlas areas, which only approximate functional areas; timings in that beat are published averages (Foxe & Simpson 2002; Woods et al. 2015) |
| MEASURED | ERP CORE Flankers, subject 001: one person, 400 trials of "press the button for the middle arrow" |
| ESTIMATED | where on the cortex the measured EEG most likely came from (dSPM, Dale et al. 2000) on the template brain |

**Data and licence.** ERP CORE (Kappenman et al. 2021, NeuroImage 225:117465) as shipped by MNE. Its licence is **CC BY-SA 4.0**:
if you publish the video, credit the authors and release the video under CC BY-SA 4.0 too.

**Processing choices** (constants at the top of `compute/punch_reaction.py`): 0.1-30 Hz, 256 Hz, average reference; correct presses
200-1000 ms after the arrows (the paper's window); epochs with outlier swings (median + 3 robust SD) dropped; noise covariance from the
0.2 s before the arrows (plain estimate plus MNE's 10 % regularisation, because the "shrunk" estimator needs scikit-learn);
dSPM with MNE's standard settings (SNR 3, loose 0.2, depth 0.8); LRP = opposite-side minus same-side of C3/C4, onset = 50 % of its peak.

**What the data shows, and the limits the video states:**

* 346 of 400 presses correct; median 414 ms; 383 ms when the outer arrows agree, 445 ms when they conflict.
* Visual waves over the back of the head at about 130 ms and 200 ms. Times count from the computer's marker; the screen lights up slightly
  later (the ERP CORE authors corrected for this; whether MNE's copy includes their correction is not documented, so the video rounds).
* Estimated activity: occipital cortex first, then parietal cortex, strongest on the right, up to ~220 ms. **After 220 ms the strongest estimated
  sources jump to deep midline areas (unlabelled medial wall, cingulate), which 30 scalp electrodes cannot resolve, so the video stops there.**
* Motor preparation (LRP) reaches half its size ~90 ms before the press. **I checked whether the estimate can place it in left motor cortex for
  the right hand: it cannot** (the hand-difference map is split about evenly between hemispheres), and the video says so instead of showing a lucky frame.
* Boxers: no boxer data. The video compares this choice task with published simple reaction times and cites Mori et al. (2002): experts gain mainly by anticipation.

What each file owns:

* `compute/punch_reaction.py`: analysis and export (`punch_trace.json`, `sources.npz`). Knobs: `FILTER`, `RT_RANGE`, `INVERSE`, `SOURCE_WINDOW`, `TRUSTED_UNTIL`, `REGIONS`.
* `compute/brain_render.py`: the cortex renderer (matplotlib, no extra dependency). Knobs: `IMAGE_PX`, `VIEWS` (camera angles), `STAGES`
  (which atlas areas, colors), `ACTIVITY_VIEWS`, `CMAP`. Reusable for any activity on fsaverage, e.g. the movement video.
* `scenes/studio/punch.py`: `PunchTrace`, `BrainFrames` (with `player`, an image that follows a time tracker), and the `Boxer` drawing.
* `scenes/punch_reaction.py`: the beats. Knobs: `BRAIN_MAIN`, `ACTIVITY_BRAINS`, `SLOW_MOTION`, plot positions.
* `tests/test_punch_reaction.py`: export checks plus one test per data sentence of the narration.

Exercise (not run): in `compute/brain_render.py` set `ACTIVITY_VIEWS = ("back", "left_back")`, run the renderer, then render only `seeing`.
The slow-motion brains now show the left side; the caption "strongest on the right" is still true, and the right side is no longer in view, so reword it.

## 12. A sixth video: "Inside a real brain's wiring" (the fruit-fly connectome)

A 2-minute tour of a complete brain wiring diagram: 12 real neurons, a pull-back to all 139,255 neurons of one fruit-fly brain
(with a full turn in 3D), how it was mapped, and a zoom into the escape circuit that lets a fly jump away from a looming threat,
drawn from its real synapses. Same split: **`compute/connectome.py` downloads and extracts, `compute/fly_render.py` draws the point clouds,
`scenes/connectome.py` composes.**

```powershell
uv run python -m compute.connectome           # first time: 27 MB + 50 MB into assets/data/flywire/, and the 317 MB synapse table streamed once (~20 s, not stored)
uv run python -m compute.fly_render           # 156 images + manifest.json into media/fly_frames/ (~1 minute)
uv run python -m compute.fly_render --check   # three quick stills into media/check/
uv run manim -ql scenes/connectome.py Connectome
$env:BEATS = "circuit"; uv run manim -ql scenes/connectome.py Connectome; Remove-Item Env:BEATS
uv run python scenes/studio/inspect_video.py media/videos/connectome/480p15/Connectome.mp4 --per-beat 3
uv run pytest -q tests/test_connectome.py
```

Beats: `few`, `whole`, `mapped`, `circuit`, `limits`, `recap`.

**Data (all CC BY 4.0, one adult female fly, brain only, no nerve cord):**

| File | Source | Pinned by |
|---|---|---|
| neuron positions, classes, types | Schlegel et al. 2024, Supplemental file 1, `flywire_annotations` release **v2.1.0** (the version in the Nature papers) | git tag + byte size |
| connections (pairs with >= 5 synapses, per brain region) | FlyWire Codex v783 export, public bucket `flywire-data` | storage generation + MD5 (the bucket is not a versioned release) |
| synapse positions | same export, `synapse_coordinates.csv.gz`; only the circuit's 36,391 synapses are kept | storage generation + MD5 of the stream |

**What is drawn, and the simplifications the video says out loud:**

* A neuron = one dot at its cell body. Neurons whose cell body is outside the brain (mostly sensory neurons) are drawn at an anchor point on the neuron.
* Lines between the first 12 neurons are straight; real contacts are made along branches. In the zoom, each dot is a real synapse position.
* Positions were checked against the EM volume's published size (sections ~750 um wide; the neurons span 818 x 395 x 279 um).
* Cell types: a FlyWire type where one exists, otherwise the matched hemibrain type (that is how LC4 and LPLC2 are typed in v2.1.0).

**What the data shows:** the right giant fiber (DNp01) receives 4,737 synapses from 275 neurons; its two largest input types are the looming
detectors LPLC2 (62 neurons, 539 synapses) and LC4 (49 neurons, 428). Their output synapses form dense hubs where the giant fiber's branch reaches in,
but only ~3-6% of those synapses are onto the giant fiber: the hubs contact many neurons. The synapse file and the connection table agree on these counts
exactly for the right giant fiber (the left one differs by 4 synapses of 3,762). Function (looming detection, size vs speed, the jump) is from the cited
experiments (Klapoetke et al. 2017; Ache et al. 2019; von Reyn et al. 2014), not from the wiring itself; the video says so.

**Caveats the video states:** wiring is not activity; only chemical synapses are mapped (no gap junctions); one fly; transmitters are predicted
(the two giant fibers' predictions disagree, glutamate 0.27 vs acetylcholine 0.32, both low confidence); the giant fiber's target is in the nerve cord, outside this map.

**Not used, and why:** FlyWire's neuron skeletons are public on Zenodo but only as one 5.4 GB file of four ~1.3 GB chunks, so even a few neurons
would mean downloading gigabytes; Codex's own download page needs a login.

What each file owns:

* `compute/connectome.py`: downloads with checks, the circuit selection (`FOCUS_TYPE`, `FOCUS_SIDE`, `FIRST_GROUP`, `LOOMING_TYPES`), the synapse stream.
* `compute/fly_render.py`: the point-cloud images. Knobs: `CLASS_COLORS`, `POINT_SIZE`, `HALF_WIDTH_UM`, `TURNTABLE_STEP`, `SYNAPSE_COLORS`, `ZOOM_PAD_UM`.
* `scenes/studio/fly.py`: `FlyTrace`, `FlyFrames` (placing data on screen, image sequences for the turntables).
* `scenes/connectome.py`: the beats. Knobs: `FIRST_BOX`, `TURN_SECONDS`, `CIRCUIT_TURN_SECONDS`, colors.
* `tests/test_connectome.py`: export checks, renderer/scene camera agreement, and one test per data sentence of the narration.

Exercise (not run): set `FOCUS_SIDE = "left"` in `compute/connectome.py`, run both compute steps, and render `circuit`. The numbers in the captions
follow the data by themselves (on the left LC4 comes first, 40 neurons, then LPLC2, 37); the zoom window moves to the other side automatically. `tests/test_connectome.py` will then fail on purpose where it pins the right side ("LPLC2, then LC4"): that sentence of the story changed.

## 13. Experiment: "What is actually doing the thinking?" (flydoom, Milestone 1)

A separate, optional experiment on the DOOM engine. Everything lives in `flydoom/` (game, eye, controllers, metrics, trials,
human recorder, export), `scenes/flydoom_draft.py` and `tests/test_flydoom.py`. It needs the optional dependency group
`flydoom` (ViZDoom 1.3.1, which brings gymnasium and pygame-ce); nothing else in the studio uses it, and no existing package version changed.

```powershell
uv run --group flydoom python -m flydoom.calibrate_eye   # how the eye's brightness threshold was chosen (~6 s)
uv run --group flydoom python -m flydoom.seeds           # screen seeds for distinct episodes (~20 s)
uv run --group flydoom python -m flydoom.trials          # all controllers x 10 test seeds, each run twice (~8 min) + the footage
uv run python -m flydoom.brain_frames                    # brain images for the draft (~30 s)
uv run --group flydoom python -m flydoom.human --practice  # you: one practice episode, nothing saved
uv run --group flydoom python -m flydoom.human           # you: the 5 seeded episodes (saved locally, ignored by Git)
uv run manim -ql scenes/flydoom_draft.py FlyDoomDraft    # the ~53 s draft (about 70 s to render at 480p)
uv run --group flydoom pytest -q tests/test_flydoom.py
```

**The game.** ViZDoom scenario `take_cover` with Freedoom graphics: imps throw fireballs, the player can only step left or right; we
add a 60 s cap. Synchronous mode, one action per tic (35 per second), seeded with `set_seed`. Same seed + same actions = bit-identical
frames (checked for every episode), and recorded `.lmp` demos replay to the same episode, also when recorded at another resolution.

**Who sees what.** Every controller gets only the RGB screen. Health, positions and object labels are used for scoring only.
The *looming eye* (`flydoom/eye.py`) turns pixels into four numbers per tic: angular size and expansion speed of bright objects in the
left and right half of the screen. The reflex (C) uses only these numbers, and the fly connectome (Milestone 2) will get exactly the same
four. The brightness threshold was chosen once, offline, using the game's object labels (95.2 % of fireball pixels, 0.003 % of others).

**Metrics** (`flydoom/metrics.py`, written before the first run; two definitions revised after it, marked REVISED in the file):
survival time, hits and damage, share of time moving, switches per second, and reaction time from a new fireball's first appearance
(game truth) to the first key press when standing still. The first run defined events with the eye at the reflex's own threshold,
which made the reflex's reaction time zero by construction; that was replaced by the game-truth definition.

**Seeds (corrected).** Many ViZDoom seeds give the *same* take_cover game: among seeds 1-120, 69 gave one identical
fireball pattern. The first seed sets were unscreened, so test seeds 1007, 1008 and 1010 were one game and calibration seeds
9001-9004 were another. `flydoom/seeds.py` now screens seeds (first seeds upward whose idle episode differs from every kept seed)
and writes `assets/flydoom/seeds.json` with each episode's hash: calibration 9001, 9005, 9008, 9010, 9014; test 1001-1006, 1009,
1011-1013. `results.json` records `distinct_episodes` per controller (10 of 10 for all), and a test fails if seeds ever collide.

**Results on 10 distinct test seeds** (median [25-75 %] survival): idle 4.3 s [4.1-5.7], always-left 4.6 [4.1-6.0], random
6.4 [4.6-9.2], reflex 7.6 [5.4-10.7], scripted AI 11.3 [9.2-12.6]. The scripted AI beats idle and always-left on 10/10 seeds and the
reflex on 7/10; the reflex beats idle on 7, always-left on 8, random on only 5. Ten seeds: descriptive, not significance tests.
Before the seed correction the reflex looked better (9.6 s), because one long 17.6 s game was counted three times.
Damage does not separate controllers (every episode ends in death). `scripted_v1` (kept for the record) behaved like "always left".
The eye reacts to a fireball *appearing*, not only to it approaching (median detection delay 0 tics).

**The draft video** (`scenes/flydoom_draft.py`, about 53 s, beats `open`, `brain`, `circuit`, `turn`, `compare`):
cold open on the game, which then shrinks to an inset; the measured fly brain (seen from behind, so its left is your left) becomes
the centerpiece: optic lobes, central brain, LC4/LPLC2 cell bodies, descending neurons, a push into the giant-fiber circuit's measured
synapses (a slow turn), one turn of the whole brain, then a compact survival comparison. Footage: the reflex's test episodes in seed
order until >= 40 s (`assets/flydoom/footage.json`: seeds 1001-1005, 44.0 s, no repeats, every death kept). The brain is NOT
simulated and does not react to the game; the drawn causal chain is eye -> reflex -> keys. Legend chips mark engineered input
(dashed), actions (keycaps), measured anatomy (dots) and "simulated activity: none".

Brain images for this cut: `uv run python -m flydoom.brain_frames` (reuses compute/fly_render.py without changing it; writes
media/flydoom_brain/: back views, optic/central layers, and the circuit close-up colored by role: right-eye inputs amber, giant fiber white).

**Changing the draft.** Knobs at the top of `scenes/flydoom_draft.py`: `GAME_BIG`, `GAME_INSET`, `BRAIN` (center/width), `CONTROLS_Y`,
`TIMELINE`, colors (`LEFT_C`, `RIGHT_C`, `OUT_C`), timings (`T_OPEN`, `T_SHRINK`, `CIRCUIT_TURN_S`, `BRAIN_TURN_S`, `TURN_ZOOM_START`),
comparison rows (`COMPARE_ROWS`). Footage: `CLIP_CONTROLLER` and `CLIP_MIN_SECONDS` in `flydoom/export.py`, then
`uv run --group flydoom python -m flydoom.export`. Which fly cells are marked: `IO_TYPES` in `flydoom/export.py`. Brain highlight layers
and close-up colors: `LAYERS`, `CIRCUIT_COLORS` in `flydoom/brain_frames.py`. The gameplay plays in real time; the brain shots follow a
fixed schedule that never reads game events.

**Licences.** ViZDoom's README states MIT for code original to ViZDoom; the bundled ZDoom engine has mixed licences; neither the wheel
nor the repository ships a LICENSE file, and the take_cover map's licence is not stated separately (open item before publishing).
Freedoom graphics: BSD-3-Clause (keep the notice, credit Freedoom). "DOOM" is a trademark: say "the DOOM engine with Freedoom".
FlyWire anatomy: CC BY 4.0. Your own recordings are personal data and stay in `assets/data/flydoom/human/` unless you decide otherwise.

### 13b. Milestone 2: the fly brain model, validated (not connected to the game)

A whole-brain leaky integrate-and-fire model of the fruit fly: a numba port of **Shiu et al. 2024** (*Nature* 634:210;
code github.com/philshiu/Drosophila_brain_model, MIT, pinned at commit 91bdd1e) running on FlyWire connectivity (CC BY 4.0).
Every neuron is the same simple model; weights are synapse counts x 0.275 mV, signed by *predicted* neurotransmitter
(acetylcholine excites; GABA and glutamate inhibit). It is a model with known simplifications, not a recording.

```powershell
uv run --group flydoom python -m flydoom.model_data       # Shiu's data files, verified by git hash (~195 MB, assets/data/shiu/)
uv run --group flydoom python -m flydoom.validate_model   # ~2 minutes -> assets/flydoom/model_validation.json
uv run --group flydoom pytest -q tests/test_fly_model.py
# optional, ~6 minutes: run Shiu's unmodified Brian2 code next to the port
uv run --no-project --python 3.11 --with brian2==2.9.0 --with "numpy<2.4" --with pandas --with pyarrow --with joblib python -m flydoom.brian_check brian
uv run --group flydoom python -m flydoom.brian_check compare
```

Files: `flydoom/model_data.py` (download, verify, sparse matrix), `flydoom/lif.py` (the simulator, ~1 s of compute per simulated
second of the whole brain on this PC), `flydoom/cells.py` (FlyWire ids), `flydoom/validate_model.py`, `flydoom/brian_check.py`.
New optional dependencies in the `flydoom` group: numba 0.68.0 (its llvmlite 0.50.0) and pyarrow 26.0.0; nothing existing changed.

**Does the port reproduce the published model?** Yes, within noise:
* Brian2's exact input spikes replayed into the port give the same target spikes (264 = 264) and the same membrane trace (0.0 mV apart).
* Whole brain, Shiu's sugar example (v630, 30 trials each): rates correlate r = 0.9999 with a fresh Brian2 run and with Shiu's own
  published output file; MN9 91.2 Hz (port) vs 93.0 Hz (Brian2) vs 93.3 Hz (published file); 5 of 334 active neurons differ by more
  than 3 standard errors (about 1 expected by chance).
* One rule had to be learned from Brian2, not from the paper: input that reaches a neuron during its 2.2 ms refractory period is
  lost (g is "unless refractory"). Without it the port fired ~20 % too much. Found by the replay test; `tests/test_fly_model.py` keeps it.

**What the model does on v783** (10 trials of 1 s each; "shuffled" = same neurons, same numbers of inputs and outputs, random partners):
| Stimulus | Real wiring | Shuffled |
|---|---|---|
| 20 sugar neurons, 200 Hz | MN9 (feeding motor neuron) 91 Hz right, 62 Hz left | 0 Hz |
| right LC4 + LPLC2 (152 cells), 50 / 100 / 200 Hz | right giant fiber 125 / 164 / 206 Hz; left giant fiber 73-102 Hz | 0 Hz |
| same | steering neurons DNa01 and DNa02: LEFT only (21-41 Hz), right 0 Hz | 0 Hz |
| nothing | no neuron fires | |
Sugar -> MN9 is the paper's validated result. Looming -> giant fiber is a model prediction consistent with anatomy and experiments
(Ache et al. 2019; von Reyn et al. 2014), not something Shiu et al. validated. One of Shiu's 21 sugar ids no longer exists in v783
(proofreading); it is left out, not replaced by a guess.

**Open questions for Milestone 3 (not decided yet):** which way the fly should "turn" from left-only DNa activity needs the
literature on DNa01/DNa02 function and a check that FlyWire's "side" labels are the fly's own left/right; how the four eye numbers
become Poisson rates (an engineered encoder that must be fixed before any test seed is run); and the real-time budget (~29 ms of
simulated time per game tic at ~1 s of compute per simulated second: about real time).

### 13c. Milestone 3: the fly model plays (closed loop), with controls

`flydoom/fly_controller.py` closes the loop, every tic (1/35 s): pixels -> the SAME looming eye the reflex uses -> Poisson input
to LC4 + LPLC2 (left screen half -> the fly's left cells; rate = min(200 Hz, gain x expansion)) -> the validated whole-brain model
(Milestone 2) runs 1/35 s -> spikes of the steering neurons DNa01 + DNa02 per side, filtered 100 ms -> step to the side whose
DNa are 10 Hz more active. Direction rule from the literature: DNa01/DNa02 fire before IPSILATERAL steering, and steering follows
the right-left DNa02 difference (Rayshubskiy et al. 2025, eLife). "Steer" becomes "step" because this player cannot turn.
The giant fibers are recorded but drive no key (take_cover has no jump). Everything except the model is engineering, written down
before the test seeds were played; the only tuned number, the input gain, was chosen on the 5 calibration seeds (2.0 Hz per deg/s).

```powershell
uv run --group flydoom python -m flydoom.fly_trials        # ~13 minutes -> assets/flydoom/fly_results.json
uv run --group flydoom pytest -q tests/test_closed_loop.py
```
`lif.Brain` steps the model one tic at a time; a test checks it gives exactly the spikes of the validated one-shot kernel.
The whole loop is reproducible (a rerun of a test seed is bit-identical). The model runs about 1.4x slower than real time.

**Results on the 10 distinct test seeds** (median [25-75 %] survival; baselines from section 13):
idle 4.3 [4.1-5.7] · random 6.4 [4.6-9.2] · reflex 7.6 [5.4-10.7] · **fly 8.0 [4.7-11.2]** · fly with giant fibers silenced 8.8 [5.8-11.7] ·
**fly with shuffled wiring 4.3, identical to idle on every seed** · scripted AI 11.3 [9.2-12.6].

What this shows, and what it does not:
* The real wiring turns a looming signal into steering away from it (82 % of first moves are away; the end-to-end test shows it on
  synthetic frames). With the same neurons and numbers of connections but random partners, nothing reaches the steering neurons:
  the "fly" never moves. So in this model the *specific* wiring carries the signal, not just the amount of it.
* The fly model is NOT better than the brainless reflex that gets the same four numbers: it beats it on 3 of 10 seeds (1 tie).
  The scripted AI, which gets more information (each blob's position), beats the fly on 9 of 10. The game is largely solved, or not,
  by the sensor and the action mapping; the 139,255-neuron model adds a working, but not a better, link between them.
* Calibration showed gain 2.0 with 16.5 s median survival on the calibration seeds; on the test seeds it gave 8.0 s. Five seeds were too
  few to tune on reliably: the drop is the honest size of the uncertainty here.
* Silencing the giant fibers does not hurt (the decoder does not read them), so this experiment says nothing about escape jumps.
* Not shown by this experiment: anything about real fly behaviour or intelligence. The model is one published simulation with
  uniform neurons and predicted transmitter signs; the eye, the encoder and the decoder are ours. `tests/test_closed_loop.py`
  pins each sentence above to the recorded numbers.

### 13d. The video of Milestone 3: "FlyPlays"

`scenes/flydoom_fly.py` (class `FlyPlays`, about 69 s at 480p15, beats `open`, `loop`, `react`, `footage`, `shuffle`, `compare`). It subclasses the
Milestone 1 draft to reuse its game view, timeline and image helpers; the draft itself is unchanged apart from splitting out `index_clips()`.

```powershell
uv run --group flydoom python -m flydoom.fly_export              # fly episodes + simulated spikes (~2 min)
uv run --group flydoom python -m flydoom.fly_export --shuffle    # the shuffled-wiring control on the first test seed
uv run --group flydoom python -m flydoom.activity_frames         # one spike image per game step, 1831 images (~3 min)
uv run --group flydoom python -m flydoom.activity_frames --shuffle
uv run manim -ql scenes/flydoom_fly.py FlyPlays
uv run --group flydoom pytest -q tests/test_fly_visual.py
```

What you see: the fly's own episodes (test seeds in order until 40 s: 1001-1005, 52.3 s, the same rule as Milestone 1), the chain
eye -> brain -> decoder -> keys as live bars and keycaps, and the model's SIMULATED spikes drawn at the neurons' measured FlyWire positions
(input cells cyan/amber by side, steering neurons white, all others magenta; the color says what a neuron is, never how strongly it fired).
`loop` freezes on one moment and explains the four steps. `react` replays the first threat step by step (seed 1001, step 76: a threat on the
right, the right input cells fire, one step later the left steering neurons fire, one step after that the key LEFT). `footage` plays the rest
at 2.5x speed (shown on screen). `shuffle` replays seed 1001 with shuffled wiring: 20,640 input spikes, 0 steering spikes, no key ever pressed.
`compare` shows the survival table above, with the two counts (fly beats reflex on 3/10, scripted AI beats fly on 9/10) computed from the data.
The 5-seed tuning problem is deliberately left out of the video; it stays in section 13c.

Every caption that makes a claim is built from the data, and `FlyPlays.read_story` asserts it before rendering (threat side, which steering
side fires, that the key matches), so changing the data cannot silently leave a false caption. `tests/test_fly_visual.py` also checks that
each clip is its recorded episode (same frame hash), that spike files and images have one entry per step, and that the controls and legend fit.
Knobs at the top of the scene: `FREEZE_BEFORE`, `SLOW`, `FAST`, `HOLD`, `CAPTION_SIZE`, `STEER_MAX_HZ`, `SHOWN` (rows of the comparison);
spike marker sizes are `SIZES` in `flydoom/activity_frames.py` (regenerate the images after changing them). An overview of the whole
experiment is in `flydoom/README.md`.
