# Visualizer Studio

Explanatory visualizations in the style of 3Blue1Brown: mathematics, neural networks, neuroscience and one experiment on whether a
simulated fly brain is "doing the thinking". Two kinds of output: **rendered animation videos** (Manim Community Edition) and
**interactive web visualizations** (Vite + TypeScript).

Every number on screen comes from a computed or measured file, and every video says which parts are **measured**, **simulated**,
**computed** or **schematic**. Nothing is drawn by hand and presented as data.

## The videos

| Video | Scene | What it shows | Data |
|---|---|---|---|
| Forward pass of a small network | `scenes/mlp_forward.py` | One input flowing through a network, number by number | computed |
| Why a saddle point is not a minimum | `scenes/saddle_point.py` (3D) | Curvature along two directions at a flat spot | computed |
| How a network reads a handwritten digit | `scenes/digit_network.py` | A real network trained on real digits, its weights, activation functions, a mistake | UCI optical digits (measured), trained weights (simulated) |
| Can we see the brain preparing to move? | `scenes/eeg_motor.py` | Real EEG during real and imagined movement | public EEG dataset (measured) |
| What happens in your brain when you react to a punch? | `scenes/punch_reaction.py` | Anatomy, then real EEG on a real brain | ERP CORE + fsaverage (measured) |
| Inside a real brain's wiring | `scenes/connectome.py` | The whole fruit-fly connectome, then its escape circuit | FlyWire (measured) |
| FlyDoom draft | `scenes/flydoom_draft.py` | Real gameplay beside the measured fly brain (brain not connected) | game + FlyWire |
| FlyPlays | `scenes/flydoom_fly.py` | A simulated fly brain plays the game; the real wiring vs shuffled wiring vs a brainless reflex | game, FlyWire anatomy (measured), neuron activity (simulated) |

The last experiment, with its results and what it does **not** show, is described in [flydoom/README.md](flydoom/README.md).
Short version: the simulated fly plays about as well as a simple reflex that sees the same signal, not better; the real wiring works as a
pathway (shuffled wiring abolishes it); this says nothing about real fly intelligence.

## Quick start

Needs Python 3.11 or newer and [uv](https://docs.astral.sh/uv/). Equations use Typst, so no LaTeX install is needed.

```powershell
uv sync                                            # create the environment from uv.lock
uv run manim checkhealth                           # run this first if anything fails to render
uv run manim -ql scenes/mlp_forward.py MlpForwardPass   # draft render (480p15)
uv run pytest -q                                   # tests
```

Common render options: `-ql` draft, `-qm` 720p30, `-s` last frame only as a PNG (the fastest way to check a layout), `-p` open the result.
Final 1080p renders are done on request only: `-r 1920,1080 --fps 30`. Output goes to `media/` (gitignored).

Most videos read a small generated file in `assets/`. The first run of a data-based video needs its compute step once, for example
`uv run python -m compute.digits`; each scene's docstring lists the exact commands, sizes of downloads and how long they take.
Downloaded datasets go to `assets/data/` (gitignored). The FlyDoom experiment needs the optional group:
`uv run --group flydoom ...`.

## How it is organized

```
compute/    the mathematics and data pipelines (numpy, MNE, ...). Writes small JSON/npz files into assets/
assets/     the small exported files the scenes read (large data is gitignored)
scenes/     Manim scenes: one topic per file, one Scene class per video
  studio/   shared pieces: beats, theme (colors), network and equation drawing, plots, review tools
flydoom/    the DOOM-engine experiment: game wrapper, eye, controllers, the fly-brain simulator, trials
web/        interactive visualizations (Vite + TypeScript)
neuro/      reserved for neuroscience pipelines
tests/      Python tests
media/      rendered output and review stills (gitignored)
```

The rule that keeps it learnable: **compute modules do the numbers, scenes only draw them.** A scene never imports `compute/`; it reads the
exported file. Constants (colors, sizes, positions) sit at the top of each file.

Useful tools while building a video:

```powershell
$env:BEATS = "react"; uv run manim -ql scenes/flydoom_fly.py FlyPlays; Remove-Item Env:BEATS    # render one named beat
uv run python scenes/studio/inspect_video.py media/videos/<module>/480p15/<Scene>.mp4 --per-beat 3   # pull review frames out of a render
```

Videos are made of named *beats*, so one part can be rendered and checked on its own. `inspect_video.py` extracts frames at chosen times,
because a video cannot be judged from code alone.

## Web

```powershell
npm --prefix web install
npm --prefix web run dev      # interactive forward-pass playground
npm --prefix web test
npm --prefix web run build
```

## Learning more

[GUIDE.md](GUIDE.md) is the beginner guide: which file does what, the edit-render-look loop, everyday changes (a caption, a color, a weight),
how to create a new scene, and one section per video with its commands and an exercise. [CLAUDE.md](CLAUDE.md) holds the project rules
used when working with an AI assistant.

## Data, sources and licences

- Each video's data source, citation and licence is in its scene docstring and its `GUIDE.md` section (for example the UCI digits are
  CC BY 4.0 and are credited on screen; FlyWire anatomy is CC BY 4.0; the fly-brain model is Shiu et al. 2024, MIT).
- No subject-level human data is committed. Public sample datasets only; your own recordings stay in gitignored folders.
- Brain-stimulation or brain-activity visuals are explanatory, not medical guidance.
- **Not yet cleared for publishing:** the licence of the ViZDoom `take_cover` map used in the FlyDoom videos is not stated separately.
  "DOOM" is a trademark: say "the DOOM engine with Freedoom".
- This repository has no LICENSE file yet; add one before sharing the code.
