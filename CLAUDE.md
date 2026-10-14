# Visualizer Studio

Goal: build explanatory visualizations in the style of 3Blue1Brown, covering mathematics,
neural networks, graphs/algorithms, and neuroscience (brain data and brain stimulation).
Two kinds of output: **rendered animation videos** and **interactive web visualizations**.

I am learning. Keep code readable, comment the *why*, and explain each tool choice in 1-2 sentences.

<!-- Maintainer note (stripped from context): facts below verified 2026-10-14. Re-verify versions
     before large changes: PyPI "manim", PyPI "pyvista", "npm view three version". -->

## 1. Pick the tool BEFORE writing code

| I want... | Use | Do NOT use |
|---|---|---|
| Animated explainer video (math, NN forward pass, graph algorithm) | **Manim Community Edition** (`from manim import *`) | ManimGL, ManimML |
| Something the viewer can interact with (sliders, rotate 3D, hover) | **Web**: Vite + TypeScript, D3 (2D), Three.js (3D) | Manim |
| Static figure for a post or paper | Matplotlib or Plotly | Manim |
| Real brain data (EEG/MEG, MRI, fMRI) | MNE-Python, nilearn, PyVista | hand-drawn or invented brains |
| Brain-stimulation fields (TMS/tDCS) | SimNIBS to simulate, PyVista to view | made-up field values |
| Cinematic 3D render | Blender, only if I ask | |

- Video vs interactive unclear -> ask me exactly one question. With no other detail, a request
  for an "animation" or "visualizer" means a Manim video.
- Pipeline for science projects: compute in Python -> export small files (npz/JSON/GLB) ->
  render in Manim (video) or the web (interactive). Keep compute and rendering in separate modules.
- Details live in `.claude/rules/` (`manim.md`, `web-interactive.md`, `neuro-data.md`); they load
  automatically when you touch matching files.

## 2. Environment (Manim CE 0.22.0, released 2026-10-08)

- Python >= 3.11. Manage everything with **uv**: `uv add <pkg>`, `uv run <cmd>`. Never `pip install` globally.
- Detect my OS before giving shell or install commands; do not assume Linux.
- Run `uv run manim checkhealth` first when anything fails to render.
- Equations: LaTeX (MiKTeX / MacTeX / TeX Live) is optional and only needed for `MathTex`/`Tex`.
  Typst is the LaTeX-free alternative: `uv add "manim[typst]"`. Current install docs list no
  separate FFmpeg step for Manim CE (checkhealth confirms).
- Your memory of Manim predates versions 0.20-0.22. Treat memory as unreliable for CLI flags,
  config files, `Code`, and Typst. Verify against the installed source
  (`uv run python -c "import manim; help(manim.Create)"`) or https://docs.manim.community/en/stable.
  Never invent a class or argument.

## 3. Commands (Manim CE 0.22)

```
uv run manim -pql scenes/x.py MyScene        # fast draft (480p15) and open it
uv run manim -ql -s scenes/x.py MyScene      # last frame only -> PNG in media/images/ (layout check)
uv run manim -qh scenes/x.py MyScene         # final 1080p60; only when I ask
uv run manim --format gif -ql scenes/x.py MyScene
uv run manim cache clear scenes/x.py MyScene
uv run manim init project --default          # scaffold a new project
```

Removed in 0.22, do not use: `--write_to_movie`, `-g/--save_pngs`, `-i/--save_as_gif`,
`--force_window`, `--flush_cache`, `--custom_folders`. Use `--format`, `-p` (preview),
`-l` (live OpenGL window), `manim cache clear`. The config section `[ffmpeg]` is now `[media]`.
`-o` is rejected when rendering several scenes at once.

## 4. Workflow for every request

1. **Plan in 3-6 bullets**: audience, the ONE key idea, scene list, tools chosen. For anything
   bigger than one scene, show the plan and wait for my OK.
2. **Build the smallest version first** (static layout, then motion).
3. **Verify visually.** You cannot watch video, so render a still (`-ql -s`) and look at the PNG.
   To inspect a mid-scene state, temporarily put `return` after that animation, then remove it.
   Check: nothing overlaps, nothing is off-screen, text is readable, colors mean the same thing everywhere.
4. **Verify it runs**: Manim renders without errors; web `npm run build` passes.
5. **Report honestly**: what you rendered and viewed, what you could not check. Never say "works"
   from code alone.
6. High-quality renders (`-qh`) only on request.

## 5. Hard rules

- `from manim import *` is Manim CE. `from manimlib import *` is ManimGL. Never mix them. The
  3b1b/videos repo is ManimGL code and will not run on CE without translation.
- Do not use or recommend `manim-ml` (ManimML): its last PyPI release is April 2023 and it is
  not verified against Manim 0.22. Build network diagrams from `Circle`, `Line`, `VGroup`.
- Default to the Cairo renderer. Use OpenGL only if I ask; the CE docs for it are incomplete.
- Manim outputs silent clips. Narration and editing happen in a video editor. Add
  `manim-voiceover` only if I ask.
- Do not fabricate data. A schematic must be labeled schematic. For "real" network behavior,
  compute actual activations from a tiny trained model.
- Human neuroscience data: use public sample datasets only. Never commit subject-level data.
  Brain-stimulation visuals are explanatory, not medical guidance.
- Ambiguity about "brain stimuli" (sensory stimuli -> neural response, single-neuron simulation,
  or non-invasive brain stimulation) -> ask which, before building.
- Pin versions (`uv add` records them; use exact versions in `package.json`). Do not use `latest` in code.
- Do not change the Python version, delete `media/`, or add heavy dependencies without telling me.

## 6. Layout

```
scenes/    Manim scenes: one topic per file, one Scene class per idea
web/       Vite + TypeScript interactive visualizations
neuro/     MNE / nilearn / PyVista / SimNIBS pipelines and their exports
assets/    small images, meshes, data   (large data is gitignored)
media/     Manim output                 (gitignored)
```

Always gitignore: `media/`, `.venv/`, `node_modules/`, `__pycache__/`, large datasets.

## 7. Definition of done

Runs from a clean command, looks right in the rendered still, the key idea is visible within the
first few seconds, constants (colors, sizes, seeds) sit at the top of the file, and you have told
me how to re-run it.
