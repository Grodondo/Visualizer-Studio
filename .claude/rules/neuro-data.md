---
paths:
  - "neuro/**"
---

# Neuroscience data and brain visualization

## First: pin down what "brain stimuli" means (ask me one question if unclear)

1. **Sensory stimulus -> neural response** (EEG/MEG evoked responses, fMRI activation maps): MNE-Python and nilearn.
2. **Single-neuron or small-circuit dynamics** (voltage, spikes): a simulator plus custom Manim or web visuals.
   Brian2, NEST, and NEURON are candidates but were NOT verified; read their current docs before adopting one.
3. **Non-invasive brain stimulation** (TMS, tDCS, tACS): SimNIBS electric-field simulation.

## Tools (verified 2026-10-14)

| Need | Tool | Notes |
|---|---|---|
| EEG/MEG processing and plots | MNE-Python | current main branch requires Python >= 3.11 |
| Brain maps on volumes and cortical surfaces | nilearn | surface API changed at 0.11 (`SurfaceImage`, `PolyMesh`); read the docs for the installed version |
| 3D meshes and custom scenes | PyVista 0.48.x (Python 3.10-3.14) | export glTF for Three.js (`Plotter.export_gltf`; check the docs) |
| TMS/tDCS electric fields | SimNIBS | separate installer (download requires registration); follow its docs, do not pip-install it into this project's environment |

## Data rules

- Use public sample datasets only (MNE `sample`, `nilearn.datasets` fetchers). Put the dataset name and
  its citation in the script header.
- Never commit subject-level or identifiable data. Downloads go to `assets/data/` (gitignored).
- Save derived results as compact `.npz` or `.json`, with one README line saying how they were generated.

## Scientific honesty

- Label every figure as **measured** (dataset), **simulated** (tool and parameters), or **schematic**.
- Never invent field strengths, latencies, or coordinates. Take them from computation or a cited source.
- Stimulation visuals are explanatory, not for planning treatment. Add a short note when presenting them.
- Colormaps: perceptually uniform (viridis-like) for magnitudes; diverging only for signed data centered on zero.

## Handoff to visuals

- Video: render PyVista frames (or export arrays), then bring them into Manim as `ImageMobject`s,
  or draw schematic brains from Manim shapes.
- Interactive: export GLB and load it with Three.js (`web/`).
- Keep compute scripts independent of the renderer, with one command that regenerates every export.
