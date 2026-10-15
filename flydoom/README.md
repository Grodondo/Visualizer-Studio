# FlyDoom: what is actually doing the thinking?

An experiment on the DOOM engine (ViZDoom, scenario `take_cover`). Question: if a validated simulation of a whole fruit-fly
brain plays the game, is the brain doing the thinking, or is it the sensor and the action mapping around it?

**Short answer (10 distinct test seeds):** the model plays about as well as a brainless reflex that sees the same signal, not better.
The real wiring works as a pathway, because shuffled wiring abolishes the behaviour. This says nothing about real fly intelligence.

| Controller | Median survival [middle 50 %] |
|---|---|
| idle | 4.3 s [4.1-5.7] |
| random | 6.4 s [4.6-9.2] |
| reflex (same eye, simple rule) | 7.6 s [5.4-10.7] |
| fly (same eye, whole-brain model) | 8.0 s [4.7-11.2] |
| fly, giant fibers silenced | 8.8 s [5.8-11.7] |
| fly, shuffled wiring | identical to idle on every seed |
| scripted AI (more information) | 11.3 s [9.2-12.6] |

The fly beats the reflex on 3 of 10 seeds; the scripted AI beats the fly on 9 of 10. Ten seeds are descriptive, not significance tests.
`tests/test_closed_loop.py` and `tests/test_fly_visual.py` pin these statements to the recorded numbers.

## How the loop works (every game step, 1/35 s)

1. **Eye (engineered)**: `eye.py` turns the screen pixels into four numbers (size and expansion speed of bright objects, left and right half).
2. **Encoder (engineered)**: the left half drives the fly's left LC4 / LPLC2 looming neurons, the right half the right ones.
3. **Brain (simulated)**: the whole-brain model (Shiu et al. 2024, leaky integrate-and-fire, FlyWire connectome) runs for 1/35 s.
4. **Decoder (engineered)**: the side whose steering neurons (DNa01 / DNa02) are more active wins and the player steps that way. The direction
   comes from the literature (Rayshubskiy et al. 2025), not from our data.
5. **Tuning**: the only tuned number, the input gain, was set on calibration seeds; every other choice was fixed before any test seed was played.

Every controller sees only the RGB screen. Health and object positions are used for scoring only.

## Run it

Needs the optional dependency group `flydoom` (`uv run --group flydoom ...`). Commands in the order you would use them:

```powershell
uv run --group flydoom python -m flydoom.seeds            # screen seeds so episodes are distinct (~20 s)
uv run --group flydoom python -m flydoom.trials           # baselines on the test seeds (~8 min)
uv run --group flydoom python -m flydoom.model_data       # download Shiu's model data, verified by hash (~195 MB)
uv run --group flydoom python -m flydoom.validate_model   # the model vs. its published results (~2 min)
uv run --group flydoom python -m flydoom.fly_trials       # the fly and its controls (~13 min)
uv run --group flydoom pytest -q                          # all tests
```

Video footage (the "FlyPlays" video, `scenes/flydoom_fly.py`):

```powershell
uv run --group flydoom python -m flydoom.fly_export             # the fly's episodes and simulated spikes (~2 min)
uv run --group flydoom python -m flydoom.fly_export --shuffle   # the shuffled-wiring control
uv run --group flydoom python -m flydoom.activity_frames        # spike images, one per game step (~3 min)
uv run --group flydoom python -m flydoom.activity_frames --shuffle
uv run python -m flydoom.brain_frames                           # anatomy image
uv run manim -ql scenes/flydoom_fly.py FlyPlays                 # draft, about 70 s
```

`activity_frames` also needs the FlyWire annotation file in `assets/data/flywire/` (gitignored).

## Files

| File | Role |
|---|---|
| `game.py`, `eye.py`, `controllers.py` | the game wrapper, the looming eye, the baseline controllers |
| `metrics.py`, `seeds.py`, `trials.py` | scoring (written before the first run), seed screening, running controllers |
| `model_data.py`, `lif.py`, `cells.py` | model data, the simulator (`Brain` steps it one game step at a time), FlyWire cell ids |
| `validate_model.py`, `brian_check.py` | checks against Shiu et al.'s published results and their Brian2 code |
| `fly_controller.py`, `fly_trials.py` | the closed loop and its controls (real wiring, shuffled wiring, giant fibers silenced) |
| `fly_export.py`, `activity_frames.py`, `brain_frames.py` | footage and images for the videos |
| `human.py` | records your own episodes (personal data, stays in `assets/data/flydoom/human/`, gitignored) |

Results and footage live in `assets/flydoom/` (see its README). The longer write-up with all caveats is `GUIDE.md`, sections 13, 13b and 13c.

## What it does not show

- Anything about real fly intelligence or behaviour: this is one published simulation (identical neurons, predicted transmitter signs)
  wrapped in our own eye and decoder.
- Anything about escape jumps: silencing the giant fibers did not matter because the decoder does not read them.
- That tuning is reliable: the chosen gain gave 16.5 s on the 5 calibration seeds but 8.0 s on the test seeds.

## Provenance and licences

- Game: ViZDoom 1.3.1 (MIT per its README), scenario `take_cover`, Freedoom graphics (BSD-3-Clause). "DOOM" is a trademark; say
  "the DOOM engine with Freedoom". **The take_cover map's licence is not stated separately: an open item before anything is published.**
- Model: Shiu et al. 2024 (MIT), code pinned at commit 91bdd1e. Connectome and positions: FlyWire (CC BY 4.0).
- Neuron activity in the videos is **simulated**; the anatomy is **measured**; the eye, encoder and decoder are **engineered**. The videos label which is which.
