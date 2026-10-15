"""Controllers. Each gets the RGB screen only (see flydoom/game.py) and returns "left", "right" or "none" every tic.

Letters match the plan in GUIDE.md section 13:
  E  idle          never moves
  E  constant      always steps left (a classic "keep moving" baseline: fireballs aimed at where you were miss)
  E  random        a fresh random choice every DECISION_TICS tics, from its own seeded generator
  C  reflex        ONLY the looming eye's four numbers: step away from the eye with the stronger expansion
  F  scripted      a conventional game AI: every bright blob's position and growth, more information than the eye gives
     scripted_v1   the first version (kept for the record): it started strafing left, so threats nearly always appeared
                   on its right, so it kept going left: frame-identical to "constant" on 9 of 10 test seeds, and on the
                   tenth (1006) it pressed right for 6 tics just before dying, at the same tic as "constant"
  (A, B, D: the connectome controllers, Milestone 2-3, will receive the same LoomingEye output as the reflex.)
Fixed parameters were chosen on CALIBRATION seeds only (flydoom/trials.py), never on the test seeds.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .eye import LoomingEye

DECISION_TICS = 4  # random controller: hold each choice this long (about 0.11 s)


class Controller:
    name = "base"
    uses = "nothing"  # what information it gets, for the report
    last_eye = None

    def reset(self, seed: int = 0):
        pass

    def act(self, rgb) -> str:
        raise NotImplementedError


class Idle(Controller):
    name, uses = "idle", "nothing"

    def act(self, rgb):
        return "none"


class Constant(Controller):
    name, uses = "constant", "nothing"

    def __init__(self, direction: str = "left"):
        self.direction = direction

    def act(self, rgb):
        return self.direction


class Random(Controller):
    name, uses = "random", "nothing (its own random numbers)"

    def __init__(self, offset: int = 0):
        self.offset = offset

    def reset(self, seed: int = 0):
        # its own generator, derived from the episode seed: different moves per episode, identical on a re-run
        self.rng = np.random.default_rng(seed + self.offset)
        self.t, self.current = 0, "none"

    def act(self, rgb):
        if self.t % DECISION_TICS == 0:
            self.current = ["none", "left", "right"][int(self.rng.integers(0, 3))]
        self.t += 1
        return self.current


class Reflex(Controller):
    """Step away from the side whose eye reports the faster expansion, while it exceeds `threshold` (deg/s).
    Below threshold: keep doing what it did (so a dodge is finished rather than stopped halfway)... unless it has been
    quiet for `settle_tics`, then stand still."""

    name, uses = "reflex", "the looming eye's 4 numbers"

    def __init__(self, threshold: float = 20.0, settle_tics: int = 10):
        self.threshold, self.settle_tics = threshold, settle_tics
        self.eye = LoomingEye()

    def reset(self, seed: int = 0):
        self.eye.reset()
        self.current, self.quiet = "none", 0

    def act(self, rgb):
        out = self.eye.see(rgb)
        self.last_eye = out.as_list()
        left, right = out.expansion_dps
        if max(left, right) > self.threshold:
            self.current = "right" if left > right else "left"
            self.quiet = 0
        else:
            self.quiet += 1
            if self.quiet > self.settle_tics:
                self.current = "none"
        return self.current


class ScriptedV1(Controller):
    """FIRST VERSION, kept for the record. Find every bright blob, take the biggest, step to the side away from it;
    if nothing is visible keep strafing."""

    name, uses = "scripted_v1", "every bright blob's position and size"

    def __init__(self, min_size_px: int = 6):
        self.min_size_px = min_size_px
        self.eye = LoomingEye()

    def reset(self, seed: int = 0):
        self.eye.reset()
        self.current = "left"

    def act(self, rgb):
        out = self.eye.see(rgb)  # recorded for the timeline only; the decision uses the blobs below
        self.last_eye = out.as_list()
        mask = self.eye.bright_mask(rgb)
        labels, n = ndimage.label(mask)
        if n:
            sizes = ndimage.sum(mask, labels, range(1, n + 1))
            big = int(np.argmax(sizes))
            if sizes[big] >= self.min_size_px:
                _, cx = ndimage.center_of_mass(mask, labels, big + 1)
                self.current = "right" if cx < mask.shape[1] / 2 else "left"
        return self.current


class Scripted(Controller):
    """Conventional game AI from pixels (REVISED after v1 collapsed into "constant"). Find bright blobs; the biggest one
    is the nearest fireball. If it is bigger than last tic (approaching) and at least `min_area_px`, step away from its
    side of the screen and keep stepping for `hold_tics`; otherwise stand still.
    Uses each blob's exact horizontal position, which the looming eye does not report."""

    name, uses = "scripted", "the biggest bright blob's position and growth"

    def __init__(self, min_area_px: float = 25.0, hold_tics: int = 8):
        self.min_area_px, self.hold_tics = min_area_px, hold_tics
        self.eye = LoomingEye()

    def reset(self, seed: int = 0):
        self.eye.reset()
        self.prev_area, self.current, self.hold = 0.0, "none", 0

    def act(self, rgb):
        mask = self.eye.bright_mask(rgb)
        labels, n = ndimage.label(mask)
        area, cx = 0.0, None
        if n:
            sizes = ndimage.sum(mask, labels, range(1, n + 1))
            big = int(np.argmax(sizes))
            area = float(sizes[big])
            cx = ndimage.center_of_mass(mask, labels, big + 1)[1]
        if cx is not None and area >= self.min_area_px and area > self.prev_area:
            self.current = "right" if cx < mask.shape[1] / 2 else "left"
            self.hold = self.hold_tics
        elif self.hold > 0:
            self.hold -= 1
        else:
            self.current = "none"
        self.prev_area = area
        return self.current


def all_controllers(reflex_threshold: float = 20.0, scripted_area: float = 25.0) -> list[Controller]:
    return [Idle(), Constant("left"), Random(), Reflex(reflex_threshold), ScriptedV1(), Scripted(scripted_area)]
