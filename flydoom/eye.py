"""The looming eye: the ONE sensory front end shared by every brain-like controller (the reflex now, the connectome later).

Pixels in, four numbers out, per tic and per eye (left half of the screen = left eye, right half = right eye):
    size_deg          angular size of the bright object(s) in that eye, in degrees  (later -> LPLC2, which encodes size)
    expansion_dps     how fast that size grows, degrees per second, never negative  (later -> LC4, which encodes speed)
The LPLC2-size / LC4-speed split follows Ache et al. 2019 (Curr Biol 29:1073). The rest is engineering, chosen here:

  * Threat = BRIGHT object. In take_cover the only threats are fireballs, and they are by far the brightest pixels:
    a one-off calibration (flydoom/calibrate_eye.py, which uses ViZDoom's object labels offline, never at play time)
    found 95.2 % of fireball pixels above luminance 150 and 0.003 % of all other pixels (seeds 9001-9005). Real fly looming detectors
    respond mostly to DARK expanding objects; this eye is tuned to this game, not a model of fly photoreceptors.
  * Size = diameter of a disc with the same area as all bright pixels in that half (several fireballs merge into one).
  * Pixels -> degrees with ViZDoom's 90 degree horizontal field of view (320 px -> 0.28 degree per pixel).
  * Expansion is the tic-to-tic growth, rectified, then smoothed with a 2-tic exponential filter (noise from sprites
    changing animation frame).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .game import HFOV_DEG, TICRATE

BRIGHT_LUMINANCE = 150.0  # 0-255, from the calibration above
SMOOTH_TICS = 2.0
LUMA = np.array([0.299, 0.587, 0.114])


@dataclass(frozen=True)
class EyeOutput:
    size_deg: tuple  # (left, right)
    expansion_dps: tuple  # (left, right)

    def as_list(self) -> list:
        return [round(v, 4) for v in (*self.size_deg, *self.expansion_dps)]


class LoomingEye:
    def __init__(self):
        self.reset()

    def reset(self):
        self.prev_size = None
        self.expansion = np.zeros(2)

    def bright_mask(self, rgb: np.ndarray) -> np.ndarray:
        return (rgb.astype(np.float32) @ LUMA) > BRIGHT_LUMINANCE

    def see(self, rgb: np.ndarray) -> EyeOutput:
        mask = self.bright_mask(rgb)
        h, w = mask.shape
        deg_per_px = HFOV_DEG / w
        halves = (mask[:, : w // 2], mask[:, w // 2:])
        size = np.array([2.0 * np.sqrt(half.sum() / np.pi) * deg_per_px for half in halves])
        if self.prev_size is None:
            growth = np.zeros(2)
        else:
            growth = np.maximum(size - self.prev_size, 0.0) * TICRATE
        a = 1.0 / SMOOTH_TICS
        self.expansion = (1 - a) * self.expansion + a * growth
        self.prev_size = size
        return EyeOutput(tuple(float(s) for s in size), tuple(float(e) for e in self.expansion))
