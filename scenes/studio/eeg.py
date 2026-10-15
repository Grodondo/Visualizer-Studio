"""Loader and drawing pieces for the EEG movement/imagery video (assets/eeg_motor/eeg_motor_trace.json).

The trace is written by `uv run python -m compute.eeg_motor`; this module never computes EEG, it only draws exported numbers.
Encodings:
  power change     orange = the rhythm got weaker (negative %), blue = stronger (positive %), dark = no change.
                   This is the studio's sign rule (blue positive, orange negative), saturating at +-POWER_VMAX %.
  executed         white solid line          imagined   teal solid line
  opposite side    of the head from the hand ("contralateral"), same side drawn dashed and dimmer
  cue period       a faint gold band: the 4.1 s the target was on screen
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Arc,
    Circle,
    DashedVMobject,
    Dot,
    Line,
    ManimColor,
    Polygon,
    Rectangle,
    Text,
    VGroup,
    VMobject,
    interpolate_color,
)

from .theme import BG, FONT, GOLD, MUTED, NEG, POS, TEXT

EXECUTED = ManimColor(TEXT)
IMAGINED = ManimColor("#3ad1c0")
CONDITION_COLOR = {"executed": EXECUTED, "imagined": IMAGINED}
CONDITION_NAME = {"executed": "moved", "imagined": "imagined"}
CUE_COLOR = ManimColor(GOLD)
POWER_VMAX = 40.0  # % change at which the map colors saturate
NEUTRAL = ManimColor("#2a313b")  # "no change": a gray just lighter than the background


@dataclass(frozen=True)
class EegTrace:
    raw: dict

    @classmethod
    def load(cls, path: Path) -> "EegTrace":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    # ---- shortcuts ----
    @property
    def times(self) -> np.ndarray:
        return np.array(self.raw["times"])

    @property
    def freqs(self) -> np.ndarray:
        return np.array(self.raw["freqs"])

    @property
    def n_subjects(self) -> int:
        return len(self.raw["dataset"]["subjects"])

    @property
    def electrodes(self) -> dict:
        return self.raw["electrodes"]

    @property
    def example(self) -> dict:
        return self.raw["example"]

    def condition(self, name: str) -> dict:
        return self.raw["conditions"][name]

    def course(self, condition: str, band: str, side: str) -> tuple[np.ndarray, np.ndarray]:
        c = self.condition(condition)["courses"][band]
        return np.array(c[f"{side}_mean"]), np.array(c[f"{side}_sem"])

    def window_values(self, condition: str, band: str, side: str) -> np.ndarray:
        return np.array(self.condition(condition)["courses"][band][f"{side}_window_per_subject"])

    def window_mean(self, condition: str, band: str, side: str = "contra") -> float:
        return float(self.window_values(condition, band, side).mean())

    def n_with_drop(self, condition: str, band: str) -> int:
        """Volunteers whose opposite-side power was lower during the task than before the cue."""
        return int((self.window_values(condition, band, "contra") < 0).sum())

    def n_contra_stronger(self, condition: str, band: str) -> int:
        """Volunteers whose drop was larger on the opposite side of the head than on the same side."""
        return int((self.window_values(condition, band, "contra") < self.window_values(condition, band, "ipsi")).sum())

    def topo(self, condition: str, band: str, hand: str) -> dict:
        return self.condition(condition)["topo"][band][hand]

    def strongest_channels(self, condition: str, check: str, n: int = 3) -> list[str]:
        by = self.condition(condition)["artifact_checks"][check]["by_channel"]
        return sorted(by, key=by.get, reverse=True)[:n]

    def trial_counts(self) -> tuple[int, int]:
        """(kept, total) cues over every volunteer, hand and condition."""
        kept = total = 0
        for row in self.raw["trial_counts"]:
            for cond in ("executed", "imagined"):
                kept += sum(row[cond]["kept"].values())
                total += sum(row[cond]["total"].values())
        return kept, total


def power_color(value: float) -> ManimColor:
    """Diverging color centered on zero: orange for a drop, blue for a rise, saturating at POWER_VMAX."""
    t = min(abs(value) / POWER_VMAX, 1.0)
    return interpolate_color(NEUTRAL, ManimColor(NEG if value < 0 else POS), t)


# ---- the head with its electrodes --------------------------------------------------------------------

class HeadMap(VGroup):
    """Top view of the head (nose up): outline, nose, ears and the 64 electrodes at their real 10-05 positions.

    Electrode coordinates come from the export in units of the head radius (the Fpz-T7-Oz-T8 ring is radius 1).
    """

    def __init__(self, electrodes: dict, radius: float = 2.0, dot_radius: float = 0.075, values: dict | None = None):
        super().__init__()
        self.radius = radius
        self.names = list(electrodes)
        self.outline = Circle(radius=radius, color=MUTED, stroke_width=3)
        nose = VMobject(color=MUTED, stroke_width=3).set_points_as_corners(
            [[-0.16 * radius, 0.985 * radius, 0], [0, 1.13 * radius, 0], [0.16 * radius, 0.985 * radius, 0]])
        ears = VGroup(*[Arc(radius=0.13 * radius, start_angle=a, angle=np.pi, color=MUTED, stroke_width=3)
                        .move_to([s * 1.04 * radius, 0, 0]) for s, a in ((-1, np.pi / 2), (1, -np.pi / 2))])
        self.head = VGroup(self.outline, nose, ears)
        self.dots = VGroup()
        self.dot_of = {}
        for name, (x, y) in electrodes.items():
            d = Dot([x * radius, y * radius, 0], radius=dot_radius, color=MUTED)
            self.dots.add(d)
            self.dot_of[name] = d
        self.add(self.head, self.dots)
        if values is not None:
            self.color_by(values)

    def color_by(self, values: dict) -> "HeadMap":
        for name, d in self.dot_of.items():
            d.set_fill(power_color(values[name]), opacity=1).set_stroke(width=0)
        return self

    def label(self, name: str, direction=UP, font_size: float = 22, color=TEXT) -> Text:
        return Text(name, font=FONT, font_size=font_size, color=color).next_to(self.dot_of[name], direction, buff=0.08)


def color_bar(width: float = 3.0, height: float = 0.16, steps: int = 40, font_size: float = 18) -> VGroup:
    """A horizontal key for power_color, from -POWER_VMAX (weaker) to +POWER_VMAX (stronger)."""
    cells = VGroup(*[
        Rectangle(width=width / steps, height=height, stroke_width=0, fill_opacity=1,
                  fill_color=power_color(-POWER_VMAX + (i + 0.5) * 2 * POWER_VMAX / steps))
        for i in range(steps)]).arrange(RIGHT, buff=0)
    left = Text(f"weaker  −{POWER_VMAX:.0f}%", font=FONT, font_size=font_size, color=MUTED).next_to(cells, LEFT, buff=0.15)
    mid = Text("0", font=FONT, font_size=font_size, color=MUTED).next_to(cells, DOWN, buff=0.06)
    right = Text(f"+{POWER_VMAX:.0f}%  stronger", font=FONT, font_size=font_size, color=MUTED).next_to(cells, RIGHT, buff=0.15)
    return VGroup(cells, left, mid, right)


# ---- time series ---------------------------------------------------------------------------------------

def polyline(xs, ys, color, stroke_width: float = 2.0) -> VMobject:
    """A line through the given scene points (xs, ys already in scene units)."""
    return VMobject(color=color, stroke_width=stroke_width).set_points_as_corners(
        np.c_[xs, ys, np.zeros(len(xs))])


def dashed(mob: VMobject, num_dashes: int = 60) -> DashedVMobject:
    return DashedVMobject(mob, num_dashes=num_dashes, dashed_ratio=0.55)


def band_polygon(axes, times, low, high, color, opacity: float = 0.25) -> Polygon:
    """The shaded area between two curves (mean - sem, mean + sem) on the given axes."""
    pts = [axes.c2p(t, v) for t, v in zip(times, high)] + [axes.c2p(t, v) for t, v in zip(times[::-1], low[::-1])]
    return Polygon(*pts, stroke_width=0, fill_color=color, fill_opacity=opacity)


def cue_band(axes, start: float, stop: float, y_low: float, y_high: float, opacity: float = 0.12) -> Rectangle:
    """The faint gold rectangle that marks when the target was on screen."""
    p0, p1 = axes.c2p(start, y_low), axes.c2p(stop, y_high)
    return Rectangle(width=p1[0] - p0[0], height=p1[1] - p0[1], stroke_width=0,
                     fill_color=CUE_COLOR, fill_opacity=opacity).move_to((p0 + p1) / 2)
