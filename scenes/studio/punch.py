"""Loader and drawing pieces for "What happens in your brain when you react to a punch?".

Data: assets/punch_reaction/punch_trace.json (compute/punch_reaction.py) and the brain images plus manifest.json in
media/brain_frames/punch_reaction/ (compute/brain_render.py). This module never computes; it only draws exported results.
The boxers are a drawing (an illustration), not data.
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
    Circle,
    ImageMobject,
    Line,
    RoundedRectangle,
    VGroup,
    ValueTracker,
)
from PIL import Image

from .theme import TEXT

ROOT = Path(__file__).resolve().parents[2]
TRACE_PATH = ROOT / "assets" / "punch_reaction" / "punch_trace.json"
FRAMES_DIR = ROOT / "media" / "brain_frames" / "punch_reaction"


@dataclass(frozen=True)
class PunchTrace:
    raw: dict

    @classmethod
    def load(cls, path: Path = TRACE_PATH) -> "PunchTrace":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def __getitem__(self, key):
        return self.raw[key]

    @property
    def rts(self) -> np.ndarray:
        return np.array(self.raw["rt"]["values_s"])

    def visual(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array(self.raw["visual"]["times"]), np.array(self.raw["visual"]["uv"])

    def lrp(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array(self.raw["lrp"]["times"]), np.array(self.raw["lrp"]["uv"])

    def top_labels(self, t: float) -> list[str]:
        """Atlas labels holding most of the strongest estimated sources at time t (nearest exported 20 ms step)."""
        steps = self.raw["where"]["stim"]
        step = min(steps, key=lambda s: abs(s["t"] - t))
        return [name for name, _ in step["top_labels"]]


class BrainFrames:
    """The rendered brain images: anatomy stages, activity frames over time, and the limitation frame."""

    def __init__(self, folder: Path = FRAMES_DIR):
        manifest = folder / "manifest.json"
        if not manifest.exists():
            raise FileNotFoundError(f"{manifest} is missing: run  uv run python -m compute.brain_render  first")
        self.folder = folder
        self.manifest = json.loads(manifest.read_text(encoding="utf-8"))
        self.times = np.array(self.manifest["activity"]["times"])
        self._cache: dict[Path, np.ndarray] = {}

    @property
    def stages(self) -> list[dict]:
        return self.manifest["stages"]

    def anatomy(self, view: str, stage: int) -> ImageMobject:
        return ImageMobject(str(self.folder / "anatomy" / f"{view}_{stage}.png"))

    def limit(self, name: str) -> ImageMobject:
        return ImageMobject(str(self.folder / "limit" / f"{name}.png"))

    def activity_path(self, view: str, index: int) -> Path:
        return self.folder / "activity" / f"{view}_{index:03d}.png"

    def index_at(self, t: float) -> int:
        return int(np.argmin(np.abs(self.times - t)))

    def pixels(self, path: Path) -> np.ndarray:
        if path not in self._cache:
            self._cache[path] = np.array(Image.open(path).convert("RGBA"))
        return self._cache[path]

    def player(self, view: str, tracker: ValueTracker, height: float) -> ImageMobject:
        """An image that always shows the activity frame nearest to tracker's value (seconds after the arrows)."""
        img = ImageMobject(str(self.activity_path(view, self.index_at(tracker.get_value())))).set_height(height)
        state = {"index": None}

        def update(mob):
            i = self.index_at(tracker.get_value())
            if i != state["index"]:
                mob.pixel_array = self.pixels(self.activity_path(view, i))
                state["index"] = i

        img.add_updater(update)
        return img


# ---- the illustration ------------------------------------------------------------------------------

class Boxer(VGroup):
    """A geometric boxer seen from the side, facing `facing` (RIGHT or LEFT). Parts: head, body, legs, two arms.

    The lead arm is a 2-segment line ending in a glove; `jab(amount)` returns the glove position for a punch
    extended by amount (0 = guard, 1 = fully extended).
    """

    def __init__(self, color=TEXT, facing=RIGHT, scale: float = 1.0):
        super().__init__()
        f = 1 if facing is RIGHT or (hasattr(facing, "__getitem__") and facing[0] > 0) else -1
        self.f = f
        style = dict(color=color, stroke_width=5)
        self.head = Circle(0.32, **style).move_to([0.05 * f, 1.55, 0])
        self.body = Line([0, 1.18, 0], [-0.1 * f, 0.05, 0], **style)
        self.legs = VGroup(Line([-0.1 * f, 0.05, 0], [0.35 * f, -1.2, 0], **style),
                           Line([-0.1 * f, 0.05, 0], [-0.55 * f, -1.2, 0], **style))
        self.shoulder = np.array([0.0, 1.0, 0])
        self.rear_glove = Circle(0.2, color=color, fill_opacity=1, stroke_width=0).move_to([0.35 * f, 1.3, 0])
        self.rear_arm = VGroup(Line(self.shoulder, [0.15 * f, 0.75, 0], **style), Line([0.15 * f, 0.75, 0], self.rear_glove.get_center(), **style))
        self.lead_glove = Circle(0.2, color=color, fill_opacity=1, stroke_width=0)
        self.lead_arm = VGroup(Line(self.shoulder, self.shoulder, **style), Line(self.shoulder, self.shoulder, **style))
        self.add(self.legs, self.body, self.head, self.rear_arm, self.rear_glove, self.lead_arm, self.lead_glove)
        self.set_lead(0.0)
        self.scale(scale)

    def set_lead(self, amount: float) -> "Boxer":
        """Place the lead arm: 0 = guard (glove near the chin), 1 = straight jab."""
        s = self.shoulder_point()
        u = self.unit()
        guard_elbow, guard_glove = s + np.array([0.3 * self.f, -0.35, 0]) * u, s + np.array([0.55 * self.f, 0.25, 0]) * u
        jab_elbow, jab_glove = s + np.array([0.75 * self.f, 0.12, 0]) * u, s + np.array([1.5 * self.f, 0.28, 0]) * u
        elbow = guard_elbow + amount * (jab_elbow - guard_elbow)
        glove = guard_glove + amount * (jab_glove - guard_glove)
        self.lead_arm[0].put_start_and_end_on(s, elbow)
        self.lead_arm[1].put_start_and_end_on(elbow, glove)
        self.lead_glove.move_to(glove)
        return self

    def unit(self) -> float:
        """Current drawing scale (the head radius is 0.32 at scale 1)."""
        return self.head.width / 0.64

    def shoulder_point(self) -> np.ndarray:
        return self.body.get_start() + (self.body.get_end() - self.body.get_start()) * 0.15
