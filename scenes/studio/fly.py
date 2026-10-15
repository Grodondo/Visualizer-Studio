"""Loader and drawing pieces for "Inside a real brain's wiring" (FlyWire connectome).

Data: assets/connectome/connectome_trace.json (compute/connectome.py) and the images + manifest.json in media/fly_frames/
(compute/fly_render.py). Scene placement: the whole-brain image is IMAGE_WIDTH scene units wide, centered at IMAGE_CENTER;
a point at screen micrometres (u, v) is drawn at IMAGE_CENTER + SCALE * (u, v). The camera math repeats
compute.fly_render.camera (tests/test_connectome.py checks that both agree).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from manim import ImageMobject, ValueTracker
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
TRACE_PATH = ROOT / "assets" / "connectome" / "connectome_trace.json"
FRAMES_DIR = ROOT / "media" / "fly_frames"

IMAGE_WIDTH = 12.0  # scene units across the whole-brain image (the brain itself is about 11 units wide)
IMAGE_CENTER = np.array([0.0, 0.45, 0.0])


@dataclass(frozen=True)
class FlyTrace:
    raw: dict

    @classmethod
    def load(cls, path: Path = TRACE_PATH) -> "FlyTrace":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def __getitem__(self, key):
        return self.raw[key]


def camera(angle_deg: float) -> np.ndarray:
    """Same as compute.fly_render.camera: rows = screen-right, screen-up, toward-viewer in data axes."""
    a = np.radians(angle_deg)
    right = np.array([-np.cos(a), 0.0, np.sin(a)])
    up = np.array([0.0, -1.0, 0.0])
    return np.vstack([right, up, np.cross(right, up)])


class FlyFrames:
    def __init__(self, folder: Path = FRAMES_DIR):
        manifest = folder / "manifest.json"
        if not manifest.exists():
            raise FileNotFoundError(f"{manifest} is missing: run  uv run python -m compute.fly_render  first")
        self.folder = folder
        self.m = json.loads(manifest.read_text(encoding="utf-8"))
        self.center = np.array(self.m["center_um"])
        self.scale = IMAGE_WIDTH / (2 * self.m["half_width_um"])  # scene units per micrometre in the whole-brain view
        self._cache: dict[Path, np.ndarray] = {}

    # ---- placing data on screen ----
    def screen_um(self, xyz, angle: float = 0.0) -> np.ndarray:
        return (np.atleast_2d(np.asarray(xyz, float)) - self.center) @ camera(angle).T

    def to_scene(self, xyz) -> np.ndarray:
        """Scene points (n x 3) of data positions in the front view."""
        s = self.screen_um(xyz)
        out = np.zeros((len(s), 3))
        out[:, :2] = IMAGE_CENTER[:2] + self.scale * s[:, :2]
        return out

    def um_to_scene(self, u: float, v: float) -> np.ndarray:
        return IMAGE_CENTER + self.scale * np.array([u, v, 0.0])

    # ---- images ----
    def image(self, relative: str, width: float = IMAGE_WIDTH, center=IMAGE_CENTER) -> ImageMobject:
        return ImageMobject(str(self.folder / relative)).set_width(width).move_to(center)

    @property
    def zoom_window(self) -> tuple[float, float, float]:
        return tuple(self.m["zoom"]["window_um"])

    def zoom_factor(self) -> float:
        """How much the whole-brain view must be magnified for the zoom window to fill IMAGE_WIDTH."""
        return self.m["half_width_um"] / self.zoom_window[2]

    def pixels(self, path: Path) -> np.ndarray:
        if path not in self._cache:
            self._cache[path] = np.array(Image.open(path).convert("RGBA"))
        return self._cache[path]

    def sequence(self, pattern: str, angles: list[int], tracker: ValueTracker, width: float = IMAGE_WIDTH) -> ImageMobject:
        """An image that shows the frame whose angle is nearest to tracker's value (degrees, wraps at 360)."""
        angles = np.array(angles)
        paths = [self.folder / pattern.format(angle=a) for a in angles]

        def nearest() -> int:
            v = tracker.get_value() % 360
            d = np.abs((angles - v + 180) % 360 - 180)
            return int(np.argmin(d))

        img = ImageMobject(str(paths[nearest()])).set_width(width).move_to(IMAGE_CENTER)
        state = {"i": None}

        def update(mob):
            i = nearest()
            if i != state["i"]:
                mob.pixel_array = self.pixels(paths[i])
                state["i"] = i

        img.add_updater(update)
        return img
