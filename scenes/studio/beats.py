"""BeatScene: a Scene that is written as named beats, so one beat can be rendered and inspected alone.

    class MyVideo(BeatScene):
        def construct(self):
            self.beat("build-network")
            ...
            self.beat("zoom-neuron")
            ...

Render everything:        uv run manim -ql scenes/my_video.py MyVideo
Render only some beats:   $env:BEATS = "zoom-neuron,recap"   (PowerShell; other beats are fast-forwarded)

Every render also writes <video>.beats.json next to the mp4: beat names and their start/end in
*video* seconds (skipped beats are marked and take no time), which scenes/studio/inspect_video.py
uses to pull frames from the right place.
"""
from __future__ import annotations

import json
import os

from manim import DOWN, FadeIn, FadeOut, MovingCameraScene, Mobject, Restore, Text, ThreeDScene

from .theme import BG, FONT, TEXT


class BeatMixin:
    """The beat bookkeeping, shared by the 2D and 3D scene classes below."""

    def setup(self) -> None:
        super().setup()
        self.camera.background_color = BG
        self._beats: list[dict] = []
        self._skipped_time = 0.0  # scene time spent in fast-forwarded beats; they are absent from the video
        selected = os.environ.get("BEATS", "").strip()
        self._only = {b.strip() for b in selected.split(",") if b.strip()} or None

    def beat(self, name: str, caption: str | None = None) -> None:
        """Start a new beat. Objects stay on screen, so a skipped beat still leaves the right state behind."""
        self._close_current_beat()
        skip = self._only is not None and name not in self._only
        self._beats.append({"name": name, "caption": caption, "skipped": skip, "scene_start": self.time})
        self.next_section(name, skip_animations=skip)

    def _close_current_beat(self) -> None:
        if not self._beats or "end" in self._beats[-1]:
            return
        current = self._beats[-1]
        duration = self.time - current["scene_start"]
        if current["skipped"]:
            self._skipped_time += duration
            current.update(start=None, end=None)
        else:
            current["start"] = current["scene_start"] - self._skipped_time
            current["end"] = current["start"] + duration

    def tear_down(self) -> None:
        super().tear_down()
        if not self._beats:
            return
        self._close_current_beat()
        try:
            video = self._get_manager().file_writer.movie_file_path
        except AttributeError:
            return  # a still-image render (-s) has no video file, so there is nothing to describe
        sidecar = video.with_name(video.stem + ".beats.json")
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        beats = [{k: v for k, v in b.items() if k != "scene_start"} for b in self._beats]
        sidecar.write_text(json.dumps({"beats": beats}, indent=1), encoding="utf-8")


class BeatScene(BeatMixin, MovingCameraScene):
    """A 2D video made of beats, with camera zoom helpers and a caption line."""

    caption_y = -2.55  # where the caption sits (scene units; the bottom edge of the screen is -4)
    caption_max_width = 13.0  # longer text is shrunk to fit; the screen is 14.22 wide

    def setup(self) -> None:
        super().setup()
        self._camera_saved = False
        self.caption_now = None

    def swap_caption(self, text, run_time: float = 0.6, font_size: float = 28):
        """Replace the caption: the old one fades out while the new one (a string or any mobject) fades in."""
        if isinstance(text, Mobject):
            new = text
        else:
            new = Text(text, font=FONT, font_size=font_size, color=TEXT)
            if new.width > self.caption_max_width:
                new.scale_to_fit_width(self.caption_max_width)
            new.move_to([0, self.caption_y, 0])
        old, self.caption_now = self.caption_now, new
        self.play(*([FadeOut(old)] if old else []), FadeIn(new), run_time=run_time)

    def zoom_to(self, center, width: float):
        """An animation that moves the camera to `center` so that `width` scene units fit across the screen.

        Use it inside play():  self.play(self.zoom_to((-1.4, 0.2), 10), other_animation)
        Smaller width = closer. The full view is 14.22 units wide.
        """
        if not self._camera_saved:
            self.camera.frame.save_state()
            self._camera_saved = True
        x, y = center[0], center[1]
        return self.camera.frame.animate.move_to([x, y, 0]).set(width=width)

    def zoom_reset(self) -> Restore:
        """An animation that returns the camera to the full view it started with."""
        return Restore(self.camera.frame)


class BeatScene3D(BeatMixin, ThreeDScene):
    """A 3D video made of beats. Move the camera with self.move_camera(phi=..., theta=...)."""
