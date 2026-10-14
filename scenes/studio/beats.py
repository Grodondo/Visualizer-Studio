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

from manim import MovingCameraScene

from .theme import BG


class BeatScene(MovingCameraScene):
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
        video = self._get_manager().file_writer.movie_file_path
        sidecar = video.with_name(video.stem + ".beats.json")
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        beats = [{k: v for k, v in b.items() if k != "scene_start"} for b in self._beats]
        sidecar.write_text(json.dumps({"beats": beats}, indent=1), encoding="utf-8")
