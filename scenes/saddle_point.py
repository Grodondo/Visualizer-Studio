"""Why is a saddle point not a minimum?  A 3D explanatory video (about 35 seconds).

    uv run python -m compute.saddle                                  # (re)write assets/saddle/saddle_trace.json
    uv run manim -ql scenes/saddle_point.py SaddlePoint              # draft, 480p15 (3D is slow: expect a minute or two)
    uv run manim -ql -s scenes/saddle_point.py SaddlePoint           # last frame only
    uv run python scenes/studio/inspect_video.py media/videos/saddle_point/480p15/SaddlePoint.mp4

Render one beat while working on it (PowerShell):
    $env:BEATS = "along-x"; uv run manim -ql scenes/saddle_point.py SaddlePoint; Remove-Item Env:BEATS

Story: at the center of f(x, y) = x^2 - y^2 the ground is flat (gradient 0), but it curves UP along x and DOWN along y.
A minimum needs every direction to go up, so this flat point is a saddle point.

All numbers (gradient, curvatures, curves) come from assets/saddle/saddle_trace.json, written by compute/saddle.py.
Colors keep their meaning from the rest of the studio: blue = positive number, orange = negative, gold = look here.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from manim import *

from studio.beats import BeatScene3D
from studio.equations import typst_number
from studio.theme import BG, FONT, GOLD, MUTED, NEG, POS, TEXT, sign_color
from studio.trace import format_value

TRACE_PATH = Path(__file__).resolve().parent.parent / "assets" / "saddle" / "saddle_trace.json"

# ----------------------------------------------------------------------------------------------
# KNOBS. Camera angles are in degrees: phi tilts the camera (0 = straight down, 90 = level with the surface),
# theta turns it around the vertical axis (-90 looks along the y axis, 0 looks along the x axis).
# ----------------------------------------------------------------------------------------------
VIEW_START = (62, -55)  # phi, theta of the first view
VIEW_ALONG_X = (80, -90)  # a side view in which the x axis runs left to right
VIEW_ALONG_Y = (80, 0)  # a side view in which the y axis runs left to right
VIEW_FINAL = (62, -40)
ROTATE_RATE = 0.2  # radians per second while the camera turns around the surface in beat 1
SURFACE_RESOLUTION = (24, 24)  # more = smoother and slower (use about 40 for a final render)
SURFACE_OPACITY = 0.85
SURFACE_SEE_THROUGH = 0.3  # the surface becomes this transparent while we look at one slice
AXES_LENGTH = (6, 6, 4)  # drawn size of the x, y and z axes
CAPTION_BUFF = 0.8  # distance of the caption from the bottom edge


@dataclass(frozen=True)
class SaddleTrace:
    function_typst: str
    hessian: tuple
    extent: float
    gradient: tuple
    curvature_x: float
    curvature_y: float
    classification: str
    x_slice: tuple  # (t values, heights)
    y_slice: tuple

    @classmethod
    def load(cls, path) -> "SaddleTrace":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            function_typst=d["function_typst"],
            hessian=tuple(tuple(row) for row in d["hessian"]),
            extent=d["extent"],
            gradient=tuple(d["gradient_at_critical_point"]),
            curvature_x=d["curvature"]["x"],
            curvature_y=d["curvature"]["y"],
            classification=d["classification"],
            x_slice=(tuple(d["slices"]["x"]["t"]), tuple(d["slices"]["x"]["z"])),
            y_slice=(tuple(d["slices"]["y"]["t"]), tuple(d["slices"]["y"]["z"])),
        )

    def height(self, x, y):
        """The surface, rebuilt from the Hessian (tests check it against compute.saddle.height)."""
        (h11, h12), (_, h22) = self.hessian
        return 0.5 * (h11 * x * x + 2 * h12 * x * y + h22 * y * y)

    @property
    def z_limit(self) -> float:
        return max(abs(z) for z in (*self.x_slice[1], *self.y_slice[1]))


def curvature_label(axis: str, value: float, font_size: float = 38) -> MathTypst:
    """f_(x x) = 2.00 with the number in its sign color (blue curves up, orange curves down)."""
    label = MathTypst(f"f_({axis} {axis}) = " + "{{ " + typst_number(value) + " : v }}", font_size=font_size, color=TEXT)
    label.select("v").set_color(sign_color(value))
    return label


class SaddlePoint(BeatScene3D):
    def construct(self):
        self.trace = SaddleTrace.load(TRACE_PATH)
        assert self.trace.classification == "saddle", "this video explains a saddle point; pick a Hessian with mixed signs"
        t = self.trace
        self.caption_now = None

        self.axes = ThreeDAxes(
            x_range=[-t.extent - 0.5, t.extent + 0.5, 1],
            y_range=[-t.extent - 0.5, t.extent + 0.5, 1],
            z_range=[-t.z_limit, t.z_limit, 2],
            x_length=AXES_LENGTH[0],
            y_length=AXES_LENGTH[1],
            z_length=AXES_LENGTH[2],
            axis_config={"color": MUTED, "include_numbers": False, "include_tip": False},
        )
        self.surface = Surface(
            lambda u, v: self.axes.c2p(u, v, t.height(u, v)),
            u_range=[-t.extent, t.extent],
            v_range=[-t.extent, t.extent],
            resolution=SURFACE_RESOLUTION,
            fill_opacity=SURFACE_OPACITY,
            stroke_width=0.5,
            stroke_color=BG,
        )
        # Height decides the color: blue above the flat level, orange below, gray at zero.
        self.surface.set_fill_by_value(axes=self.axes, colorscale=[(NEG, -t.z_limit), (MUTED, 0), (POS, t.z_limit)], axis=2)

        self.set_camera_orientation(phi=VIEW_START[0] * DEGREES, theta=VIEW_START[1] * DEGREES)
        self.beat_surface()
        self.beat_flat_spot()
        self.beat_along("along-x", "x", VIEW_ALONG_X, t.curvature_x, "Walk along x: the ground rises on both sides.", slot=0)
        self.beat_along("along-y", "y", VIEW_ALONG_Y, t.curvature_y, "Walk along y: the ground falls on both sides.", slot=1)
        self.beat_conclusion()

    # ---- helpers -------------------------------------------------------------------------------

    def fixed(self, *mobjects):
        """Pin text to the screen so it does not rotate with the 3D scene, and return it unshown."""
        self.add_fixed_in_frame_mobjects(*mobjects)
        self.remove(*mobjects)
        return mobjects[0] if len(mobjects) == 1 else mobjects

    def say(self, new, run_time: float = 0.6):
        """Swap the caption at the bottom of the screen."""
        new.to_edge(DOWN, buff=CAPTION_BUFF)
        self.fixed(new)
        old, self.caption_now = self.caption_now, new
        self.play(*([FadeOut(old)] if old else []), FadeIn(new), run_time=run_time)

    def caption(self, text: str) -> Text:
        return Text(text, font=FONT, font_size=30, color=TEXT)

    def slice_curve(self, axis: str, colour) -> VMobject:
        """The surface cut along one axis, drawn as a 3D curve from the exported sample points."""
        ts, zs = self.trace.x_slice if axis == "x" else self.trace.y_slice
        points = [self.axes.c2p(a, 0, z) if axis == "x" else self.axes.c2p(0, a, z) for a, z in zip(ts, zs)]
        return VMobject(color=colour, stroke_width=7).set_points_smoothly(points)

    # ---- beat 1: a 3D surface, turning --------------------------------------------------------

    def beat_surface(self):
        self.beat("surface")
        t = self.trace
        title = self.fixed(Text("Why is a saddle point not a minimum?", font=FONT, font_size=38, color=TEXT).to_edge(UP, buff=0.5))
        self.equation = self.fixed(MathTypst(f"f(x, y) = {t.function_typst}", font_size=38, color=TEXT).to_corner(UL, buff=0.5))
        footer = self.fixed(Text("A mathematical example: the surface is exact, nothing is simulated or measured.", font=FONT, font_size=18, color=MUTED).to_edge(DOWN, buff=0.25))
        labels = [Text(name, font=FONT, font_size=26, color=MUTED).move_to(self.axes.c2p(*pos)) for name, pos in (("x", (t.extent + 0.9, 0, 0)), ("y", (0, t.extent + 0.9, 0)), ("z", (0, 0, t.z_limit + 0.6)))]
        self.add_fixed_orientation_mobjects(*labels)
        self.remove(*labels)

        self.play(Write(title), run_time=1.2)
        self.play(FadeIn(self.axes), FadeIn(self.surface), FadeIn(*labels), FadeIn(footer), run_time=2.0)
        self.begin_ambient_camera_rotation(rate=ROTATE_RATE)
        self.wait(3.0)
        # The title and the equation share the top of the screen, so one leaves as the other arrives.
        self.play(FadeOut(title), FadeIn(self.equation), run_time=0.8)
        self.wait(0.7)
        self.stop_ambient_camera_rotation()

    # ---- beat 2: the ground is flat at the center ----------------------------------------------

    def beat_flat_spot(self):
        self.beat("flat-spot")
        t = self.trace
        self.move_camera(phi=VIEW_FINAL[0] * DEGREES, theta=VIEW_FINAL[1] * DEGREES, run_time=1.4)
        self.center_dot = Dot3D(self.axes.c2p(0, 0, 0), radius=0.14, color=GOLD)
        # A flat sheet at the height of the critical point: the "ground" a minimum would have to sit under.
        self.flat_sheet = Surface(lambda u, v: self.axes.c2p(u, v, 0), u_range=[-1.2, 1.2], v_range=[-1.2, 1.2], resolution=(2, 2), fill_color=GOLD, fill_opacity=0.45, stroke_width=0)
        self.play(FadeIn(self.center_dot), FadeIn(self.flat_sheet), run_time=1.0)
        gx, gy = (format_value(v) for v in t.gradient)
        line = VGroup(self.caption("At the center the ground is flat:"), MathTypst(f"nabla f = ({gx}, {gy})", font_size=34, color=TEXT)).arrange(RIGHT, buff=0.35)
        self.say(line)
        self.wait(2.0)

    # ---- beats 3 and 4: look along one direction ---------------------------------------------

    def beat_along(self, name, axis, view, curvature, sentence, slot):
        self.beat(name)
        t = self.trace
        curve = self.slice_curve(axis, sign_color(curvature))
        value_label = curvature_label(axis, curvature)
        value_label.to_corner(UR, buff=0.6).shift(DOWN * 0.75 * slot)  # the second number sits under the first
        self.fixed(value_label)
        self.say(self.caption(sentence))
        fade_sheet = [FadeOut(self.flat_sheet)] if slot == 0 else []  # the gold sheet has done its job after the first slice
        self.move_camera(phi=view[0] * DEGREES, theta=view[1] * DEGREES, run_time=1.6, added_anims=[self.surface.animate.set_opacity(SURFACE_SEE_THROUGH), *fade_sheet])
        self.play(Create(curve), run_time=1.0)

        position = ValueTracker(0.0)
        point = (lambda p: self.axes.c2p(p, 0, t.height(p, 0))) if axis == "x" else (lambda p: self.axes.c2p(0, p, t.height(0, p)))
        walker = always_redraw(lambda: Dot3D(point(position.get_value()), radius=0.16, color=GOLD))
        self.add(walker)
        for target, seconds in ((t.extent, 1.0), (0.0, 0.8), (-t.extent, 1.0), (0.0, 0.6)):
            self.play(position.animate.set_value(target), run_time=seconds, rate_func=smooth)
        self.play(FadeIn(value_label), run_time=0.6)
        self.wait(0.4)
        self.remove(walker)

    # ---- beat 5: put it together -------------------------------------------------------------

    def beat_conclusion(self):
        self.beat("conclusion")
        self.move_camera(phi=VIEW_FINAL[0] * DEGREES, theta=VIEW_FINAL[1] * DEGREES, run_time=1.6, added_anims=[self.surface.animate.set_opacity(SURFACE_OPACITY)])
        self.begin_ambient_camera_rotation(rate=ROTATE_RATE * 0.4)  # a gentle turn: a fast one ends in a side view
        self.say(self.caption("Up one way, down the other: this flat point is a saddle point."))
        self.wait(2.2)
        self.say(self.caption("A minimum needs every direction to go up."))
        self.wait(2.2)
        self.stop_ambient_camera_rotation()
