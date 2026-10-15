"""Can we see the brain preparing to move?  Real EEG during real and imagined fist clenching (about 2 minutes).

    uv run python -m compute.eeg_motor                               # download 20 volunteers once (~300 MB), analyse, write the trace
    uv run manim -ql scenes/eeg_motor.py EegMotor                    # draft, 480p15
    uv run manim -qm scenes/eeg_motor.py EegMotor                    # draft, 720p30
    uv run python scenes/studio/inspect_video.py media/videos/eeg_motor/480p15/EegMotor.mp4 --per-beat 3

Render one beat while working on it (PowerShell):
    $env:BEATS = "average"; uv run manim -ql scenes/eeg_motor.py EegMotor; Remove-Item Env:BEATS

Beats: question, task, raw, rhythm, average, map, limits, recap.
MEASURED: every trace, curve, map and number comes from assets/eeg_motor/eeg_motor_trace.json (PhysioNet EEG Motor
Movement/Imagery v1.0.0, Schalk et al. 2004, ODC-By 1.0), written by compute/eeg_motor.py.
SCHEMATIC (badged on screen): the drawing of the task, and the patch of neurons under one electrode.
Colors: orange = a rhythm got weaker, blue = stronger, white = real movement, teal = imagined, gold band = target on screen.
"""
from pathlib import Path

import numpy as np
from manim import *

from studio.beats import BeatScene
from studio.eeg import (
    CONDITION_COLOR,
    CUE_COLOR,
    EXECUTED,
    IMAGINED,
    EegTrace,
    HeadMap,
    band_polygon,
    color_bar,
    cue_band,
    polyline,
    power_color,
)
from studio.theme import BG, FONT, GOLD, MUTED, NEG, POS, TEXT

TRACE_PATH = Path(__file__).resolve().parent.parent / "assets" / "eeg_motor" / "eeg_motor_trace.json"

# ----------------------------------------------------------------------------------------------
# KNOBS. Positions are scene units: the screen is 14.22 wide and 8 high, (0, 0) is the center, y grows upward.
# ----------------------------------------------------------------------------------------------
RAW_WINDOW = (2.0, 17.0)  # seconds of the example recording shown in beat "raw" (rest, right fist, rest, left fist)
RAW_X = (-6.0, 6.0)  # where that window is drawn across the screen
RAW_Y = {"C3": 1.05, "C4": -0.75}  # baseline height of each trace
UV_SCALE = 0.011  # scene units per microvolt for the raw traces
MU_UV_SCALE = 0.05  # scene units per microvolt for the mu-band trace (it is much smaller than the raw signal)
ZOOM_SECONDS = (3.0, 6.5)  # beat "rhythm" zooms onto this stretch: the end of rest and the start of the right-fist cue
PLOT_SIZE = (7.6, 3.9)  # width, height of the averaged-power plot
PLOT_CENTER = (-1.9, 0.3)
Y_RANGE = (-40, 30)  # % power change shown on the plot
HEAD_RADIUS = 1.55
MAP_X = (-3.1, 3.1)  # centers of the two scalp maps (moved, imagined)
MAP_Y = 0.35
TASK_SCREEN = (-1.6, 1.0, 0)  # beat "task": the screen with the target, and the drawn fist
TASK_FIST = (2.6, 1.05, 0)
BAND = "mu"  # the rhythm the story follows (8-13 Hz); "beta" also exists in the trace
FOOTER_Y = -3.68


def label(text: str, size: float = 24, color=TEXT, **kw) -> Text:
    return Text(text, font=FONT, font_size=size, color=color, **kw)


def badge(kind: str) -> VGroup:
    """A small corner tag that says whether what is on screen was measured or is a drawing."""
    color = POS if kind == "MEASURED" else GOLD
    t = label(kind, 18, color, weight=BOLD)
    box = SurroundingRectangle(t, buff=0.09, corner_radius=0.06, color=color, stroke_width=1.5)
    return VGroup(box, t).to_corner(UR, buff=0.3)


def pct(v: float) -> str:
    return f"{v:+.0f}%".replace("-", "−")


class EegMotor(BeatScene):
    caption_y = -2.95

    def construct(self):
        self.trace = EegTrace.load(TRACE_PATH)
        self.tag = None
        self.beat_question()
        self.beat_task()
        self.beat_raw()
        self.beat_rhythm()
        self.beat_average()
        self.beat_map()
        self.beat_limits()
        self.beat_recap()

    # ---- helpers -------------------------------------------------------------------------------

    def set_tag(self, kind: str | None):
        """Swap the MEASURED / SCHEMATIC corner tag."""
        new = badge(kind) if kind else None
        anims = ([FadeOut(self.tag)] if self.tag else []) + ([FadeIn(new)] if new else [])
        self.tag = new
        if anims:
            self.play(*anims, run_time=0.4)

    def footer(self) -> Text:
        d = self.trace.raw["dataset"]
        return label(f"Measured: {d['name']} · Schalk et al. 2004 · {d['license']} · "
                     f"{self.trace.n_subjects} volunteers", 15, MUTED).move_to([0, FOOTER_Y, 0])

    def clear_stage(self, *keep, run_time: float = 0.6):
        keep = set(keep) | {self.tag, self.foot, self.caption_now}
        gone = [m for m in self.mobjects if m not in keep]
        if gone:
            self.play(*[FadeOut(m) for m in gone], run_time=run_time)

    # ---- beat 1: the question and the electrodes ------------------------------------------------

    def beat_question(self):
        self.beat("question")
        title = label("Can we see the brain preparing to move?", 40).to_edge(UP, buff=0.45)
        self.foot = self.footer()
        head = HeadMap(self.trace.electrodes, radius=HEAD_RADIUS).move_to([0, 0.05, 0])
        self.play(Write(title), FadeIn(self.foot), run_time=1.2)
        self.set_tag("MEASURED")
        self.play(Create(head.head), run_time=0.8)
        self.play(LaggedStart(*[GrowFromCenter(d) for d in head.dots], lag_ratio=0.02), run_time=1.6)
        self.swap_caption("64 electrodes on the scalp, at their standard positions. Seen from above, nose up.")
        self.wait(1.2)
        c3, c4 = head.dot_of["C3"], head.dot_of["C4"]
        names = VGroup(head.label("C3", UP), head.label("C4", UP))
        self.play(c3.animate.set_color(GOLD).scale(1.6), c4.animate.set_color(GOLD).scale(1.6), FadeIn(names), run_time=0.8)
        self.swap_caption("C3 and C4 sit over the motor cortex. Left hemisphere (C3) drives the right hand, and vice versa.")
        self.wait(2.0)
        self.head, self.title = VGroup(head, names), title

    # ---- beat 2: what the volunteers did --------------------------------------------------------

    def beat_task(self):
        self.beat("task")
        self.play(self.head.animate.scale(0.55).to_corner(UL, buff=0.4).shift(DOWN * 0.5), FadeOut(self.title), run_time=0.8)
        self.set_tag("SCHEMATIC")
        screen = RoundedRectangle(width=3.2, height=2.0, corner_radius=0.1, color=MUTED, stroke_width=3).move_to(TASK_SCREEN)
        target = Square(0.42, color=GOLD, fill_opacity=0.9, stroke_width=0).move_to(screen.get_right() + LEFT * 0.45)
        screen_lbl = label("target appears on the right", 20, MUTED).next_to(screen, DOWN, buff=0.15)
        fist = self.fist().move_to(TASK_FIST)
        fist_lbl = label("open and close the right fist", 20, MUTED).next_to(fist, DOWN, buff=0.15)
        self.play(Create(screen), run_time=0.6)
        self.play(FadeIn(target, scale=0.5), FadeIn(screen_lbl), run_time=0.6)
        self.play(FadeIn(fist), FadeIn(fist_lbl), run_time=0.5)
        for _ in range(2):
            self.play(fist.animate.scale(0.82), run_time=0.3, rate_func=there_and_back)
        self.swap_caption("A target appears on the left or right of a screen. The volunteer opens and closes that fist until it disappears.")
        self.wait(1.4)
        # the timeline of one trial, from the export's cue durations
        cues = self.trace.example["cues"]
        rest, task = cues[0]["duration"], cues[1]["duration"]
        bar = VGroup(
            Rectangle(width=rest * 0.75, height=0.4, stroke_color=MUTED, stroke_width=2),
            Rectangle(width=task * 0.75, height=0.4, stroke_width=0, fill_color=CUE_COLOR, fill_opacity=0.35),
        ).arrange(RIGHT, buff=0).move_to([0.3, -1.35, 0])
        bar_lbls = VGroup(
            label(f"rest  {rest:.1f} s", 20, MUTED).move_to(bar[0]),
            label(f"target on screen  {task:.1f} s", 20, TEXT).move_to(bar[1]),
        )
        self.play(FadeIn(bar), FadeIn(bar_lbls), run_time=0.7)
        imagine = label("Other runs: the same, but only IMAGINE the movement. The hand stays still.", 24, IMAGINED).move_to([0.3, -2.1, 0])
        self.play(FadeIn(imagine), run_time=0.6)
        self.swap_caption("The recording marks when the target appeared, not when the hand moved: there was no movement sensor.")
        self.wait(2.6)
        self.play(FadeOut(VGroup(screen, target, screen_lbl, fist, fist_lbl, bar, bar_lbls, imagine)), run_time=0.6)

    def fist(self) -> VGroup:
        """A simple drawn fist: four finger bumps over a palm."""
        """A simple drawn fist seen from the front: four curled fingers above a palm, the thumb across them."""
        style = dict(color=TEXT, stroke_width=3, fill_color=BG, fill_opacity=1)
        palm = RoundedRectangle(width=1.25, height=0.75, corner_radius=0.2, **style)
        fingers = VGroup(*[RoundedRectangle(width=0.3, height=0.55, corner_radius=0.13, **style) for _ in range(4)]).arrange(RIGHT, buff=0.02)
        fingers.next_to(palm, UP, buff=-0.2)
        thumb = RoundedRectangle(width=0.75, height=0.26, corner_radius=0.12, **style).move_to(fingers.get_bottom() + LEFT * 0.18 + DOWN * 0.08)
        return VGroup(palm, fingers, thumb)

    # ---- beat 3: the raw signal ---------------------------------------------------------------------

    def raw_axes_map(self, t):
        t0, t1 = RAW_WINDOW
        return RAW_X[0] + (np.asarray(t) - t0) / (t1 - t0) * (RAW_X[1] - RAW_X[0])

    def example_line(self, key: str, channel: str, scale: float, color, y: float, width: float = 1.6) -> VMobject:
        ex = self.trace.example
        sf, t0 = ex["sfreq"], ex["t0"]
        i = ex["channels"].index(channel)
        a, b = int((RAW_WINDOW[0] - t0) * sf), int((RAW_WINDOW[1] - t0) * sf)
        values = np.array(ex[key][i][a:b])
        times = RAW_WINDOW[0] + np.arange(len(values)) / sf
        return polyline(self.raw_axes_map(times), y + values * scale, color, width)

    def beat_raw(self):
        self.beat("raw")
        self.set_tag("MEASURED")
        ex = self.trace.example
        bands, names = VGroup(), VGroup()
        for c in ex["cues"]:
            if c["label"] == "T0":
                continue
            start, stop = max(c["onset"], RAW_WINDOW[0]), min(c["onset"] + c["duration"], RAW_WINDOW[1])
            if stop <= start:
                continue
            x0, x1 = self.raw_axes_map([start, stop])
            bands.add(Rectangle(width=x1 - x0, height=3.6, stroke_width=0, fill_color=CUE_COLOR, fill_opacity=0.12).move_to([(x0 + x1) / 2, 0.15, 0]))
            hand = "right" if c["label"] == "T2" else "left"
            names.add(label(f"{hand} fist", 22, GOLD).move_to([(x0 + x1) / 2, 2.25, 0]))
        lines = {ch: self.example_line("raw_uv", ch, UV_SCALE, TEXT, RAW_Y[ch]) for ch in ("C3", "C4")}
        ch_lbls = VGroup(*[label(ch, 24, GOLD).next_to([RAW_X[0], RAW_Y[ch], 0], LEFT, buff=0.15) for ch in lines])
        # scale bars: 1 second and 50 microvolts
        sb_x = self.raw_axes_map(RAW_WINDOW[1] - 1.0)
        sec = Line([sb_x, -2.05, 0], [RAW_X[1], -2.05, 0], color=MUTED, stroke_width=3)
        uv = Line([RAW_X[1], -2.05, 0], [RAW_X[1], -2.05 + 50 * UV_SCALE, 0], color=MUTED, stroke_width=3)
        sb_lbl = label("1 s,  50 µV", 18, MUTED).next_to(sec, DOWN, buff=0.08)
        title = label(f"Volunteer {ex['subject']}, run {ex['run']}: the raw voltage", 26, MUTED).to_edge(UP, buff=0.35)
        self.play(FadeOut(self.head), FadeIn(title), FadeIn(ch_lbls), FadeIn(bands), FadeIn(names), FadeIn(VGroup(sec, uv, sb_lbl)), run_time=0.8)
        self.swap_caption("Real EEG from one volunteer (the first in the dataset), as the session happened.")
        self.play(*[Create(l) for l in lines.values()], run_time=6.0, rate_func=linear)
        self.swap_caption("Gold bands: the target was on screen. By eye, the moving and resting stretches look alike.")
        self.wait(2.2)
        self.raw_parts = dict(lines=lines, ch_lbls=ch_lbls, bands=bands, names=names, scale=VGroup(sec, uv, sb_lbl), title=title)

    # ---- beat 4: the mu rhythm inside the signal -----------------------------------------------------

    def beat_rhythm(self):
        self.beat("rhythm")
        p = self.raw_parts
        y = RAW_Y["C3"]
        mu = self.example_line("mu_uv", "C3", MU_UV_SCALE, POS, y, 2.0)
        env = self.example_line("mu_envelope_uv", "C3", MU_UV_SCALE, GOLD, y, 3.0)
        mu_lbl = label("blue: the 8–13 Hz part only (the “mu” rhythm)", 22, POS).move_to([-2.4, -1.35, 0])
        self.play(FadeOut(p["lines"]["C4"]), FadeOut(p["ch_lbls"][1]), FadeOut(p["scale"]), run_time=0.5)
        self.play(ReplacementTransform(p["lines"]["C3"], mu), FadeIn(mu_lbl), run_time=1.4)
        self.swap_caption("Keep only the 8–13 Hz part of C3. Motor cortex at rest hums at this rhythm, called mu.")
        self.wait(1.0)
        x0, x1 = self.raw_axes_map(ZOOM_SECONDS)
        self.play(self.zoom_to(((x0 + x1) / 2, y), (x1 - x0) * 1.15), run_time=1.6)
        self.wait(2.0)
        self.play(self.zoom_reset(), run_time=1.2)
        env_lbl = label("gold: its size (envelope)", 22, GOLD).next_to(mu_lbl, RIGHT, buff=0.6)
        self.play(mu.animate.set_stroke(opacity=0.45), run_time=0.3)
        self.play(Create(env), FadeIn(env_lbl), run_time=2.5, rate_func=linear)
        self.swap_caption("Gold: how large the rhythm is. In one trial it dips during the right-fist cue, barely during the left one.")
        self.wait(2.4)
        self.swap_caption("A single trial is too noisy to decide. So we average many trials, and many volunteers.")
        self.wait(1.8)
        self.clear_stage()

    # ---- beat 5: the averaged power change -------------------------------------------------------------

    def power_axes(self) -> VGroup:
        t = self.trace.times
        axes = Axes(x_range=[t[0], t[-1], 1], y_range=[Y_RANGE[0], Y_RANGE[1], 10], x_length=PLOT_SIZE[0], y_length=PLOT_SIZE[1],
                    axis_config={"color": MUTED, "include_tip": False, "include_numbers": False}).move_to([PLOT_CENTER[0], PLOT_CENTER[1], 0])
        xt = VGroup(*[label(f"{v:g} s" if v else "0 s", 18, MUTED).next_to(axes.c2p(v, Y_RANGE[0]), DOWN, buff=0.12) for v in range(-1, 5)])
        yt = VGroup(*[label(pct(v) if v else "0", 18, MUTED).next_to(axes.c2p(t[0], v), LEFT, buff=0.12) for v in (-40, -20, 0, 20)])
        zero = DashedLine(axes.c2p(t[0], 0), axes.c2p(t[-1], 0), color=MUTED, stroke_width=1.5, dash_length=0.08)
        # the x axis line sits at y=0 by default; move it to the bottom so the 0% line is a separate dashed reference
        axes.x_axis.shift(axes.c2p(0, Y_RANGE[0]) - axes.c2p(0, 0))
        ylab = label("mu power vs. rest", 20, MUTED).rotate(PI / 2).next_to(yt, LEFT, buff=0.15)
        return VGroup(axes, xt, yt, zero, ylab)

    def course_mobs(self, axes, condition: str, side: str = "contra"):
        t = self.trace.times
        mean, sem = self.trace.course(condition, BAND, side)
        color = CONDITION_COLOR[condition]
        line = VMobject(color=color, stroke_width=3.5).set_points_as_corners([axes.c2p(a, b) for a, b in zip(t, mean)])
        band = band_polygon(axes, t, mean - sem, mean + sem, color, 0.22)
        return line, band

    def beat_average(self):
        self.beat("average")
        tr = self.trace
        plot = self.power_axes()
        axes = plot[0]
        task = tr.example["cues"][1]["duration"]
        cue = cue_band(axes, 0, min(task, tr.times[-1]), Y_RANGE[0], Y_RANGE[1])
        cue_lbl = label("target on screen", 20, GOLD).next_to(axes.c2p(2.0, Y_RANGE[1]), DOWN, buff=0.1)
        zero_line = Line(axes.c2p(0, Y_RANGE[0]), axes.c2p(0, Y_RANGE[1]), color=GOLD, stroke_width=2)
        kept, total = tr.trial_counts()
        title = label(f"Average of {tr.n_subjects} volunteers, opposite side of the head from the hand", 24, MUTED).to_edge(UP, buff=0.35)
        self.play(FadeIn(plot), FadeIn(cue), FadeIn(zero_line), FadeIn(cue_lbl), FadeIn(title), run_time=0.9)
        self.swap_caption(f"Cue-locked averages: {kept:,} of {total:,} trials kept. Zero = the power during the second before the cue.")
        self.wait(1.6)
        legend = VGroup()
        stats = {}
        for i, cond in enumerate(("executed", "imagined")):
            line, band = self.course_mobs(axes, cond)
            m = tr.window_mean(cond, BAND)
            stats[cond] = m
            name = "moved the fist" if cond == "executed" else "only imagined it"
            entry = VGroup(Line(ORIGIN, RIGHT * 0.45, color=CONDITION_COLOR[cond], stroke_width=4),
                           label(f"{name}\n{pct(m)}", 22, CONDITION_COLOR[cond], line_spacing=0.8)).arrange(RIGHT, buff=0.15)
            legend.add(entry)
            legend.arrange(DOWN, aligned_edge=LEFT, buff=0.22).next_to(axes, RIGHT, buff=0.4).shift(UP * 0.6)
            self.play(Create(line), FadeIn(band), run_time=3.0, rate_func=linear)
            self.play(FadeIn(entry), run_time=0.5)
            if cond == "executed":
                self.swap_caption(f"Moving the fist: the mu rhythm weakens by about {abs(m):.0f}% while the target is up.")
            else:
                self.swap_caption(f"Only imagining: it weakens too, by about {abs(m):.0f}%. The motor system engages without any movement.")
            self.wait(2.2)
        w = tr.raw["processing"]["erd_window_s"]
        note = label(f"numbers: mean change {w[0]:g}–{w[1]:g} s after the cue · shading: ±1 standard error across volunteers",
                     16, MUTED).next_to(plot, DOWN, buff=0.15)
        self.play(FadeIn(note), run_time=0.4)
        # timing: the export's "drop time" (first time after the post-cue blip at which the drop reaches half its size)
        drop = tr.condition("executed")["drop_time"]
        drop_imagined = tr.condition("imagined")["drop_time"]
        d_line = DashedLine(axes.c2p(drop, Y_RANGE[0]), axes.c2p(drop, Y_RANGE[1]), color=TEXT, stroke_width=2)
        d_lbl = label(f"{drop:.2f} s", 20, TEXT).next_to(axes.c2p(drop, Y_RANGE[0]), UP, buff=0.1).shift(RIGHT * 0.3)
        self.play(Create(d_line), FadeIn(d_lbl), run_time=0.6)
        self.swap_caption(f"Half of the drop is reached {drop:.2f} s after the target appears when moving, {drop_imagined:.2f} s when imagining.")
        self.wait(2.2)
        # the blip right after the cue is not a motor signal: say what the data shows about it
        eyes = tr.strongest_channels("executed", "burst", 2)
        blip_pt = axes.c2p(0.25, 22)
        arrow = Arrow(blip_pt + RIGHT * 1.4 + UP * 0.25, blip_pt, color=MUTED, buff=0.05, stroke_width=3, max_tip_length_to_length_ratio=0.2)
        self.play(GrowArrow(arrow), FadeOut(d_line), FadeOut(d_lbl), run_time=0.6)
        self.swap_caption(f"The brief rise just after the cue is strongest at {eyes[0]} and {eyes[1]}, beside the eyes: most likely the eyes jumping to the target.")
        self.wait(2.6)
        self.clear_stage()

    # ---- beat 6: where on the scalp -----------------------------------------------------------------------

    def beat_map(self):
        self.beat("map")
        tr = self.trace
        maps, titles, c_lbls = VGroup(), VGroup(), VGroup()
        for x, cond in zip(MAP_X, ("executed", "imagined")):
            hm = HeadMap(tr.electrodes, radius=HEAD_RADIUS, dot_radius=0.11, values=tr.topo(cond, BAND, "right")).move_to([x, MAP_Y, 0])
            maps.add(hm)
            name = "moved the right fist" if cond == "executed" else "imagined the right fist"
            titles.add(label(name, 26, CONDITION_COLOR[cond]).next_to(hm, UP, buff=0.2))
            for ch in ("C3", "C4"):
                ring = Circle(radius=0.17, color=GOLD, stroke_width=2.5).move_to(hm.dot_of[ch])
                txt = label(ch, 20, GOLD).next_to(ring, DOWN, buff=0.04)
                c_lbls.add(ring, BackgroundRectangle(txt, color=BG, fill_opacity=0.85, buff=0.03), txt)
        bar = color_bar().next_to(maps, DOWN, buff=0.35)
        self.play(*[Create(m.head) for m in maps], FadeIn(titles), run_time=0.8)
        self.play(*[LaggedStart(*[FadeIn(d, scale=0.5) for d in m.dots], lag_ratio=0.015) for m in maps], FadeIn(bar), run_time=1.6)
        self.play(FadeIn(c_lbls), run_time=0.4)
        self.swap_caption(f"Change in mu power at every electrode, {tr.raw['processing']['erd_window_s'][0]:g}–{tr.raw['processing']['erd_window_s'][1]:g} s after a right-fist target. Orange = weaker.")
        self.wait(2.4)
        n = tr.n_subjects
        exec_both = tr.n_contra_stronger("executed", BAND)
        imag_contra = tr.n_contra_stronger("imagined", BAND)
        self.swap_caption("Textbooks predict the drop over the opposite side: C3 for the right hand.")
        self.wait(1.8)
        self.swap_caption(f"Real movement: the drop covers both sides (opposite side larger in only {exec_both} of {n} volunteers).")
        self.wait(2.6)
        self.swap_caption(f"Imagery: opposite side larger in {imag_contra} of {n}. Untrained volunteers often show both sides (McFarland et al. 2000).")
        self.wait(2.8)
        self.clear_stage()

    # ---- beat 7: what EEG can and cannot reveal ---------------------------------------------------------

    def beat_limits(self):
        self.beat("limits")
        tr = self.trace
        self.swap_caption("What do these recordings really measure?")
        head = HeadMap(tr.electrodes, radius=1.2, dot_radius=0.06).move_to([-4.6, 0.5, 0])
        c3 = head.dot_of["C3"].copy().set_color(GOLD).scale(1.8)
        self.play(FadeIn(head), FadeIn(c3), run_time=0.6)
        self.set_tag("SCHEMATIC")
        # one electrode -> a patch of cortex with many neurons (a drawing; numbers in the caption come from the literature)
        patch = RoundedRectangle(width=4.2, height=2.6, corner_radius=0.15, color=MUTED, stroke_width=2).move_to([0.6, 0.5, 0])
        rng = np.random.default_rng(3)  # fixed seed: the drawing is the same every render
        cells = VGroup(*[Triangle(color=TEXT, stroke_width=1, fill_opacity=0.5).scale(0.07).move_to(
            [0.6 + rng.uniform(-1.95, 1.95), 0.5 + rng.uniform(-1.15, 1.15), 0]) for _ in range(260)])
        links = VGroup(DashedLine(c3.get_center(), patch.get_corner(UL), color=GOLD, stroke_width=1.5),
                       DashedLine(c3.get_center(), patch.get_corner(DL), color=GOLD, stroke_width=1.5))
        patch_lbl = label("cortex under one electrode (drawing, not to scale)", 18, MUTED).next_to(patch, DOWN, buff=0.12)
        self.play(Create(links), Create(patch), run_time=0.8)
        self.play(LaggedStart(*[FadeIn(c) for c in cells], lag_ratio=0.004), FadeIn(patch_lbl), run_time=1.4)
        self.swap_caption("Each electrode adds up the synchronized currents of millions of neurons spread over centimetres of cortex.")
        self.wait(2.4)
        self.play(LaggedStart(*[c.animate.set_fill(GOLD, 0.9) for c in cells[:130]], lag_ratio=0.003), run_time=1.0)
        self.play(LaggedStart(*[c.animate.set_fill(TEXT, 0.5) for c in cells[:130]], lag_ratio=0.003), run_time=0.8)
        self.swap_caption("It cannot see a single neuron or a single spike. Only large, aligned, synchronized populations reach the scalp.")
        self.wait(2.6)
        self.clear_stage(self.tag)
        self.set_tag(None)
        shows = VGroup(label("What these recordings reveal", 26, POS),
                       label("• motor-cortex rhythms weaken within about half a second of the cue", 21),
                       label("• imagining a movement does it too, with no movement at all", 21),
                       label("• the timing, to a fraction of a second", 21)).arrange(DOWN, aligned_edge=LEFT, buff=0.16)
        cannot = VGroup(label("What they cannot reveal", 26, NEG),
                        label("• which neurons fired, or when a single neuron spiked", 21),
                        label("• exactly where in the cortex: the skull blurs and mixes sources", 21),
                        label("• whether the change came before the hand moved: movement was not recorded", 21)
                        ).arrange(DOWN, aligned_edge=LEFT, buff=0.16)
        cols = VGroup(shows, cannot).arrange(DOWN, aligned_edge=LEFT, buff=0.5).move_to([0, 0.55, 0])
        self.play(FadeIn(shows, shift=UP * 0.1), run_time=0.8)
        self.wait(2.4)
        self.play(FadeIn(cannot, shift=UP * 0.1), run_time=0.8)
        self.swap_caption("With self-paced movements, mu starts to weaken about 2 s before the hand moves (Pfurtscheller & Lopes da Silva 1999; Magnani et al. 1998).")
        self.wait(3.2)
        refs = label("Buzsáki, Anastassiou & Koch 2012, Nat Rev Neurosci · Nunez & Srinivasan 2006, Electric Fields of the Brain\n"
                     "McFarland et al. 2000, Brain Topogr · Pfurtscheller & Lopes da Silva 1999, Clin Neurophysiol · Magnani et al. 1998, Mov Disord", 15, MUTED,
                     line_spacing=0.8).move_to([0, -2.3, 0])
        self.play(FadeIn(refs), run_time=0.5)
        self.wait(1.6)
        self.clear_stage()

    # ---- beat 8: recap -----------------------------------------------------------------------------------

    def beat_recap(self):
        self.beat("recap")
        if self.caption_now:
            self.play(FadeOut(self.caption_now), run_time=0.3)
            self.caption_now = None
        self.set_tag("MEASURED")
        tr = self.trace
        e, i = tr.window_mean("executed", BAND), tr.window_mean("imagined", BAND)
        lines = VGroup(
            label("So: can we see the brain preparing to move?", 32),
            label(f"We see the motor rhythm weaken: {pct(e)} when moving, {pct(i)} when only imagining.", 24, MUTED),
            label("We see it within half a second of the cue, in the average of many trials.", 24, MUTED),
            label("We do not see neurons, and this data cannot say if it came before the movement.", 24, MUTED),
        ).arrange(DOWN, buff=0.35).move_to([0, 0.4, 0])
        self.play(LaggedStart(*[FadeIn(l, shift=UP * 0.1) for l in lines], lag_ratio=0.5), run_time=3.0)
        self.wait(4.0)
