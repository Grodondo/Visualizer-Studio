"""What happens in your brain when you react to a punch?  Illustration, anatomy, then real EEG (about 2 minutes).

    uv run python -m compute.punch_reaction                         # download ERP CORE + fsaverage once (~520 MB), analyse, export
    uv run python -m compute.brain_render                           # draw the brain images (about 25 s) into media/brain_frames/
    uv run manim -ql scenes/punch_reaction.py PunchReaction         # draft, 480p15
    uv run python scenes/studio/inspect_video.py media/videos/punch_reaction/480p15/PunchReaction.mp4 --per-beat 3

Render one beat (PowerShell):  $env:BEATS = "seeing"; uv run manim -ql scenes/punch_reaction.py PunchReaction; Remove-Item Env:BEATS
Beats: punch, pathway, experiment, seeing, preparing, fighters, recap.

What is what (a corner tag says it on screen):
  ILLUSTRATION  the boxers. Not a recording of anyone.
  ANATOMY       the brain is the fsaverage template cortex (FreeSurfer); colored regions are Desikan-Killiany atlas areas,
                which only approximate functional areas. Stage timings in that beat are published averages (cited on screen).
  MEASURED      EEG of one person (ERP CORE, Flankers task, subject 001; Kappenman et al. 2021; CC BY-SA 4.0): waveforms, reaction times.
  ESTIMATED     where on the cortex the measured EEG most likely came from (dSPM on the template brain). Shown only up to
                the time the estimate stays consistent (trusted_until_s in the trace).
"""
import numpy as np
from manim import *

from studio.beats import BeatScene
from studio.eeg import HeadMap, polyline
from studio.punch import BrainFrames, Boxer, PunchTrace
from studio.theme import BG, FONT, GOLD, MUTED, NEG, POS, TEXT

# ----------------------------------------------------------------------------------------------
# KNOBS. Positions are scene units: the screen is 14.22 wide and 8 high, (0, 0) is the center, y grows upward.
# ----------------------------------------------------------------------------------------------
BRAIN_MAIN = (-2.9, 0.15, 7.6)  # x, y, WIDTH of the big lateral brain in beat "pathway"
BRAIN_MEDIAL = (4.1, 1.4, 3.6)  # x, y, WIDTH of the small "seen from the midline" brain
LEGEND_TOP_LEFT = (2.45, 0.0)  # where the list of the four stages starts
ACTIVITY_BRAINS = ((1.8, 0.85, 3.6), (5.1, 0.85, 3.6))  # x, y, height of the back and right-back views in beat "seeing"
WAVE_PLOT = dict(center=(-3.4, 0.6), size=(5.6, 2.8))  # the visual-response plot in beat "seeing"
SLOW_MOTION = 40.0  # beat "seeing" plays brain time this many times slower than real time
LRP_PLOT = dict(center=(-1.6, 0.4), size=(7.2, 3.0))
RT_PLOT = dict(center=(2.4, 0.1), size=(6.2, 2.8))
RT_BIN = 0.025  # s, width of one bar of the reaction-time histogram
FOOTER_Y = -3.68

TAG_COLORS = {"ILLUSTRATION": GOLD, "ANATOMY": "#8fd694", "MEASURED": POS, "ESTIMATED": "#ff7ab6"}


def label(text: str, size: float = 24, color=TEXT, **kw) -> Text:
    return Text(text, font=FONT, font_size=size, color=color, **kw)


def ms(seconds: float, step: int = 10) -> str:
    """'about 130 ms' style rounding for latencies quoted in captions."""
    return f"{int(round(seconds * 1000 / step) * step)} ms"


class PunchReaction(BeatScene):
    caption_y = -2.95

    def construct(self):
        self.trace = PunchTrace.load()
        self.brains = BrainFrames()
        self.stage_color = {s["key"]: s["color"] for s in self.brains.stages}
        self.tag = None
        self.foot = None
        self.beat_punch()
        self.beat_pathway()
        self.beat_experiment()
        self.beat_seeing()
        self.beat_preparing()
        self.beat_fighters()
        self.beat_recap()

    # ---- helpers -------------------------------------------------------------------------------

    def set_tag(self, kind: str | None, note: str = ""):
        """Corner tag saying what kind of picture is on screen, with an optional one-line note under it."""
        new = None
        if kind:
            color = TAG_COLORS[kind]
            t = label(kind, 18, color, weight=BOLD)
            box = SurroundingRectangle(t, buff=0.09, corner_radius=0.06, color=color, stroke_width=1.5)
            new = VGroup(box, t).to_corner(UR, buff=0.3)
            if note:
                new.add(label(note, 14, MUTED).next_to(box, DOWN, buff=0.08, aligned_edge=RIGHT))
        anims = ([FadeOut(self.tag)] if self.tag else []) + ([FadeIn(new)] if new else [])
        self.tag = new
        if anims:
            self.play(*anims, run_time=0.4)

    def set_footer(self, text: str | None):
        new = label(text, 15, MUTED).move_to([0, FOOTER_Y, 0]) if text else None
        anims = ([FadeOut(self.foot)] if self.foot else []) + ([FadeIn(new)] if new else [])
        self.foot = new
        if anims:
            self.play(*anims, run_time=0.4)

    def data_footer(self):
        d = self.trace["dataset"]
        self.set_footer(f"Measured: {d['name']} · {d['citation']} · {d['license']}")

    def clear_stage(self, *keep, run_time: float = 0.6):
        keep = set(keep) | {self.tag, self.foot, self.caption_now}
        gone = [m for m in self.mobjects if m not in keep]
        if gone:
            self.play(*[FadeOut(m) for m in gone], run_time=run_time)

    def plot_axes(self, x_range, y_range, center, size, x_ticks, y_ticks, x_unit="ms", y_label="µV", invert=False):
        """Axes with Pango tick labels (no LaTeX). invert=True puts negative voltage up (an EEG convention, said on screen)."""
        yr = [-y_range[1], -y_range[0], y_range[2]] if invert else y_range
        axes = Axes(x_range=x_range, y_range=yr, x_length=size[0], y_length=size[1],
                    axis_config={"color": MUTED, "include_tip": False, "include_numbers": False}).move_to([center[0], center[1], 0])
        axes.x_axis.shift(axes.c2p(0, yr[0]) - axes.c2p(0, 0))
        sign = -1 if invert else 1
        xt = VGroup(*[label(f"{v * 1000:.0f}".replace("-", "−") if x_unit == "ms" else f"{v:g}", 17, MUTED)
                      .next_to(axes.c2p(v, yr[0]), DOWN, buff=0.1) for v in x_ticks])
        xu = label(x_unit, 17, MUTED).next_to(axes.c2p(x_range[1], yr[0]), DOWN, buff=0.1).shift(RIGHT * 0.45)
        fmt = (lambda v: f"{v:g}") if y_label == "trials" else (lambda v: f"{v:+g}".replace("-", "−") if v else "0")
        yt = VGroup(*[label(fmt(v), 17, MUTED).next_to(axes.c2p(x_range[0], sign * v), LEFT, buff=0.1) for v in y_ticks])
        yl = label(y_label, 17, MUTED).next_to(axes.c2p(x_range[0], yr[1]), UP, buff=0.12)
        zero = DashedLine(axes.c2p(x_range[0], 0), axes.c2p(x_range[1], 0), color=MUTED, stroke_width=1.2, dash_length=0.07)
        return axes, VGroup(axes, xt, xu, yt, yl, zero)

    def curve(self, axes, times, values, color, invert=False, width=3.0):
        sign = -1 if invert else 1
        inside = (times >= axes.x_range[0]) & (times <= axes.x_range[1])
        if inside.sum() < 2:
            return VMobject()
        return VMobject(color=color, stroke_width=width).set_points_as_corners(
            [axes.c2p(t, sign * v) for t, v in zip(times[inside], values[inside])])

    # ---- beat 1: the punch (illustration) ------------------------------------------------------------

    def beat_punch(self):
        self.beat("punch")
        title = label("What happens in your brain when you react to a punch?", 32).to_edge(UP, buff=0.45).to_edge(LEFT, buff=0.6)
        attacker = Boxer(color=NEG, facing=RIGHT, scale=1.15).move_to([-2.6, -0.3, 0])
        defender = Boxer(color=POS, facing=LEFT, scale=1.15).move_to([1.6, -0.3, 0])
        floor = Line([-5.5, -1.75, 0], [5.5, -1.75, 0], color=MUTED, stroke_width=2)
        self.play(Write(title), run_time=1.2)
        self.set_tag("ILLUSTRATION", "a drawing, not a recording")
        self.play(Create(floor), FadeIn(attacker), FadeIn(defender), run_time=0.9)
        amount = ValueTracker(0.0)
        attacker.add_updater(lambda m: m.set_lead(amount.get_value()))
        self.play(amount.animate.set_value(0.55), run_time=0.45, rate_func=rush_into)
        self.swap_caption("A jab is coming. Between seeing it and moving, the brain has a fraction of a second.")
        freeze = label("time stops here", 22, GOLD).next_to(attacker.lead_glove, UP, buff=0.35)
        self.play(FadeIn(freeze), Indicate(attacker.lead_glove, color=GOLD), run_time=0.8)
        self.wait(1.4)
        self.swap_caption("Light from the glove enters the eyes. What happens next?")
        eye = Dot(defender.head.get_center() + LEFT * 0.12 + UP * 0.05, radius=0.06, color=GOLD)
        ray = DashedLine(attacker.lead_glove.get_center(), eye.get_center(), color=GOLD, stroke_width=2.5)
        self.play(Create(ray), FadeIn(eye), run_time=0.8)
        self.wait(1.2)
        attacker.clear_updaters()
        self.play(FadeOut(VGroup(title, attacker, defender, floor, freeze, ray, eye)), run_time=0.7)

    # ---- beat 2: the pathway on a real brain (anatomy) --------------------------------------------------

    def beat_pathway(self):
        self.beat("pathway")
        b = self.brains
        self.set_tag("ANATOMY", "template brain · atlas areas are approximate")
        self.set_footer("Brain: fsaverage template (Fischl et al. 1999) · areas: Desikan-Killiany atlas (Desikan et al. 2006)")
        x, y, h = BRAIN_MAIN
        mx, my, mh = BRAIN_MEDIAL
        main = b.anatomy("left", 0).set_width(h).move_to([x, y, 0])
        medial = b.anatomy("left_medial", 0).set_width(mh).move_to([mx, my, 0])
        main_lbl = label("left side of the brain", 20, MUTED).move_to([x, 2.75, 0])
        med_lbl = label("same half, seen from the middle", 18, MUTED).move_to([mx, 2.85, 0])
        front = label("front", 18, MUTED).move_to([x - 3.2, y - 2.25, 0])
        back = label("back", 18, MUTED).move_to([x + 3.2, y - 2.25, 0])
        self.play(FadeIn(main), FadeIn(main_lbl), FadeIn(front), FadeIn(back), run_time=1.0)
        self.swap_caption("A real human cortex (an average brain shape), seen from the left. The folds are real anatomy.")
        self.wait(1.6)
        self.play(FadeIn(medial), FadeIn(med_lbl), run_time=0.8)
        self.wait(0.6)
        stage_text = [
            ("1  Visual input", "Signals from the eyes reach primary visual cortex (V1), at the very back, via the thalamus (not shown)."),
            ("2  Processing", "Visual and parietal areas work out what is moving, and where."),
            ("3  Motor planning", "Premotor cortex prepares the response."),
            ("4  Movement", "Primary motor cortex sends the command down the spinal cord to the muscles."),
        ]
        legend = VGroup()
        for k, (stage, (head, text)) in enumerate(zip(b.stages, stage_text), start=1):
            new_main = b.anatomy("left", k).set_width(h).move_to(main)
            new_med = b.anatomy("left_medial", k).set_width(mh).move_to(medial)
            entry = VGroup(Square(0.22, stroke_width=0, fill_color=stage["color"], fill_opacity=1),
                           VGroup(label(head, 21, stage["color"]), label(stage["name"], 16, MUTED)).arrange(DOWN, aligned_edge=LEFT, buff=0.04)
                           ).arrange(RIGHT, buff=0.15)
            legend.add(entry)
            legend.arrange(DOWN, aligned_edge=LEFT, buff=0.16).move_to([LEGEND_TOP_LEFT[0], LEGEND_TOP_LEFT[1], 0], aligned_edge=UL)
            self.play(FadeIn(new_main), FadeIn(new_med), FadeIn(entry, shift=RIGHT * 0.1), run_time=0.9)
            self.remove(main, medial)
            main, medial = new_main, new_med
            self.swap_caption(text)
            self.wait(2.2)
        # cited timeline
        self.clear_stage()
        self.timeline()

    def timeline(self):
        """Published averages, not this video's data: V1 onset (Foxe & Simpson 2002), simple reaction time (Woods et al. 2015)."""
        y = 0.4
        axis = NumberLine(x_range=[0, 300, 50], length=10, color=MUTED, include_numbers=False).move_to([0, y, 0])
        ticks = VGroup(*[label(f"{v}", 18, MUTED).next_to(axis.n2p(v), DOWN, buff=0.15) for v in range(0, 301, 50)])
        unit = label("ms after the light reaches the eye", 18, MUTED).next_to(ticks, DOWN, buff=0.15)
        self.play(Create(axis), FadeIn(ticks), FadeIn(unit), run_time=0.8)
        marks = [
            (56, "visual cortex starts to respond", "Foxe & Simpson 2002", self.stage_color["input"], UP),
            (80, "frontal cortex already active", "Foxe & Simpson 2002", self.stage_color["processing"], DOWN),
            (231, "average simple reaction time", "Woods et al. 2015 (213 ms after\ncorrecting for equipment delay)", self.stage_color["movement"], UP),
        ]
        for value, what, who, color, side in marks:
            p = axis.n2p(value)
            stem = Line(p, p + side * 1.0, color=color, stroke_width=3)
            dot = Dot(p, color=color, radius=0.08)
            txt = VGroup(label(f"{value} ms: {what}", 20, color), label(who, 15, MUTED, line_spacing=0.8)).arrange(DOWN, buff=0.05)
            txt.next_to(stem, side, buff=0.08)
            if side is DOWN:
                txt.next_to(unit, DOWN, buff=0.15).set_x(p[0])
                stem.put_start_and_end_on(p, [p[0], txt.get_top()[1] + 0.05, 0])
            self.play(GrowFromCenter(dot), Create(stem), FadeIn(txt), run_time=0.7)
            self.wait(0.6)
        self.swap_caption("Published lab averages for a simple “see it, press it” task. Not measured on boxers.")
        self.wait(2.6)
        self.clear_stage()

    # ---- beat 3: the experiment (measured) ------------------------------------------------------------

    def beat_experiment(self):
        self.beat("experiment")
        tr = self.trace
        self.set_tag("MEASURED")
        self.data_footer()
        # the task, drawn
        screen = RoundedRectangle(width=3.6, height=1.5, corner_radius=0.1, color=MUTED, stroke_width=3).move_to([-4.4, 1.2, 0])
        arrows = label("◀ ◀ ▶ ◀ ◀", 34, TEXT).move_to(screen)
        task = label("press the button for the MIDDLE arrow", 18, MUTED).next_to(screen, DOWN, buff=0.15)
        self.play(Create(screen), FadeIn(arrows), FadeIn(task), run_time=0.8)
        t = tr["trials"]
        self.swap_caption(f"One volunteer, {t['total']} trials: arrows flash, press left or right as fast as possible.")
        self.wait(1.8)
        # histogram of the counted reaction times
        rts = tr.rts
        edges = np.arange(0.2, 0.8 + 1e-9, RT_BIN)
        counts, _ = np.histogram(rts, edges)
        axes, frame = self.plot_axes([0.2, 0.8, 0.1], [0, counts.max() * 1.15, 10], RT_PLOT["center"], RT_PLOT["size"],
                                     x_ticks=[0.2, 0.4, 0.6, 0.8], y_ticks=[0, 20, 40], y_label="trials")
        bars = VGroup(*[Rectangle(width=axes.c2p(edges[i + 1], 0)[0] - axes.c2p(edges[i], 0)[0] - 0.02,
                                  height=max(axes.c2p(0, c)[1] - axes.c2p(0, 0)[1], 0.001), stroke_width=0,
                                  fill_color=self.stage_color["movement"], fill_opacity=0.85)
                        .move_to(axes.c2p((edges[i] + edges[i + 1]) / 2, 0), aligned_edge=DOWN) for i, c in enumerate(counts)])
        self.play(FadeIn(frame), run_time=0.6)
        self.play(LaggedStart(*[GrowFromEdge(bar, DOWN) for bar in bars], lag_ratio=0.05), run_time=1.6)
        med = tr["rt"]["median_s"]
        med_line = DashedLine(axes.c2p(med, 0), axes.c2p(med, counts.max() * 1.1), color=TEXT, stroke_width=2.5)
        med_lbl = label(f"median {ms(med, 1)}", 20).next_to(med_line, UP, buff=0.08)
        self.play(Create(med_line), FadeIn(med_lbl), run_time=0.6)
        self.swap_caption(f"{t['counted']} correct presses. Half were faster than {ms(med, 1)}: a choice takes longer than a simple reaction.")
        self.wait(2.4)
        c, i = tr["rt"]["compatible_median_s"], tr["rt"]["incompatible_median_s"]
        self.swap_caption(f"When the outer arrows point the other way, the press is slower: {ms(c, 1)} vs {ms(i, 1)} (medians).")
        self.wait(2.6)
        self.clear_stage()

    # ---- beat 4: seeing, in slow motion (measured + estimated) ----------------------------------------------

    def beat_seeing(self):
        self.beat("seeing")
        tr, b = self.trace, self.brains
        until = tr["trusted_until_s"]
        times, uv = tr.visual()
        axes, frame = self.plot_axes([-0.05, until, 0.05], [-9, 4, 1], WAVE_PLOT["center"], WAVE_PLOT["size"],
                                     x_ticks=[0, 0.1, 0.2], y_ticks=[-8, -4, 0, 4], invert=True)
        title = label("measured at the back of the head", 20, MUTED).next_to(frame, UP, buff=0.3)
        chans = label("(electrodes " + ", ".join(tr["processing"]["visual_channels"]) + ")", 15, MUTED).next_to(title, DOWN, buff=0.06)
        wave = self.curve(axes, times, uv, self.stage_color["input"], invert=True)
        self.play(FadeIn(frame), FadeIn(title), FadeIn(chans), run_time=0.7)
        self.swap_caption(f"Average of {tr['trials']['stim_epochs_kept']} trials, time 0 = arrows appear. Negative voltage is drawn upward (EEG habit).")
        self.wait(1.8)
        # the brains, synchronized with a cursor
        clock = ValueTracker(-0.05)
        brains = Group()
        names = {"back": "from behind", "right_back": "right side, from behind"}
        for (x, y, h), view in zip(ACTIVITY_BRAINS, b.manifest["activity"]["views"]):
            img = b.player(view, clock, h).move_to([x, y, 0])
            brains.add(Group(img, label(names[view], 16, MUTED).next_to(img, DOWN, buff=-0.15)))
        est = VGroup(label("estimated activity on the cortex", 20, TAG_COLORS["ESTIMATED"]),
                     label("where the measured EEG most likely came from", 15, MUTED)).arrange(DOWN, buff=0.05).next_to(brains, UP, buff=0.0)
        lo, hi = axes.y_range[0], axes.y_range[1]  # drawn (inverted) range
        cursor = always_redraw(lambda: Line(axes.c2p(clock.get_value(), lo), axes.c2p(clock.get_value(), hi), color=GOLD, stroke_width=2))
        readout = always_redraw(lambda: label(f"{clock.get_value() * 1000:4.0f} ms", 24, GOLD).next_to(axes, DOWN, buff=0.55).align_to(axes, LEFT))
        partial = always_redraw(lambda: self.curve(axes, times[times <= clock.get_value()], uv[times <= clock.get_value()],
                                                   self.stage_color["input"], invert=True))
        self.set_tag("ESTIMATED")
        self.play(FadeIn(brains), FadeIn(est), FadeIn(cursor), FadeIn(readout), run_time=0.8)
        self.add(partial)
        self.swap_caption(f"Now in slow motion ({SLOW_MOTION:.0f}× slower). Color = estimated activity, only where it beats the noise before the arrows.")
        p1, n1 = tr["visual"]["p1"], tr["visual"]["n1"]
        run = lambda t_to: self.play(clock.animate.set_value(t_to), run_time=(t_to - clock.get_value()) * SLOW_MOTION, rate_func=linear)  # noqa: E731
        run(p1["latency_s"])
        p1_lbl = label(f"first wave ~{ms(p1['latency_s'])}", 16, self.stage_color["input"]).next_to(axes.c2p(p1["latency_s"], -p1["amplitude_uv"]), LEFT, buff=0.12).shift(DOWN * 0.12)
        self.add(p1_lbl)
        self.swap_caption("The first wave: visual cortex at the back responds to the arrows.")
        run(n1["latency_s"])
        n1_lbl = label(f"big wave, about {ms(n1['latency_s'])}", 18, self.stage_color["processing"]).next_to(axes.c2p(n1["latency_s"], -n1["amplitude_uv"]), UP, buff=0.12)
        self.add(n1_lbl)
        self.swap_caption("Then a larger wave: activity spreads from visual cortex into parietal cortex, strongest on the right.")
        run(until)
        self.wait(1.4)
        self.remove(partial)
        self.add(wave)
        self.swap_caption(f"After about {ms(until)} the estimate drifts to deep midline areas that 30 scalp electrodes cannot resolve. We stop here.")
        self.wait(2.8)
        self.swap_caption("Times count from the computer's marker; the screen itself lights up a few tens of milliseconds later.")
        self.wait(2.2)
        for m in (cursor, readout):
            m.clear_updaters()
        for g in brains:
            g[0].clear_updaters()
        self.clear_stage()

    # ---- beat 5: preparing the press (measured) -----------------------------------------------------------

    def beat_preparing(self):
        self.beat("preparing")
        tr = self.trace
        self.set_tag("MEASURED")
        head = HeadMap(tr["electrodes"], radius=1.05, dot_radius=0.06).move_to([4.6, 1.3, 0])
        for ch in ("C3", "C4"):
            head.dot_of[ch].set_color(self.stage_color["planning"]).scale(1.6)
        c_lbls = VGroup(head.label("C3", DOWN, 18, self.stage_color["planning"]), head.label("C4", DOWN, 18, self.stage_color["planning"]))
        self.play(FadeIn(head), FadeIn(c_lbls), run_time=0.7)
        self.swap_caption("Now line the trials up on the button press instead, and look over the two motor areas (C3, C4).")
        self.wait(1.8)
        times, uv = tr.lrp()
        axes, frame = self.plot_axes([-0.6, 0.2, 0.1], [-4, 1, 1], LRP_PLOT["center"], LRP_PLOT["size"],
                                     x_ticks=[-0.6, -0.4, -0.2, 0, 0.2], y_ticks=[-4, -2, 0], invert=True)
        title = label("opposite-side minus same-side motor electrode", 20, MUTED).next_to(frame, UP, buff=0.25)
        press = DashedLine(axes.c2p(0, axes.y_range[0]), axes.c2p(0, axes.y_range[1]), color=self.stage_color["movement"], stroke_width=2.5)
        press_lbl = label("button pressed", 18, self.stage_color["movement"]).next_to(axes.c2p(0, 4), UP, buff=0.05)
        line = self.curve(axes, times, uv, self.stage_color["planning"], invert=True)
        self.play(FadeIn(frame), FadeIn(title), Create(press), FadeIn(press_lbl), run_time=0.8)
        self.play(Create(line), run_time=3.0, rate_func=linear)
        onset = tr["lrp_onset_s"]
        peak_i = int(np.argmin(np.where(times <= 0, uv, np.inf)))
        on_dot = Dot(axes.c2p(onset, -uv[int(np.argmin(np.abs(times - onset)))]), color=GOLD)
        on_lbl = label(f"half-way up: {ms(-onset)} before the press", 18, GOLD).next_to(on_dot, LEFT, buff=0.15).shift(UP * 0.35)
        self.play(GrowFromCenter(on_dot), FadeIn(on_lbl), run_time=0.6)
        self.swap_caption(f"The motor area opposite the pressing hand charges up first: the brain committing to one hand, {ms(-onset)} before the press.")
        self.wait(2.8)
        self.swap_caption(f"It peaks {ms(-times[peak_i])} before the press. In the last moments the command travels down to the finger muscles.")
        self.wait(2.4)
        self.clear_stage()
        # the limitation, on the brain
        self.set_tag("ESTIMATED")
        base = self.brains.limit("base").set_height(4.4).move_to([-2.6, 0.2, 0])
        smear = self.brains.limit("hand_contrast").set_height(4.4).move_to(base)
        lbl = label("brain seen from above, front up", 17, MUTED).next_to(base, DOWN, buff=-0.1)
        lr = VGroup(label("left", 17, MUTED).next_to(base, LEFT, buff=-0.2), label("right", 17, MUTED).next_to(base, RIGHT, buff=-0.2))
        self.play(FadeIn(base), FadeIn(lbl), FadeIn(lr), run_time=0.6)
        self.play(FadeIn(smear), run_time=1.0)
        split = tr["where"]["hand_contrast_central_mean"]
        text = VGroup(label("Where exactly is the press prepared?", 24),
                      label("Estimated difference between right- and left-hand presses,", 18, MUTED),
                      label("averaged over the last 150 ms before the press.", 18, MUTED),
                      label(f"Motor strip, left vs right half: {split['lh']:.2f} vs {split['rh']:.2f}", 18, MUTED),
                      label("With 30 electrodes and no MRI of this person,", 18, TEXT),
                      label("the estimate smears over both halves.", 18, TEXT),
                      label("The scalp signal is clear; its source is not.", 18, GOLD)).arrange(DOWN, aligned_edge=LEFT, buff=0.12).move_to([3.6, 0.3, 0])
        self.play(FadeIn(text, shift=UP * 0.1), run_time=0.8)
        self.swap_caption("Textbooks place it in the left motor cortex for the right hand. This data cannot show that on its own.")
        self.wait(3.4)
        self.clear_stage()

    # ---- beat 6: back to the ring --------------------------------------------------------------------------

    def beat_fighters(self):
        self.beat("fighters")
        tr = self.trace
        self.set_tag("ILLUSTRATION")
        self.set_footer("Mori, Ohtani & Imanaka 2002, Hum Mov Sci (karate athletes) · Woods et al. 2015, Front Hum Neurosci")
        attacker = Boxer(color=NEG, facing=RIGHT, scale=0.95).move_to([-5.0, 0.2, 0])
        defender = Boxer(color=POS, facing=LEFT, scale=0.95).move_to([-1.9, 0.2, 0])
        self.play(FadeIn(attacker), FadeIn(defender), run_time=0.6)
        rows = VGroup(
            label(f"this lab task (a choice):  median {ms(tr['rt']['median_s'], 1)}", 21, TEXT),
            label("simple “see it, press it”:  about 213–231 ms on average", 21, TEXT),
            label("Experts are not mainly faster at reacting.", 22, GOLD),
            label("They anticipate: they read the opponent's body", 21, MUTED),
            label("and start responding before the punch is thrown.", 21, MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.16)
        rows.set(width=min(rows.width, 6.3)).move_to([3.3, 0.6, 0])
        self.play(FadeIn(rows[:2], shift=UP * 0.1), run_time=0.7)
        self.swap_caption("Even a quick reaction takes about a quarter of a second, and a choice takes longer.")
        self.wait(2.2)
        cue = label("shoulder starts to turn", 18, GOLD).next_to(attacker.head, UP, buff=0.3)
        self.play(FadeIn(rows[2:], shift=UP * 0.1), FadeIn(cue), Indicate(attacker.body, color=GOLD), run_time=0.9)
        self.swap_caption("Karate athletes beat novices mostly by anticipating the attack, not by faster raw reactions (Mori et al. 2002).")
        amount = ValueTracker(0.0)
        attacker.add_updater(lambda m: m.set_lead(amount.get_value()))
        self.play(amount.animate.set_value(1.0), defender.animate.shift(RIGHT * 0.5 + DOWN * 0.15), run_time=0.6)
        attacker.clear_updaters()
        self.wait(2.4)
        self.clear_stage()
        self.set_footer(None)

    # ---- beat 7: recap --------------------------------------------------------------------------------------

    def beat_recap(self):
        self.beat("recap")
        tr = self.trace
        if self.caption_now:
            self.play(FadeOut(self.caption_now), run_time=0.3)
            self.caption_now = None
        self.set_tag(None)
        p1, n1 = tr["visual"]["p1"]["latency_s"], tr["visual"]["n1"]["latency_s"]
        rows = [
            ("input", "1  See", f"visual cortex responds (measured wave at about {ms(p1)})"),
            ("processing", "2  Process", f"activity spreads to parietal cortex (about {ms(n1)}; estimated)"),
            ("planning", "3  Plan", f"one hand's motor area charges up ({ms(-tr['lrp_onset_s'])} before the press; measured)"),
            ("movement", "4  Move", f"the press, median {ms(tr['rt']['median_s'], 1)} after the arrows (measured)"),
        ]
        group = VGroup(label("Reacting, in one person's brain", 32))
        for key, head, text in rows:
            group.add(VGroup(label(head, 24, self.stage_color[key]), label(text, 21, MUTED)).arrange(RIGHT, buff=0.3))
        group.arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to([0, 0.5, 0])
        group[0].set_x(0)
        note = label("The boxers were drawings; the brain is a template; the EEG is real (ERP CORE, subject 001).", 18, MUTED).next_to(group, DOWN, buff=0.5)
        self.play(LaggedStart(*[FadeIn(g, shift=UP * 0.1) for g in group], lag_ratio=0.4), run_time=2.6)
        self.play(FadeIn(note), run_time=0.6)
        self.wait(4.0)
