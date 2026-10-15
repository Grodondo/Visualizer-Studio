"""FlyDoom draft (Milestone 1, revised): real gameplay beside the measured fly brain. About 53 seconds, almost no text.

    uv run --group flydoom python -m flydoom.seeds        # distinct seed sets (once)
    uv run --group flydoom python -m flydoom.trials       # all controllers on the same seeds + the footage clips
    uv run python -m flydoom.brain_frames                 # brain images for this cut (needs media/fly_frames from compute.fly_render)
    uv run manim -ql scenes/flydoom_draft.py FlyDoomDraft
    uv run python scenes/studio/inspect_video.py media/videos/flydoom_draft/480p15/FlyDoomDraft.mp4 --per-beat 3

What is on screen, and its status (shown by visual style, see the legend chips):
  GAMEPLAY   recorded, reproducible episodes of the REFLEX controller, test seeds in order (assets/flydoom/footage.json).
  EYE        engineered sensor (dashed boxes): the looming eye's expansion, left = cyan, right = amber.
  KEYS       actions (keycaps), chosen by the reflex from the eye only. Chain drawn: eye -> reflex -> keys.
  FLY BRAIN  measured FlyWire anatomy (dots, synapses). It is NOT simulated and is NOT connected to the game: nothing in the
             brain reacts to gameplay. Its camera moves follow a fixed schedule, not game events.
  COMPARISON survival of the baseline controllers on 10 distinct seeds (assets/flydoom/results.json).
Beats: open, brain, circuit, turn, compare.
"""
import json
from pathlib import Path

import numpy as np
from manim import *
from PIL import Image

from studio.beats import BeatScene
from studio.fly import camera
from studio.theme import BG, FONT, GOLD, MUTED, TEXT

ROOT = Path(__file__).resolve().parent.parent
A = ROOT / "assets"
RESULTS, FOOTAGE, FLY_IO = A / "flydoom" / "results.json", A / "flydoom" / "footage.json", A / "flydoom" / "fly_io.json"
BRAIN_DIR, TURNTABLE_DIR = ROOT / "media" / "flydoom_brain", ROOT / "media" / "fly_frames"
POINTS = A / "connectome" / "brain_points.npz"

# ----------------------------------------------------------------------------------------------
# KNOBS. Scene units: the screen is 14.22 x 8. Every color keeps ONE meaning in the whole cut.
# ----------------------------------------------------------------------------------------------
LEFT_C, RIGHT_C = ManimColor("#5ec8ff"), ManimColor("#ffb454")  # left eye / right eye, and anatomy fed by it
OUT_C = ManimColor(TEXT)  # outputs (descending neurons) and actions
EYE_OVERLAY_OPACITY = 0.55
GAME_BIG = dict(center=(0.0, 0.35), width=8.6)  # cold open
GAME_INSET = dict(center=(-4.85, 2.05), width=3.9)
BRAIN = dict(center=(2.15, 0.45), width=10.0)  # the centerpiece
CONTROLS_Y = -0.45  # eye bars and keys, under the inset
TIMELINE = dict(y=(-3.5, -2.75), x=(-6.25, 6.85))
EXP_MAX = 60.0  # deg/s at full bar / trace height (clipped above)
T_OPEN, T_SHRINK = 3.0, 1.2  # seconds of fullscreen gameplay, then the move to the inset
CIRCUIT_TURN_S, BRAIN_TURN_S = 12.0, 8.0
TURN_ZOOM_START = 1.3  # the whole-brain turn starts this magnified, then pulls back
COMPARE_ROWS = [("idle", "idle"), ("constant", "always left"), ("random", "random"), ("scripted", "scripted AI"), ("reflex", "reflex")]


def label(text, size=16, color=TEXT, **kw) -> Text:
    return Text(text, font=FONT, font_size=size, color=color, **kw)


class FlyDoomDraft(BeatScene):
    def construct(self):
        self.load()
        self.clock = ValueTracker(0.0)  # seconds of gameplay shown so far; runs with real time once added
        self.beat_open()
        self.beat_brain()
        self.beat_circuit()
        self.beat_turn()
        self.beat_compare()

    # ---- data ------------------------------------------------------------------------------------

    def load(self):
        self.results = json.loads(RESULTS.read_text(encoding="utf-8"))
        foot = json.loads(FOOTAGE.read_text(encoding="utf-8"))
        self.clips = [json.loads((A / "flydoom" / c).read_text(encoding="utf-8")) for c in foot["clips"]]
        self.io = json.loads(FLY_IO.read_text(encoding="utf-8"))
        self.bm = json.loads((BRAIN_DIR / "manifest.json").read_text(encoding="utf-8"))
        self.tm = json.loads((TURNTABLE_DIR / "manifest.json").read_text(encoding="utf-8"))
        self.rate = self.clips[0]["ticrate"]
        self.starts = np.cumsum([0] + [c["tics"] for c in self.clips])  # global tic where each episode starts
        self.total_tics = int(self.starts[-1])
        self.T = self.total_tics / self.rate
        self.actions = sum((c["actions"] for c in self.clips), [])
        self.eye = np.array(sum((c["eye"] for c in self.clips), []))
        self.onsets = []  # (global tic, side) of every new fireball (game truth)
        for k, c in enumerate(self.clips):
            seen = set()
            for t, balls in enumerate(c["fireballs"]):
                for oid, x, _ in balls:
                    if oid not in seen:
                        seen.add(oid)
                        self.onsets.append((int(self.starts[k]) + t, "left" if x < 160 else "right"))

    def gtic(self) -> int:
        return int(np.clip(self.clock.get_value() * self.rate, 0, self.total_tics - 1))

    def local(self, g: int) -> tuple[int, int]:
        k = int(np.searchsorted(self.starts, g, side="right") - 1)
        return k, g - int(self.starts[k])

    def start_clock(self):
        self.clock.add_updater(lambda m, dt: m.set_value(min(self.T, m.get_value() + dt)))
        self.add(self.clock)

    # ---- placing anatomy (seen from behind, like the game: the fly's left on the viewer's left) ----

    def to_brain(self, xyz) -> np.ndarray:
        s = (np.atleast_2d(np.asarray(xyz, float)) - np.array(self.bm["center_um"])) @ camera(self.bm["angle"]).T
        k = BRAIN["width"] / (2 * self.bm["half_width_um"])
        out = np.zeros((len(s), 3))
        out[:, 0] = BRAIN["center"][0] + k * s[:, 0]
        out[:, 1] = BRAIN["center"][1] + k * s[:, 1]
        return out

    def image(self, path: Path, width: float, center) -> ImageMobject:
        return ImageMobject(str(path)).set_width(width).move_to([center[0], center[1], 0])

    def sequence(self, paths: list, value, width: float, center) -> ImageMobject:
        """An image that shows paths[value()] (value is called every frame)."""
        cache = {}
        img = ImageMobject(str(paths[0])).set_width(width).move_to([center[0], center[1], 0])
        state = {"i": 0}

        def update(m):
            i = int(value())
            if i != state["i"]:
                if i not in cache:
                    cache[i] = np.array(Image.open(paths[i]).convert("RGBA"))
                m.pixel_array = cache[i]
                state["i"] = i

        img.add_updater(update)
        return img

    # ---- the game, the eye, the keys, the timeline -------------------------------------------------

    def game_view(self):
        paths, eyes = [], []
        for c in self.clips:
            folder = ROOT / c["frames"]
            paths += [folder / f"{i:04d}.png" for i in range(c["tics"])]
            eyes += [folder / "eye" / f"{i:04d}.png" for i in range(c["tics"])]
        game = self.sequence(paths, self.gtic, GAME_BIG["width"], GAME_BIG["center"]).set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        eye = self.sequence(eyes, self.gtic, GAME_BIG["width"], GAME_BIG["center"]).set_opacity(EYE_OVERLAY_OPACITY)
        frame = always_redraw(lambda: SurroundingRectangle(game, buff=0, color=MUTED, stroke_width=1.2))
        split = always_redraw(lambda: DashedLine(game.get_bottom(), game.get_top(), color=TEXT, stroke_width=1, dash_length=0.06, stroke_opacity=0.35))
        seed = always_redraw(lambda: label(f"seed {self.clips[self.local(self.gtic())[0]]['seed']}", 12, MUTED)
                             .next_to(game.get_corner(UL), DR, buff=0.06))
        return Group(game, eye, frame, split, seed), game

    def controls(self) -> VGroup:
        """Engineered eye (dashed boxes) -> reflex -> keys (solid keycaps). The causal chain of THIS footage."""
        x0 = GAME_INSET["center"][0] - 1.55
        y = CONTROLS_Y
        boxes = VGroup()
        for k, (idx, color) in enumerate(((2, LEFT_C), (3, RIGHT_C))):
            base = np.array([x0 + k * 0.32, y - 0.45, 0])
            outline = DashedVMobject(Rectangle(width=0.2, height=0.9, color=color, stroke_width=1.5), num_dashes=14).move_to(base + UP * 0.45)

            def fill(base=base, idx=idx, color=color):
                h = 0.9 * min(self.eye[self.gtic(), idx] / EXP_MAX, 1.0)
                return Rectangle(width=0.2, height=max(h, 0.001), stroke_width=0, fill_color=color, fill_opacity=0.9).move_to(base + UP * h / 2)
            boxes.add(outline, always_redraw(fill))
        eye_icon = VGroup(Ellipse(width=0.42, height=0.24, color=TEXT, stroke_width=1.5), Dot(radius=0.06, color=TEXT)).next_to(boxes, UP, buff=0.1)
        reflex = VGroup(RoundedRectangle(width=0.9, height=0.34, corner_radius=0.08, color=MUTED, stroke_width=1.2), label("reflex", 13, MUTED))
        reflex.move_to([x0 + 1.45, y, 0])
        keys = VGroup()
        for k, (name, color, rot) in enumerate((("left", LEFT_C, PI / 2), ("right", RIGHT_C, -PI / 2))):
            cap = RoundedRectangle(width=0.42, height=0.42, corner_radius=0.07, color=OUT_C, stroke_width=1.5).move_to([x0 + 2.55 + k * 0.52, y, 0])
            arrow = Triangle(color=OUT_C, fill_opacity=1, stroke_width=0).scale(0.09).rotate(rot).move_to(cap)
            lit = always_redraw(lambda cap=cap, name=name, color=color: cap.copy().set_fill(color, 0.85 if self.actions[self.gtic()] == name else 0).set_stroke(width=0))
            keys.add(lit, cap, arrow)
        a1 = Arrow(boxes.get_right(), reflex.get_left(), buff=0.12, color=MUTED, stroke_width=2, max_tip_length_to_length_ratio=0.25)
        a2 = Arrow(reflex.get_right(), keys.get_left(), buff=0.12, color=MUTED, stroke_width=2, max_tip_length_to_length_ratio=0.25)
        return VGroup(boxes, eye_icon, a1, reflex, a2, keys)

    def legend(self) -> VGroup:
        chip = lambda m, t: VGroup(m, label(t, 12, MUTED)).arrange(RIGHT, buff=0.1)  # noqa: E731
        spike = VMobject(color=MUTED, stroke_width=1.5).set_points_as_corners([[-0.15, 0, 0], [-0.04, 0, 0], [0, 0.14, 0], [0.04, -0.06, 0], [0.08, 0, 0], [0.15, 0, 0]])
        rows = VGroup(
            chip(DashedVMobject(Rectangle(width=0.22, height=0.14, color=LEFT_C, stroke_width=1.5), num_dashes=8), "engineered input"),
            chip(RoundedRectangle(width=0.2, height=0.2, corner_radius=0.04, color=OUT_C, stroke_width=1.5), "action · reflex"),
            chip(VGroup(*[Dot(radius=0.03, color=c).shift(RIGHT * 0.08 * i) for i, c in enumerate((LEFT_C, TEXT, RIGHT_C))]), "measured anatomy"),
            chip(VGroup(spike, Line([-0.17, -0.1, 0], [0.17, 0.12, 0], color=MUTED, stroke_width=1.5)), "simulated activity: none"),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.07)
        return rows.to_corner(DR, buff=0.2).shift(UP * 1.2)

    def timeline(self) -> VGroup:
        (y0, y1), (x0, x1) = TIMELINE["y"], TIMELINE["x"]
        X = lambda g: x0 + (x1 - x0) * g / self.total_tics  # noqa: E731  global tic -> x
        g = np.arange(self.total_tics)
        traces = VGroup()
        for idx, color in ((2, LEFT_C), (3, RIGHT_C)):
            ys = y0 + 0.17 + 0.4 * np.minimum(self.eye[:, idx] / EXP_MAX, 1.0)
            traces.add(VMobject(color=color, stroke_width=1.4, stroke_opacity=0.9).set_points_as_corners(np.c_[X(g), ys, np.zeros(len(g))]))
        ticks = VGroup(*[Line([X(t), y1 - 0.1, 0], [X(t), y1, 0], color=LEFT_C if s == "left" else RIGHT_C, stroke_width=1.5) for t, s in self.onsets])
        keys = VGroup()
        run = None
        for t in range(self.total_tics + 1):  # merge consecutive presses into bars
            a = self.actions[t] if t < self.total_tics else None
            if run and (a != run[0]):
                keys.add(Rectangle(width=max(X(t) - X(run[1]), 0.004), height=0.07, stroke_width=0, fill_opacity=0.9,
                                   fill_color=LEFT_C if run[0] == "left" else RIGHT_C).move_to([(X(t) + X(run[1])) / 2, y0 + 0.05, 0]))
                run = None
            if a in ("left", "right") and run is None:
                run = (a, t)
        seps = VGroup()
        for k, s in enumerate(self.starts[1:-1], start=1):
            seps.add(Line([X(s), y0 - 0.02, 0], [X(s), y1 + 0.02, 0], color=MUTED, stroke_width=1, stroke_opacity=0.6))
        deaths = VGroup(*[Cross(scale_factor=0.05, stroke_color=TEXT, stroke_width=2).move_to([X(e - 1), y1 + 0.1, 0]) for e in self.starts[1:]])
        names = VGroup(*[label(t, 11, MUTED).next_to([x0, y, 0], LEFT, buff=0.06) for t, y in (("threat", y1 - 0.05), ("eye", y0 + 0.3), ("keys", y0 + 0.05))])
        cursor = always_redraw(lambda: Line([X(self.clock.get_value() * self.rate), y0 - 0.06, 0],
                                            [X(self.clock.get_value() * self.rate), y1 + 0.18, 0], color=GOLD, stroke_width=2.5))
        return VGroup(traces, ticks, keys, seps, deaths, names, cursor)

    def keep_front(self):
        """The brain never covers the game, its controls, the timeline or the legend."""
        self.bring_to_front(self.game_group, self.ctrl, self.tl, self.leg)

    # ---- beat 1: cold open on the game ----------------------------------------------------------------

    def beat_open(self):
        self.beat("open")
        self.game_group, self.game = self.game_view()
        self.add(self.game_group)
        self.start_clock()
        self.wait(T_OPEN)
        c = GAME_INSET["center"]
        k = GAME_INSET["width"] / GAME_BIG["width"]
        self.ctrl, self.tl, self.leg = self.controls(), self.timeline(), self.legend()
        self.play(self.game_group[0].animate.scale(k).move_to([c[0], c[1], 0]),
                  self.game_group[1].animate.scale(k).move_to([c[0], c[1], 0]), run_time=T_SHRINK)
        self.play(FadeIn(self.ctrl), FadeIn(self.tl), run_time=0.6)

    # ---- beat 2: the measured brain, highlighted -------------------------------------------------------

    def beat_brain(self):
        self.beat("brain")
        w, c = BRAIN["width"], BRAIN["center"]
        base = self.image(BRAIN_DIR / "back_all.png", w, c).set_opacity(0.7)
        optic = self.image(BRAIN_DIR / "back_optic.png", w, c)
        central = self.image(BRAIN_DIR / "back_central.png", w, c)
        self.play(FadeIn(base, scale=0.96), FadeIn(self.leg), run_time=1.6)
        self.keep_front()
        self.wait(0.8)
        pts = np.load(POINTS)
        names = [str(n) for n in pts["class_names"]]
        sc = self.to_brain(pts["xyz"][pts["cls"] == names.index("optic")])
        lobe_l = sc[sc[:, 0] < c[0]].mean(axis=0)
        lobe_r = sc[sc[:, 0] >= c[0]].mean(axis=0)
        mid = self.to_brain(pts["xyz"][pts["cls"] == names.index("central")]).mean(axis=0)
        ol = VGroup(label("optic lobe", 14, MUTED).move_to(lobe_l + UP * 1.75), label("optic lobe", 14, MUTED).move_to(lobe_r + UP * 1.75))
        self.play(FadeIn(optic), base.animate.set_opacity(0.35), FadeIn(ol), run_time=1.2)
        self.wait(1.8)
        cb = label("central brain", 14, MUTED).move_to(mid + UP * 1.55)
        self.play(FadeOut(optic), FadeIn(central), FadeOut(ol), FadeIn(cb), run_time=1.2)
        self.wait(1.8)
        self.play(FadeOut(central), FadeOut(cb), base.animate.set_opacity(0.55), run_time=1.0)
        # the cells the fly pipeline will use (positions only), colored by the eye that will feed them
        somata = {side: VGroup(*[Dot(self.to_brain(x["xyz"])[0], radius=0.026, color=LEFT_C if side == "left" else RIGHT_C, fill_opacity=0.95)
                                 for x in self.io["cells"] if x["side"] == side and x["type"] in ("LC4", "LPLC2")]) for side in ("left", "right")}
        outs = VGroup(*[Star(5, outer_radius=0.075, color=OUT_C, fill_opacity=1, stroke_width=0).move_to(self.to_brain(x["xyz"])[0])
                        for x in self.io["cells"] if x["type"] in ("DNa01", "DNa02", "DNp01")])
        cl = VGroup(label("LC4 · LPLC2", 14, LEFT_C).next_to(somata["left"], LEFT, buff=0.25),
                    label("LC4 · LPLC2", 14, RIGHT_C).next_to(somata["right"], RIGHT, buff=0.25))
        self.play(LaggedStart(*[GrowFromCenter(d) for d in somata["left"]], lag_ratio=0.01),
                  LaggedStart(*[GrowFromCenter(d) for d in somata["right"]], lag_ratio=0.01), FadeIn(cl), run_time=2.0)
        self.wait(1.4)
        dn = label("descending neurons", 14, OUT_C).next_to(outs, DOWN, buff=0.35)
        self.play(LaggedStart(*[GrowFromCenter(s) for s in outs], lag_ratio=0.15), FadeIn(dn), run_time=1.2)
        self.wait(1.4)
        self.keep_front()
        self.stage = base
        self.stage_marks = VGroup(somata["left"], somata["right"], outs, cl, dn)

    # ---- beat 3: close-up of the measured circuit -----------------------------------------------------

    def beat_circuit(self):
        self.beat("circuit")
        circ = self.bm["circuit"]
        pivot = self.to_brain(circ["pivot_um"])[0]
        k = self.bm["half_width_um"] / circ["half_width_um"]
        c = np.array([*BRAIN["center"], 0])
        self.play(FadeOut(self.stage_marks), run_time=0.6)  # dots would balloon when magnified; only the image zooms
        self.play(self.stage.animate.scale(k, about_point=pivot).shift(c - pivot), run_time=2.4, rate_func=smooth)
        angle = ValueTracker(180.0)
        angles = circ["angles"]
        paths = [BRAIN_DIR / "circuit" / f"{a:03d}.png" for a in angles]
        idx = lambda: int(round((angle.get_value() % 360) / (angles[1] - angles[0]))) % len(angles)  # noqa: E731
        close = self.sequence(paths, idx, BRAIN["width"], BRAIN["center"])
        self.play(FadeIn(close), FadeOut(self.stage), run_time=0.9)
        self.keep_front()
        # labels where the measured synapses are, in this first (unrotated) view
        cam = camera(180)
        kk = BRAIN["width"] / (2 * circ["half_width_um"])
        at = {n: c + kk * np.r_[((np.array(v) - circ["pivot_um"]) @ cam.T)[:2], 0] for n, v in circ["centroids_um"].items()}
        tags = VGroup(label("LPLC2", 14, ManimColor(circ["colors"]["LPLC2"])).next_to(at["LPLC2"], UP, buff=0.55),
                      label("LC4", 14, ManimColor(circ["colors"]["LC4"])).next_to(at["LC4"], DOWN, buff=0.55),
                      label("giant fiber", 14, OUT_C).next_to(at["GF"], RIGHT, buff=1.2).shift(DOWN * 0.8),
                      label("each dot: one synapse", 12, MUTED).move_to(c + UP * 2.55 + RIGHT * 3.3))
        self.play(FadeIn(tags), run_time=0.6)
        self.wait(2.2)
        self.play(FadeOut(tags[:3]), run_time=0.4)
        self.play(angle.animate.set_value(180 + 360), run_time=CIRCUIT_TURN_S, rate_func=linear)
        self.close, self.close_note = close, tags[3]

    # ---- beat 4: back out, one turn of the whole brain ---------------------------------------------------

    def beat_turn(self):
        self.beat("turn")
        tt = self.tm["turntable_angles"]
        angle = ValueTracker(180.0)
        step = tt[1] - tt[0]
        paths = [TURNTABLE_DIR / self.tm["turntable"].format(angle=a) for a in tt]
        spin = self.sequence(paths, lambda: int(round((angle.get_value() % 360) / step)) % len(tt), BRAIN["width"], BRAIN["center"])
        spin.scale(TURN_ZOOM_START)  # start magnified, then pull back
        self.play(FadeIn(spin), FadeOut(self.close), FadeOut(self.close_note), run_time=0.9)
        self.keep_front()
        self.play(spin.animate.scale(1 / TURN_ZOOM_START), run_time=1.6)
        remaining = max(self.T - self.clock.get_value() - 0.3, 2.0)
        self.play(angle.animate.set_value(180 + 360), run_time=min(BRAIN_TURN_S, remaining), rate_func=smooth)
        self.wait(max(self.T - self.clock.get_value(), 0) + 0.4)  # let the last episode finish
        self.spin = spin

    # ---- beat 5: compact comparison -----------------------------------------------------------------------

    def beat_compare(self):
        self.beat("compare")
        r = self.results
        n = len(r["test_seeds"])
        surv = {}
        for e in r["episodes"]:
            surv.setdefault(e["controller"], []).append(e["survival_s"])
        smax = 22.0
        cx, cy = BRAIN["center"]
        x0, x1 = cx - 2.4, cx + 3.6
        X = lambda s: x0 + (x1 - x0) * min(s, smax) / smax  # noqa: E731
        plot = VGroup()
        for i, (key, name) in enumerate(COMPARE_ROWS):
            y = cy + 1.4 - i * 0.62
            v = np.array(surv[key])
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            strong = key == "reflex"
            col = RIGHT_C if strong else TEXT
            plot.add(label(name, 15, col if strong else MUTED).next_to([x0 - 0.2, y, 0], LEFT, buff=0))
            plot.add(Line([X(q1), y, 0], [X(q3), y, 0], color=col, stroke_width=5, stroke_opacity=0.35))
            plot.add(*[Dot([X(s), y + ((j % 3) - 1) * 0.07, 0], radius=0.032, color=col, fill_opacity=0.75) for j, s in enumerate(v)])
            plot.add(Line([X(med), y - 0.17, 0], [X(med), y + 0.17, 0], color=col, stroke_width=3))
        ya = cy + 1.4 - len(COMPARE_ROWS) * 0.62 + 0.25
        plot.add(Line([x0, ya, 0], [x1, ya, 0], color=MUTED, stroke_width=1.2))
        plot.add(*[label(f"{s}", 12, MUTED).next_to([X(s), ya, 0], DOWN, buff=0.06) for s in (0, 5, 10, 15, 20)])
        plot.add(label("s survived", 12, MUTED).next_to([x1, ya, 0], DOWN, buff=0.06).shift(RIGHT * 0.45))
        plot.add(label(f"n = {n} distinct seeds per controller · tick = median · band = middle 50 %", 12, MUTED).move_to([cx + 0.6, cy + 2.05, 0]))
        self.play(self.spin.animate.set_opacity(0.18), FadeOut(self.ctrl), self.game_group[0].animate.set_opacity(0.35),
                  self.game_group[1].animate.set_opacity(0.15), run_time=0.8)
        self.play(LaggedStart(*[FadeIn(m) for m in plot], lag_ratio=0.01), run_time=1.6)
        self.wait(4.5)
