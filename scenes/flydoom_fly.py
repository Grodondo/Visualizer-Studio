"""FlyPlays (Milestone 3): the simulated fly brain plays the game, in a closed loop. About 60 seconds, little text.

First time (needs the FlyWire annotation file in assets/data/flywire, see assets/flydoom/README.md):
    uv run --group flydoom python -m flydoom.fly_export              # fly episodes + simulated spikes (about 2 minutes)
    uv run --group flydoom python -m flydoom.fly_export --shuffle    # the shuffled-wiring control on the first test seed
    uv run --group flydoom python -m flydoom.activity_frames         # spike images, one per game tic (about 3 minutes)
    uv run --group flydoom python -m flydoom.activity_frames --shuffle
    uv run python -m flydoom.brain_frames                            # anatomy image (needs media/fly_frames from compute.fly_render)
Then:
    uv run manim -ql scenes/flydoom_fly.py FlyPlays
    uv run python scenes/studio/inspect_video.py media/videos/flydoom_fly/480p15/FlyPlays.mp4 --per-beat 3

What is on screen, and its status:
  GAMEPLAY    recorded, reproducible episodes of the fly controller on the fixed test seeds (assets/flydoom/fly_footage.json).
  EYE         engineered sensor (dashed bars): the looming eye's expansion, left = cyan, right = amber.
  SPIKES      SIMULATED: which of the model's neurons fired in each 1/35 s step, drawn at their measured FlyWire positions.
              Colors say what a neuron IS, never how strongly it fired: input cells cyan (left) / amber (right),
              steering neurons white, every other neuron magenta.
  DECODER     engineered (dashed bars): steering-neuron rate per side, the larger side wins and becomes the key.
  CONTROL     the same seed with shuffled wiring (assets/flydoom/shuffle_footage.json).
  COMPARISON  survival on 10 distinct seeds (assets/flydoom/fly_results.json, assets/flydoom/results.json).
Beats: open, loop, react, footage, shuffle, compare.
Playback speed is shown on screen; the footage itself is never altered, only played faster or in steps.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from manim import *

from flydoom_draft import A, BRAIN, BRAIN_DIR, CONTROLS_Y, EXP_MAX, GAME_BIG, GAME_INSET, LEFT_C, OUT_C, RIGHT_C, RESULTS, ROOT, T_SHRINK, FlyDoomDraft, label
from studio.theme import GOLD, MUTED, TEXT

FLY_RESULTS = A / "flydoom" / "fly_results.json"

# ----------------------------------------------------------------------------------------------
# KNOBS. Scene units: the screen is 14.22 x 8.
# ----------------------------------------------------------------------------------------------
SIM_C = ManimColor("#ff5fd2")  # simulated activity; the same magenta as the spike images
FREEZE_BEFORE = 4  # the cold open stops this many tics before the first threat appears
SLOW = 0.2  # playback speed while stepping through the first reaction
FAST = 2.5  # playback speed of the rest of the footage
HOLD = 1.6  # seconds a caption stays while the picture is paused
CAPTION_SIZE = 26
STEER_MAX_HZ = 80.0  # full height of a steering bar
BAR_H = 0.9
SHOWN = [("idle", "idle"), ("random", "random"), ("reflex", "reflex (same eye, simple rule)"), ("fly", "fly (same eye, whole-brain model)"),
         ("fly_shuffled", "fly, shuffled wiring"), ("scripted", "scripted AI (sees more)")]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class FlyPlays(FlyDoomDraft):
    caption_y = -2.4

    def construct(self):
        self.load()
        self.clock = ValueTracker(0.0)  # seconds of footage shown so far
        self.speed = ValueTracker(1.0)  # footage seconds per video second
        self.beat_open()
        self.beat_loop()
        self.beat_react()
        self.beat_footage()
        self.beat_shuffle()
        self.beat_compare()

    # ---- data ------------------------------------------------------------------------------------

    def read_clips(self, manifest: str) -> list:
        return [read_json(A / "flydoom" / c) for c in read_json(A / "flydoom" / manifest)["clips"]]

    def load(self):
        self.fly_results, self.baselines = read_json(FLY_RESULTS), read_json(RESULTS)
        self.fly_clips, self.shuffle_clips = self.read_clips("fly_footage.json"), self.read_clips("shuffle_footage.json")
        self.clips = self.fly_clips
        self.index_clips()
        self.story = self.read_story(self.fly_clips[0])
        self.freeze_tic = self.story.onset - FREEZE_BEFORE
        self.shuffle_inputs, self.shuffle_steering = self.spike_totals(self.shuffle_clips[0])
        assert self.shuffle_steering == 0, "the shuffle beat says the steering neurons never fire; the data disagrees"

    def index_clips(self):
        super().index_clips()
        self.steer = np.array(sum((c["steer_hz"] for c in self.clips), []))

    @staticmethod
    def spike_totals(clip) -> tuple:
        sums = {k: sum(v) for k, v in clip["counts"].items()}
        return sum(v for k, v in sums.items() if k.startswith(("LC4", "LPLC2"))), sum(v for k, v in sums.items() if k.startswith("DNa"))

    @staticmethod
    def read_story(clip) -> SimpleNamespace:
        """The first threat of the first episode, step by step. The captions are built from this and the story is checked here."""
        seen, onset, side = set(), None, None
        for t, balls in enumerate(clip["fireballs"]):
            for oid, x, _ in balls:
                if onset is None and oid not in seen:
                    onset, side = t, "left" if x < 160 else "right"
                seen.add(oid)
        act = next(t for t in range(onset, clip["tics"]) if clip["actions"][t] != "none")
        steering = [clip["counts"][f"DNa01_{s}"][t] + clip["counts"][f"DNa02_{s}"][t] for t in range(clip["tics"]) for s in ("left", "right")]
        steering = np.array(steering).reshape(-1, 2)
        first_spike = next(t for t in range(onset, act + 1) if steering[t].sum() > 0)
        winner = "left" if clip["steer_hz"][act][0] > clip["steer_hz"][act][1] else "right"
        assert clip["counts"][f"LC4_{side}"][onset] > 0, "the input cells on the threat's side must fire when it appears"
        assert steering[first_spike].argmax() == ("left", "right").index(winner), "steering spikes must come from the winning side"
        assert winner != side and clip["actions"][act] == winner, "the story is: threat on one side, steering on the other, step to the steering side"
        return SimpleNamespace(onset=onset, side=side, first_spike=first_spike, act=act, winner=winner)

    def gtic(self) -> int:
        return int(np.clip(self.clock.get_value() * self.rate + 1e-6, 0, self.total_tics - 1))

    def start_clock(self):
        self.clock.add_updater(lambda m, dt: m.set_value(min(self.T, m.get_value() + dt * self.speed.get_value())))
        self.add(self.clock)

    def activity_paths(self, folder: str) -> list:
        return [ROOT / "media" / folder / f"{t:05d}.png" for t in range(self.total_tics)]

    def say(self, text: str, run_time: float = 0.6, extra=()):
        self.swap_caption(text, run_time=run_time, font_size=CAPTION_SIZE, extra=extra)

    def freeze(self, tic: int | None = None):
        self.speed.set_value(0)
        if tic is not None:
            self.clock.set_value(tic / self.rate)

    def play_to(self, tic: int, speed: float):
        """Play the footage at `speed` until global tic `tic`, then stay paused exactly on it."""
        self.speed.set_value(speed)
        self.wait(max((tic / self.rate - self.clock.get_value()) / speed, 0.02))
        self.freeze(tic)

    # ---- the chain: eye -> brain model -> steering -> keys -------------------------------------------

    def controls(self) -> VGroup:
        x0, y = GAME_INSET["center"][0] - 1.85, CONTROLS_Y

        def bars(x, get, vmax, colors):
            group = VGroup()
            for k, color in enumerate(colors):
                base = np.array([x + k * 0.32, y - 0.45, 0])
                group.add(DashedVMobject(Rectangle(width=0.2, height=BAR_H, color=color, stroke_width=1.5), num_dashes=14).move_to(base + UP * BAR_H / 2))

                def fill(base=base, k=k, color=color):
                    h = BAR_H * min(get(self.gtic(), k) / vmax, 1.0)
                    return Rectangle(width=0.2, height=max(h, 0.001), stroke_width=0, fill_color=color, fill_opacity=0.9).move_to(base + UP * h / 2)
                group.add(always_redraw(fill))
            return group

        eye = bars(x0, lambda g, k: self.eye[g, 2 + k], EXP_MAX, (LEFT_C, RIGHT_C))
        eye_icon = VGroup(Ellipse(width=0.42, height=0.24, color=TEXT, stroke_width=1.5), Dot(radius=0.06, color=TEXT)).next_to(eye, UP, buff=0.1)
        model = VGroup(RoundedRectangle(width=0.8, height=0.6, corner_radius=0.08, color=SIM_C, stroke_width=2), label("brain", 13, SIM_C)).move_to([x0 + 1.35, y, 0])
        steer = bars(x0 + 2.3, lambda g, k: self.steer[g, k], STEER_MAX_HZ, (OUT_C, OUT_C))
        keys = VGroup()
        for k, (name, color) in enumerate((("left", LEFT_C), ("right", RIGHT_C))):
            cap = RoundedRectangle(width=0.36, height=0.36, corner_radius=0.06, color=OUT_C, stroke_width=1.5).move_to([x0 + 3.35 + k * 0.45, y, 0])
            arrow = Triangle(color=OUT_C, fill_opacity=1, stroke_width=0).scale(0.08).rotate(PI / 2 if k == 0 else -PI / 2).move_to(cap)
            lit = always_redraw(lambda cap=cap, name=name, color=color: cap.copy().set_fill(color, 0.85 if self.actions[self.gtic()] == name else 0).set_stroke(width=0))
            keys.add(lit, cap, arrow)
        arrow_to = lambda a, b: Arrow([x0 + a, y, 0], [x0 + b, y, 0], buff=0, color=MUTED, stroke_width=2, max_tip_length_to_length_ratio=0.45)  # noqa: E731
        arrows = VGroup(arrow_to(0.5, 0.92), arrow_to(1.8, 2.12), arrow_to(2.78, 3.08))
        names = VGroup(*[label(t, 11, c).move_to([x, y - 0.78, 0]) for t, c, x in
                         (("eye", MUTED, x0 + 0.16), ("simulated", SIM_C, x0 + 1.35), ("decoder", MUTED, x0 + 2.46), ("keys", MUTED, x0 + 3.57))])
        self.parts = {"eye": VGroup(eye, eye_icon), "encoder": arrows[0], "model": model, "decoder": steer, "keys": keys}
        return VGroup(eye, eye_icon, model, steer, keys, arrows, names)

    def legend(self) -> VGroup:
        chip = lambda m, t: VGroup(m, label(t, 12, MUTED)).arrange(RIGHT, buff=0.1)  # noqa: E731
        rows = VGroup(
            chip(DashedVMobject(Rectangle(width=0.22, height=0.14, color=LEFT_C, stroke_width=1.5), num_dashes=8), "engineered: eye, decoder"),
            chip(RoundedRectangle(width=0.2, height=0.2, corner_radius=0.04, color=OUT_C, stroke_width=1.5), "action: keys"),
            chip(VGroup(Dot(radius=0.04, color=LEFT_C), Dot(radius=0.04, color=RIGHT_C).shift(RIGHT * 0.1)), "simulated: input neurons L / R"),
            chip(Dot(radius=0.06, color=OUT_C), "simulated: steering neurons"),
            chip(Dot(radius=0.03, color=SIM_C), "simulated: every other neuron"),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.07)
        return rows.to_corner(UR, buff=0.2).shift(DOWN * 0.5)

    def speed_chip(self):
        def text():
            s = self.speed.get_value()
            return label("paused" if s == 0 else f"playback ×{s:g}", 14, MUTED).to_corner(UR, buff=0.25)
        return always_redraw(text)

    # ---- beat 1: cold open, then the controls appear ------------------------------------------------

    def beat_open(self):
        self.beat("open")
        self.game_group, self.game = self.game_view()
        self.add(self.game_group)
        self.start_clock()
        self.wait(self.freeze_tic / self.rate)
        self.freeze(self.freeze_tic)
        c, k = GAME_INSET["center"], GAME_INSET["width"] / GAME_BIG["width"]
        self.ctrl, self.tl, self.leg, self.chip = self.controls(), self.timeline(), self.legend(), self.speed_chip()
        self.base = self.image(BRAIN_DIR / "back_all.png", BRAIN["width"], BRAIN["center"]).set_opacity(0.7)
        self.activity = self.sequence(self.activity_paths("flydoom_fly_activity"), self.gtic, BRAIN["width"], BRAIN["center"])
        self.play(self.game_group[0].animate.scale(k).move_to([c[0], c[1], 0]), self.game_group[1].animate.scale(k).move_to([c[0], c[1], 0]), run_time=T_SHRINK)
        self.play(FadeIn(self.base), FadeIn(self.ctrl), FadeIn(self.tl), FadeIn(self.leg), FadeIn(self.chip), run_time=1.0)
        self.keep_front()

    # ---- beat 2: the loop, frozen on one moment -------------------------------------------------------

    def beat_loop(self):
        self.beat("loop")
        p = self.parts
        stages = [
            ("The eye reads the screen's pixels: how fast each half is expanding.", p["eye"]),
            ("The left half drives the fly's left looming neurons, the right half the right ones.", p["encoder"]),
            ("The whole-brain model runs for one 1/35 s step: every neuron simulated.", p["model"]),
            ("The side whose steering neurons fire more wins, and the player steps that way.", VGroup(p["decoder"], p["keys"])),
        ]
        box = None
        for i, (text, part) in enumerate(stages):
            target = SurroundingRectangle(part, color=GOLD, buff=0.12, stroke_width=2.5)
            move = Create(target) if box is None else Transform(box, target)  # the box and its caption change together
            self.say(text, extra=[move])
            box = box or target
            if i == 2:
                self.play(FadeIn(self.activity), run_time=0.8)
                self.keep_front()
            self.wait(HOLD)
        self.play(FadeOut(box), run_time=0.4)

    # ---- beat 3: the first reaction, one 1/35 s step at a time ----------------------------------------

    def beat_react(self):
        self.beat("react")
        s = self.story
        self.say("Replay of the first threat, one step at a time.")
        self.play_to(s.onset, SLOW)
        self.say(f"A threat appears on the {s.side}: that eye's expansion jumps.")
        self.wait(HOLD)
        self.say(f"The {s.side} eye drives the {s.side} looming neurons, and they fire.")
        self.wait(HOLD)
        self.play_to(s.first_spike, SLOW)
        self.say(f"The signal crosses the brain: the steering neurons on the {s.winner} fire.")
        self.wait(HOLD)
        self.play_to(s.act, SLOW)
        self.say(f"The {s.winner} side wins, so the player steps {s.winner}: away from the threat.")
        self.wait(HOLD + 0.5)

    # ---- beat 4: the rest of the footage ----------------------------------------------------------------

    def beat_footage(self):
        self.beat("footage")
        self.say(f"The same fly, {len(self.fly_clips)} episodes in a row, at {FAST:g}× speed.")
        self.speed.set_value(FAST)
        self.wait((self.T - self.clock.get_value()) / FAST + 0.4)
        self.freeze(self.total_tics - 1)

    # ---- beat 5: the control: same seed, shuffled wiring --------------------------------------------------

    def beat_shuffle(self):
        self.beat("shuffle")
        self.play(FadeOut(self.game_group), FadeOut(self.tl), FadeOut(self.activity), run_time=0.6)
        self.clips = self.shuffle_clips
        self.index_clips()
        self.freeze(0)
        c, k = GAME_INSET["center"], GAME_INSET["width"] / GAME_BIG["width"]
        self.game_group, self.game = self.game_view()
        for m in self.game_group[:2]:
            m.scale(k).move_to([c[0], c[1], 0])
        self.tl = self.timeline()
        self.activity = self.sequence(self.activity_paths("flydoom_shuffle_activity"), self.gtic, BRAIN["width"], BRAIN["center"])
        self.play(FadeIn(self.game_group), FadeIn(self.tl), FadeIn(self.activity), run_time=0.8)
        self.keep_front()
        self.say("Control: the same seed and the same eye, but the wiring is shuffled.")
        self.speed.set_value(1.0)
        self.wait(1.6)
        self.say("Same neurons, same number of connections each; only who connects to whom is random.")
        self.wait(max(self.T - self.clock.get_value(), 0) + 0.2)
        self.freeze(self.total_tics - 1)
        self.say(f"The input cells still fire ({self.shuffle_inputs:,} spikes). The steering neurons never do. The player never moves.")
        self.wait(HOLD + 1.5)

    # ---- beat 6: how good is it -------------------------------------------------------------------------------

    def beat_compare(self):
        self.beat("compare")
        seeds = self.fly_results["test_seeds"]
        survival = {}
        for source in (self.baselines, self.fly_results):
            for e in source["episodes"]:
                if e["seed"] in seeds:
                    survival.setdefault(e["controller"], {})[e["seed"]] = e["survival_s"]
        fly_beats_reflex = sum(survival["fly"][s] > survival["reflex"][s] for s in seeds)
        scripted_beats_fly = sum(survival["scripted"][s] > survival["fly"][s] for s in seeds)

        x0, x1, smax = -1.4, 4.6, 22.0
        X = lambda s: x0 + (x1 - x0) * min(s, smax) / smax  # noqa: E731
        top = 2.2
        plot = VGroup(label(f"survival on {len(seeds)} distinct seeds · tick = median · band = middle 50 %", 13, MUTED).move_to([(x0 + x1) / 2, top + 0.7, 0]))
        for i, (key, name) in enumerate(SHOWN):
            y = top - i * 0.62
            v = np.array([survival[key][s] for s in seeds])
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            col = SIM_C if key.startswith("fly") else (MUTED if key in ("idle", "random") else TEXT)
            plot.add(label(name, 15, col).next_to([x0 - 0.2, y, 0], LEFT, buff=0))
            plot.add(Line([X(q1), y, 0], [X(q3), y, 0], color=col, stroke_width=5, stroke_opacity=0.35))
            plot.add(*[Dot([X(s), y + ((j % 3) - 1) * 0.07, 0], radius=0.032, color=col, fill_opacity=0.75) for j, s in enumerate(v)])
            plot.add(Line([X(med), y - 0.17, 0], [X(med), y + 0.17, 0], color=col, stroke_width=3))
            plot.add(label(f"{med:.1f} s", 14, col).next_to([x1 + 0.3, y, 0], RIGHT, buff=0))
        ya = top - len(SHOWN) * 0.62 + 0.25
        plot.add(Line([x0, ya, 0], [x1, ya, 0], color=MUTED, stroke_width=1.2))
        plot.add(*[label(f"{s}", 12, MUTED).next_to([X(s), ya, 0], DOWN, buff=0.06) for s in (0, 5, 10, 15, 20)])
        plot.add(label("s survived", 12, MUTED).next_to([x1, ya, 0], DOWN, buff=0.06).shift(RIGHT * 0.45))

        self.play(*[FadeOut(m) for m in (self.game_group, self.tl, self.activity, self.base, self.ctrl, self.leg, self.chip)], run_time=0.8)
        self.play(LaggedStart(*[FadeIn(m) for m in plot], lag_ratio=0.01), run_time=1.6)
        self.say(f"The fly beats the reflex on {fly_beats_reflex} of {len(seeds)} seeds. The scripted AI, which sees more, beats the fly on {scripted_beats_fly}.")
        self.wait(3.0)
        self.say("The wiring works as a pathway, but the model is not better than a simple reflex.")
        self.wait(4.0)
