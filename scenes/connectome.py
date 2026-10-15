"""Inside a real brain's wiring: the whole fruit-fly connectome, then its escape circuit (about 2 minutes).

    uv run python -m compute.connectome         # download ~77 MB once and stream the synapse file once (~20 s); export
    uv run python -m compute.fly_render         # draw the brain images (~1 minute) into media/fly_frames/
    uv run manim -ql scenes/connectome.py Connectome
    uv run python scenes/studio/inspect_video.py media/videos/connectome/480p15/Connectome.mp4 --per-beat 3

One beat (PowerShell):  $env:BEATS = "circuit"; uv run manim -ql scenes/connectome.py Connectome; Remove-Item Env:BEATS
Beats: few, whole, mapped, circuit, limits, recap.

MEASURED (FlyWire FAFB v783, CC BY 4.0): every neuron position, every connection count, every synapse dot.
  Positions are cell bodies (or, for neurons whose cell body is outside the brain, a point on the neuron).
  Straight lines between neurons are a simplification: real connections are made along winding branches.
SCHEMATIC (tagged): the drawing of the brain being sliced.
"""
import numpy as np
from manim import *

from studio.beats import BeatScene
from studio.fly import IMAGE_CENTER, IMAGE_WIDTH, FlyFrames, FlyTrace
from studio.theme import FONT, GOLD, MUTED, POS, TEXT

# ----------------------------------------------------------------------------------------------
# KNOBS. Positions are scene units: the screen is 14.22 wide and 8 high, (0, 0) is the center, y grows upward.
# ----------------------------------------------------------------------------------------------
FIRST_BOX = (11.5, 4.6)  # the first 12 neurons are magnified until they fill this width x height (scene units)...
FIRST_CENTER = np.array([0.0, 0.55, 0.0])  # ...centered here, clear of the caption
DOT_RADIUS = 0.035  # neuron dots, in whole-brain scene units (they grow with the zoom)
TURN_SECONDS = 10.0  # one full turn of the brain
CIRCUIT_TURN_SECONDS = 9.0
GF_COLOR = ManimColor("#f5c542")  # same gold as compute.fly_render.SYNAPSE_COLORS["GF"]
TYPE_COLOR = {"LPLC2": ManimColor("#3ad1c0"), "LC4": ManimColor("#ff7ab6")}
FOOTER_Y = -3.68
TAGS = {"MEASURED": POS, "SCHEMATIC": GOLD}


def label(text: str, size: float = 24, color=TEXT, **kw) -> Text:
    return Text(text, font=FONT, font_size=size, color=color, **kw)


def n(v: int) -> str:
    return f"{v:,}"


class Connectome(BeatScene):
    caption_y = -2.95

    def construct(self):
        self.trace = FlyTrace.load()
        self.frames = FlyFrames()
        self.tag = None
        self.foot = None
        self.beat_few()
        self.beat_whole()
        self.beat_mapped()
        self.beat_circuit()
        self.beat_limits()
        self.beat_recap()

    # ---- helpers -------------------------------------------------------------------------------

    def set_tag(self, kind):
        new = None
        if kind:
            t = label(kind, 18, TAGS[kind], weight=BOLD)
            new = VGroup(SurroundingRectangle(t, buff=0.09, corner_radius=0.06, color=TAGS[kind], stroke_width=1.5), t).to_corner(UR, buff=0.3)
        anims = ([FadeOut(self.tag)] if self.tag else []) + ([FadeIn(new)] if new else [])
        self.tag = new
        if anims:
            self.play(*anims, run_time=0.4)

    def set_footer(self, text):
        new = label(text, 15, MUTED).move_to([0, FOOTER_Y, 0]) if text else None
        anims = ([FadeOut(self.foot)] if self.foot else []) + ([FadeIn(new)] if new else [])
        self.foot = new
        if anims:
            self.play(*anims, run_time=0.4)

    def clear_stage(self, *keep, run_time=0.6):
        keep = set(keep) | {self.tag, self.foot, self.caption_now}
        gone = [m for m in self.mobjects if m not in keep]
        if gone:
            self.play(*[FadeOut(m) for m in gone], run_time=run_time)

    def scale_bar(self, microns: float, units_per_um: float, at) -> VGroup:
        line = Line(ORIGIN, RIGHT * microns * units_per_um, color=TEXT, stroke_width=3).move_to(at)
        return VGroup(line, label(f"{microns:g} µm", 17, MUTED).next_to(line, DOWN, buff=0.08))

    # ---- beat 1: a few real neurons -----------------------------------------------------------

    def beat_few(self):
        self.beat("few")
        tr, fr = self.trace, self.frames
        self.set_tag("MEASURED")
        d = tr["dataset"]
        self.set_footer(f"Data: {d['name']} · Dorkenwald et al. 2024, Schlegel et al. 2024 (Nature) · {d['license']}")
        group = tr["first_group"]
        ids = [g["id"] for g in group["neurons"]]
        pts = fr.to_scene([g["xyz"] for g in group["neurons"]])
        where = dict(zip(ids, pts))
        dots = VGroup(*[Dot(p, radius=DOT_RADIUS * (1.6 if i == 0 else 1), color=GF_COLOR if i == 0 else TEXT) for i, p in enumerate(pts)])
        max_syn = max(e["syn"] for e in group["edges"])
        edges = VGroup(*[Line(where[e["pre"]], where[e["post"]], color=MUTED, stroke_width=0.6 + 3.4 * np.log1p(e["syn"]) / np.log1p(max_syn),
                              stroke_opacity=0.75) for e in group["edges"]])
        self.brain = fr.image("front.png")
        self.vectors = VGroup(edges, dots)
        # the whole-brain view is saved; the pull-back returns here. (Image and vectors are restored separately:
        # Restore of a Group mixing an image and vector shapes fails in Manim 0.22.)
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        middle = (lo + hi) / 2
        zoom = min(FIRST_BOX[0] / max(hi[0] - lo[0], 1e-6), FIRST_BOX[1] / max(hi[1] - lo[1], 1e-6))
        for m in (self.brain, self.vectors):
            m.save_state()
            m.scale(zoom, about_point=middle).shift(FIRST_CENTER - middle)
        self.brain.set_opacity(0)
        self.add(self.brain)
        self.play(LaggedStart(*[GrowFromCenter(dt) for dt in dots], lag_ratio=0.08), run_time=1.6)
        self.swap_caption(f"{len(dots)} real neurons from one fruit fly's brain, each drawn where its cell body is.")
        self.wait(1.6)
        self.play(LaggedStart(*[Create(e) for e in edges], lag_ratio=0.02), run_time=2.0)
        self.swap_caption(f"{len(edges)} connections between them, each of at least 5 synapses. Thicker = more synapses.")
        self.wait(2.0)
        self.swap_caption("Lines are drawn straight; the real connections are made along long, branching fibres.")
        self.wait(2.0)
        self.play(Indicate(dots[0], color=GF_COLOR, scale_factor=1.6), run_time=1.0)
        self.swap_caption("The gold one is called the giant fiber. Keep an eye on it.")
        self.wait(1.6)
        self.dots, self.edges = dots, edges

    # ---- beat 2: pull back to the whole brain ----------------------------------------------------------

    def beat_whole(self):
        self.beat("whole")
        tr, fr = self.trace, self.frames
        tot = tr["totals"]
        self.play(Restore(self.brain), Restore(self.vectors), run_time=3.0)
        self.swap_caption(f"Pull back: every neuron in the brain, {n(tot['neurons'])} of them, each one dot.")
        self.wait(1.8)
        self.play(FadeOut(self.edges), FadeOut(self.dots), run_time=0.6)
        # the legend: classes and counts, colors from the renderer's manifest
        colors, names = fr.m["class_colors"], fr.m["class_names"]
        merged = {}
        for k, v in tot["classes"].items():
            merged.setdefault(names[k], [colors[k], 0])[1] += v
        rows = VGroup(*[VGroup(Dot(radius=0.07, color=c), label(f"{name}  {n(v)}", 18, MUTED)).arrange(RIGHT, buff=0.12)
                        for name, (c, v) in merged.items()]).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        rows.to_corner(UL, buff=0.35).shift(DOWN * 0.1)
        sides = VGroup(label("fly's right", 18, MUTED).move_to([-5.6, -2.25, 0]), label("fly's left", 18, MUTED).move_to([5.6, -2.25, 0]))
        view = label("seen from the front", 18, MUTED).move_to([0, -2.25, 0])
        bar = self.scale_bar(100, fr.scale, [4.6, 2.9, 0])
        self.play(FadeIn(rows), FadeIn(sides), FadeIn(view), FadeIn(bar), run_time=0.8)
        share = tot["classes"]["optic"] / tot["neurons"]
        self.swap_caption(f"More than half of them ({share:.0%}) are in the two optic lobes, which process what the eyes see.")
        self.wait(2.4)
        self.swap_caption(f"{n(tot['connections_ge5'])} neuron-to-neuron connections of at least 5 synapses, with {tot['synapses_in_those'] / 1e6:.0f} million synapses in them.")
        self.wait(2.4)
        # turntable
        angle = ValueTracker(0)
        spin = fr.sequence(fr.m["turntable"], fr.m["turntable_angles"], angle)
        self.play(FadeOut(sides), FadeOut(view), FadeOut(bar), run_time=0.4)
        self.add(spin)
        self.remove(self.brain)
        self.swap_caption("It is a real 3D object: turn it around.")
        self.play(angle.animate.set_value(360), run_time=TURN_SECONDS, rate_func=linear)
        spin.clear_updaters()
        self.add(self.brain)
        self.remove(spin)
        self.wait(0.6)
        self.legend = rows

    # ---- beat 3: how it was measured ----------------------------------------------------------------------

    def beat_mapped(self):
        self.beat("mapped")
        tr = self.trace
        z_um = tr["extent_um"]["z"][1] - tr["extent_um"]["z"][0]
        slices = int(round(z_um / 0.04, -3))
        self.play(FadeOut(self.legend), self.brain.animate.set_opacity(0.25), run_time=0.6)
        self.set_tag("SCHEMATIC")
        stack = VGroup(*[Polygon([-1.6, 0, 0], [0.6, 0, 0], [1.4, 0.7, 0], [-0.8, 0.7, 0], color=MUTED, stroke_width=1.5,
                                 fill_color="#1c2430", fill_opacity=0.9).shift(UP * 0.16 * i + RIGHT * 0.05 * i) for i in range(10)])
        stack.move_to([-3.6, 0.6, 0])
        dots_lbl = label("…", 30, MUTED).next_to(stack, UP, buff=0.05)
        text = VGroup(
            label(f"cut into about {n(slices)} slices,", 24),
            label("each 40 nanometres thick", 24),
            label("every slice imaged with an electron microscope", 22, MUTED),
            label("computers traced each neuron through the slices;", 22, MUTED),
            label("people checked and corrected the tracing", 22, MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12).move_to([2.2, 0.6, 0])
        self.play(LaggedStart(*[FadeIn(s, shift=UP * 0.05) for s in stack], lag_ratio=0.08), FadeIn(dots_lbl), run_time=1.2)
        self.play(FadeIn(text[:3]), run_time=0.6)
        self.swap_caption(f"How do we know? The brain was sliced: its depth of {z_um:.0f} µm divided into 40 nm sections (Zheng et al. 2018).")
        self.wait(2.4)
        self.play(FadeIn(text[3:]), run_time=0.6)
        self.swap_caption("From the images, the FlyWire team reconstructed every neuron and its chemical synapses (Dorkenwald et al. 2024).")
        self.wait(2.6)
        self.play(FadeOut(stack), FadeOut(dots_lbl), FadeOut(text), self.brain.animate.set_opacity(1), run_time=0.6)
        self.set_tag("MEASURED")

    # ---- beat 4: the escape circuit --------------------------------------------------------------------------

    def beat_circuit(self):
        self.beat("circuit")
        tr, fr = self.trace, self.frames
        f = tr["focus"]
        gf_pts = fr.to_scene([f["xyz"], f["other_side"]["xyz"]])
        stars = VGroup(*[Star(5, outer_radius=0.13, color=GF_COLOR, fill_opacity=1, stroke_width=0).move_to(p) for p in gf_pts])
        self.play(LaggedStart(*[GrowFromCenter(s) for s in stars], lag_ratio=0.3), self.brain.animate.set_opacity(0.45), run_time=0.9)
        self.swap_caption("Back to the giant fiber. There is one on each side of the brain.")
        self.wait(1.6)
        # zoom: magnify the whole-brain view so the zoom window fills the image area, then swap in the sharp images
        cx, cy, hw = fr.zoom_window
        pivot = fr.um_to_scene(cx, cy)
        k = fr.zoom_factor()
        self.play(FadeOut(stars), run_time=0.4)
        self.play(self.brain.animate.scale(k, about_point=pivot).shift(IMAGE_CENTER - pivot), run_time=2.4)
        sharp = {name: fr.image(f"zoom/{name}.png") for name in ("brain_dim", "GF", "LPLC2", "LC4")}
        self.play(FadeIn(sharp["brain_dim"]), FadeOut(self.brain), run_time=0.6)
        bar = self.scale_bar(20, fr.scale * k, [5.4, 2.9, 0])
        self.play(FadeIn(bar), run_time=0.3)
        ext = tr["synapse_extract"]
        self.play(FadeIn(sharp["GF"]), run_time=1.2)
        self.swap_caption(f"Each gold dot is one synapse onto the giant fiber: {n(ext['focus_inputs'])} from {f['n_inputs']} neurons. Together they outline its branches.")
        self.wait(2.8)
        types = {t["type"]: t for t in f["input_types"]}
        for typ, what, cite in (("LPLC2", "respond selectively to an object looming toward the eye", "Klapoetke et al. 2017"),
                                ("LC4", "signal how fast the looming object approaches", "Ache et al. 2019")):
            t = types[typ]
            self.play(FadeIn(sharp[typ]), run_time=1.0)
            key = VGroup(Dot(radius=0.08, color=TYPE_COLOR[typ]), label(f"{typ}: {t['neurons']} neurons, {n(t['synapses'])} synapses onto the giant fiber", 19, TYPE_COLOR[typ])).arrange(RIGHT, buff=0.15)
            key.to_corner(UL, buff=0.4).shift(DOWN * (0.0 if typ == "LPLC2" else 0.4))
            self.play(FadeIn(key), run_time=0.4)
            share = t["synapses"] / ext["looming_outputs"][typ]
            self.swap_caption(f"{typ} neurons {what} ({cite}).")
            self.wait(2.2)
            self.swap_caption("Their output synapses crowd into one hub, right where the giant fiber's branch reaches in.")
            self.wait(2.2)
            self.swap_caption(f"The hub contacts many neurons: only {share:.0%} of these {n(ext['looming_outputs'][typ])} {typ} synapses land on the giant fiber itself.")
            self.wait(2.8)
        top2 = [t["type"] for t in f["input_types"][:2]]
        self.swap_caption(f"Of its {f['n_inputs']} input neurons, these two looming-detector types give it the most synapses: {top2[0]}, then {top2[1]}.")
        self.wait(2.6)
        # turn the circuit around
        angle = ValueTracker(0)
        spin = fr.sequence(fr.m["circuit_turntable"]["files"], fr.m["circuit_turntable"]["angles"], angle)
        self.play(FadeOut(bar), FadeOut(Group(*sharp.values())), FadeIn(spin), run_time=0.6)
        self.swap_caption("Turn it around: the contact zones sit where the giant fiber's side branch reaches toward the optic lobe.")
        self.play(angle.animate.set_value(360), run_time=CIRCUIT_TURN_SECONDS, rate_func=linear)
        spin.clear_updaters()
        self.swap_caption("Its trunk leaves the brain for the nerve cord, where it triggers the jump: a fast take-off (von Reyn et al. 2014).")
        self.wait(3.0)
        self.swap_caption("The nerve cord is not part of this brain map, so that last step is from other studies, not from these data.")
        self.wait(2.4)
        self.clear_stage()
        self.set_footer(None)

    # ---- beat 5: what a wiring diagram does and does not tell ---------------------------------------------------

    def beat_limits(self):
        self.beat("limits")
        tr = self.trace
        f = tr["focus"]
        a, b = f, f["other_side"]
        shows = VGroup(label("What this map shows", 26, POS),
                       label("• which neuron connects to which, for every neuron in one brain", 20),
                       label("• where on the neurons the synapses are, and how many", 20)).arrange(DOWN, aligned_edge=LEFT, buff=0.14)
        cannot = VGroup(label("What it does not show", 26, GOLD),
                        label("• activity: when neurons fire, or how strong a synapse is", 20),
                        label("• electrical synapses (gap junctions): only chemical synapses are mapped", 20),
                        label("• how other flies differ: this is one adult female fly", 20),
                        label(f"• transmitters, only predicted: the two giant fibers got {a['predicted_nt']} ({a['predicted_nt_confidence']:.2f})", 20),
                        label(f"   and {b['predicted_nt']} ({b['predicted_nt_confidence']:.2f}), both with low confidence", 20)).arrange(DOWN, aligned_edge=LEFT, buff=0.14)
        VGroup(shows, cannot).arrange(DOWN, aligned_edge=LEFT, buff=0.45).move_to([0, 0.5, 0])
        self.play(FadeIn(shows, shift=UP * 0.1), run_time=0.7)
        self.swap_caption("A connectome is a map of the wires, measured once, in one brain.")
        self.wait(2.2)
        self.play(FadeIn(cannot, shift=UP * 0.1), run_time=0.7)
        self.swap_caption("To learn what the wires do, it is combined with recordings and behaviour experiments, like the ones cited here.")
        self.wait(3.4)
        self.clear_stage()

    # ---- beat 6: recap ----------------------------------------------------------------------------------------------

    def beat_recap(self):
        self.beat("recap")
        tr = self.trace
        if self.caption_now:
            self.play(FadeOut(self.caption_now), run_time=0.3)
            self.caption_now = None
        self.set_tag(None)
        f = tr["focus"]
        types = {t["type"]: t for t in f["input_types"]}
        lines = VGroup(
            label("A fly's “react to a punch” circuit, in real wiring", 30),
            label("eye  →  looming detectors (LPLC2, LC4)  →  giant fiber  →  jump", 24, GF_COLOR),
            label(f"{types['LPLC2']['neurons'] + types['LC4']['neurons']} detector neurons, {n(types['LPLC2']['synapses'] + types['LC4']['synapses'])} synapses onto one giant fiber,", 21, MUTED),
            label(f"inside a brain of {n(tr['totals']['neurons'])} neurons.", 21, MUTED),
        ).arrange(DOWN, buff=0.3).move_to([0, 0.5, 0])
        refs = label("Klapoetke et al. 2017 · Ache et al. 2019 · von Reyn et al. 2014 · Zheng et al. 2018 · Dorkenwald et al. 2024 · Schlegel et al. 2024", 15, MUTED).next_to(lines, DOWN, buff=0.6)
        self.play(LaggedStart(*[FadeIn(l, shift=UP * 0.1) for l in lines], lag_ratio=0.4), run_time=2.4)
        self.play(FadeIn(refs), run_time=0.5)
        self.wait(4.0)
