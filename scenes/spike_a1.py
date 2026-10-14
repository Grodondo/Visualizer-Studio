"""A1 spikes: prove that the Typst-only workflow supports what the first video needs.

Probes (render a still with -s):
    ProbeTypstDecimal, ProbeTextDecimal   changing numeric labels without LaTeX
Full spike (render a short video, then inspect it):
    A1Spike   labels, partial equation highlighting, tanh curve, camera move, beats

    uv run manim -ql -s scenes/spike_a1.py ProbeTypstDecimal
    uv run manim -ql scenes/spike_a1.py A1Spike
    uv run python scenes/studio/inspect_video.py media/videos/spike_a1/480p15/A1Spike.mp4
"""
from manim import *

from studio.beats import BeatScene
from studio.theme import BG, FOCUS, FONT, GOLD, MUTED, NEG, POS, TEXT, sign_color


class ProbeTypstDecimal(Scene):
    """DecimalNumber defaults to LaTeX (MathTex); does it accept MathTypst instead?"""

    def construct(self):
        self.camera.background_color = BG
        a = DecimalNumber(2.4, num_decimal_places=2, mob_class=MathTypst, font_size=72, color=POS)
        b = DecimalNumber(-0.92, num_decimal_places=2, mob_class=MathTypst, font_size=72, color=NEG)
        VGroup(a, b).arrange(RIGHT, buff=1.5)
        self.add(a, b)
        a.set_value(0.98)  # changing the value is the point of DecimalNumber
        self.add(Text("MathTypst digits", font=FONT, font_size=20, color=MUTED).to_edge(DOWN))


class ProbeTextDecimal(Scene):
    def construct(self):
        self.camera.background_color = BG
        a = DecimalNumber(2.4, num_decimal_places=2, mob_class=Text, font_size=72, color=POS)
        b = DecimalNumber(-0.92, num_decimal_places=2, mob_class=Text, font_size=72, color=NEG)
        VGroup(a, b).arrange(RIGHT, buff=1.5)
        self.add(a, b)
        a.set_value(0.98)
        self.add(Text("Text digits", font=FONT, font_size=20, color=MUTED).to_edge(DOWN))


def caption(text: str) -> Text:
    return Text(text, font=FONT, font_size=26, color=MUTED).to_edge(DOWN, buff=0.5)


def rest_of(eq: MathTypst, labels: list[str]) -> VGroup:
    """The glyphs of an equation that are not part of any named group (symbols like z, =, +)."""
    grouped = [m for label in labels for m in eq.select(label).submobjects]
    return VGroup(*[m for m in eq.submobjects if not any(m is g for g in grouped)])


def morph(scene: Scene, pairs: list[tuple[VGroup, VGroup]], run_time: float = 1.0) -> None:
    """Turn each source group into its target group.

    ReplacementTransform (not Transform) matters here: afterwards the scene holds the *target's* glyphs,
    so the next step can select() from the target again. Plain Transform leaves stray glyphs when a
    group changes its glyph count (e.g. `b` becoming `- 2.0`).
    """
    scene.play(*[ReplacementTransform(src, dst) for src, dst in pairs], run_time=run_time)


class A1Spike(BeatScene):
    def construct(self):
        self.beat_labels()
        self.beat_equation()
        self.beat_tanh()
        self.beat_camera()

    def clear_beat(self):
        for m in self.mobjects:
            m.clear_updaters()
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.4)

    # 1. A number that changes, inside a neuron whose fill follows its sign.
    def beat_labels(self):
        self.beat("labels")
        value = ValueTracker(0.0)
        circle = Circle(radius=1.1, color=MUTED, stroke_width=3)
        number = DecimalNumber(0, num_decimal_places=2, mob_class=MathTypst, font_size=64)

        def refresh(_mob):
            v = value.get_value()
            colour = sign_color(v)
            # Tint stays light so the number, drawn in the same sign color, remains readable on top of it.
            circle.set_stroke(colour).set_fill(colour, opacity=0.3 * min(1, abs(v)))
            number.set_value(v).set_color(colour).move_to(circle)

        number.add_updater(refresh)
        cap = caption("Number labels change with the value; color follows the sign")
        self.play(FadeIn(circle), FadeIn(number), FadeIn(cap))
        self.play(value.animate.set_value(0.98), run_time=1.5)
        self.wait(0.5)
        self.play(value.animate.set_value(-0.92), run_time=1.5)
        self.wait(0.5)
        self.clear_beat()

    # 2. Partial highlighting and rearranging an equation.
    def beat_equation(self):
        self.beat("equation")
        labels = ["w1", "x1", "w2", "x2", "b"]
        sym = MathTypst("z = {{ w_1 : w1 }} {{ x_1 : x1 }} + {{ w_2 : w2 }} {{ x_2 : x2 }} + {{ b : b }}", font_size=64, color=TEXT)
        sub = MathTypst("z = {{ 4.0 : w1 }} dot {{ 0.8 : x1 }} + {{ 4.0 : w2 }} dot {{ 0.3 : x2 }} {{ - 2.0 : b }}", font_size=64, color=TEXT)
        prod = MathTypst("z = {{ 3.2 : p1 }} + {{ 1.2 : p2 }} {{ - 2.0 : b }}", font_size=64, color=TEXT)
        for eq in (sym, sub):
            for label, value in zip(labels, [4.0, 0.8, 4.0, 0.3, -2.0]):
                eq.select(label).set_color(sign_color(value))
        prod.select("p1").set_color(POS)
        prod.select("p2").set_color(POS)
        prod.select("b").set_color(NEG)

        self.play(Write(sym))
        box = SurroundingRectangle(sym.select("w1"), color=GOLD, buff=0.1)
        self.play(Create(box))
        self.play(Transform(box, SurroundingRectangle(sym.select("x1"), color=GOLD, buff=0.1)))
        self.play(FadeOut(box))
        # Substitute numbers: each named part transforms into its numeric version.
        morph(self, [(sym.select(l), sub.select(l)) for l in labels] + [(rest_of(sym, labels), rest_of(sub, labels))])
        self.wait(0.4)
        # Combine two parts into one: w1 and x1 become the product 3.2.
        prod_labels = ["p1", "p2", "b"]
        morph(
            self,
            [
                (VGroup(*sub.select("w1"), *sub.select("x1")), prod.select("p1")),
                (VGroup(*sub.select("w2"), *sub.select("x2")), prod.select("p2")),
                (sub.select("b"), prod.select("b")),
                (rest_of(sub, labels), rest_of(prod, prod_labels)),
            ],
            run_time=1.5,
        )
        self.wait(0.6)
        self.add(caption("Parts keep their sign color while they transform"))
        self.wait(0.6)
        self.clear_beat()

    # 3. The tanh curve with a moving point, built without LaTeX-backed axis numbers.
    def beat_tanh(self):
        self.beat("tanh")
        axes = Axes(
            x_range=[-4, 4, 1],
            y_range=[-1.25, 1.25, 0.5],
            x_length=7,
            y_length=4,
            axis_config={"color": MUTED, "include_tip": False, "include_numbers": False},
        ).shift(DOWN * 0.2)
        curve = axes.plot(np.tanh, x_range=[-4, 4], color=TEXT, stroke_width=4)
        ticks = VGroup(
            *[MathTypst(s, font_size=30, color=MUTED).next_to(axes.c2p(x, 0), DOWN, buff=0.15) for s, x in (("-4", -4), ("0", 0), ("4", 4))],
            *[MathTypst(s, font_size=30, color=MUTED).next_to(axes.c2p(0, y), LEFT, buff=0.15) for s, y in (("-1", -1), ("1", 1))],
        )
        z = 2.4
        a = float(np.tanh(z))
        z_dot = Dot(axes.c2p(z, 0), color=sign_color(z))
        a_dot = Dot(axes.c2p(z, a), color=sign_color(a))
        v_line = DashedLine(axes.c2p(z, 0), axes.c2p(z, a), color=GOLD)
        h_line = DashedLine(axes.c2p(z, a), axes.c2p(0, a), color=GOLD)
        z_label = DecimalNumber(z, num_decimal_places=2, mob_class=MathTypst, font_size=36, color=sign_color(z)).next_to(z_dot, DOWN, buff=0.45)
        a_label = DecimalNumber(a, num_decimal_places=2, mob_class=MathTypst, font_size=36, color=sign_color(a)).next_to(axes.c2p(0, a), LEFT, buff=0.5)

        self.play(Create(axes), FadeIn(ticks))
        self.play(Create(curve), run_time=1.2)
        self.play(FadeIn(z_dot), FadeIn(z_label))
        self.play(Create(v_line), FadeIn(a_dot))
        self.play(Create(h_line), FadeIn(a_label))
        self.add(caption("tanh curve, axes and number labels without LaTeX"))
        self.wait(0.8)
        self.clear_beat()

    # 4. Camera move: zoom into one neuron, dim the rest, then restore.
    def beat_camera(self):
        self.beat("camera")
        self.camera.frame.save_state()
        inputs = VGroup(*[Circle(radius=0.35, color=MUTED, stroke_width=3) for _ in range(2)]).arrange(DOWN, buff=1.2).shift(LEFT * 4)
        hidden = VGroup(*[Circle(radius=0.35, color=MUTED, stroke_width=3) for _ in range(4)]).arrange(DOWN, buff=0.6)
        output = VGroup(*[Circle(radius=0.35, color=MUTED, stroke_width=3) for _ in range(2)]).arrange(DOWN, buff=1.2).shift(RIGHT * 4)
        edges = VGroup(
            *[Line(i.get_right(), h.get_left(), color=POS, stroke_width=2, stroke_opacity=0.6) for i in inputs for h in hidden],
            *[Line(h.get_right(), o.get_left(), color=NEG, stroke_width=2, stroke_opacity=0.6) for h in hidden for o in output],
        )
        target = hidden[0]
        target.set_color(FOCUS)
        label = MathTypst("z = 2.4", font_size=40, color=POS).next_to(target, UP, buff=0.3)
        scene_group = VGroup(edges, inputs, hidden, output)
        self.play(FadeIn(scene_group))
        self.wait(0.3)
        others = VGroup(edges, inputs, output, *hidden[1:])
        others.save_state()  # dimming with set_opacity(1) later would also fill the empty circles
        self.play(self.camera.frame.animate.move_to(target).set(width=5.5), others.animate.set_opacity(0.15), run_time=1.6)
        self.play(FadeIn(label))
        self.wait(0.6)
        self.play(Restore(self.camera.frame), Restore(others), FadeOut(label), run_time=1.4)
        self.wait(0.4)
