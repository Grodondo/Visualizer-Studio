"""How does a neural network read a handwritten digit?  A real network, trained on real digits (about 95 seconds).

    uv run python -m compute.digits                                  # download the small dataset once, train, write the trace
    uv run manim -ql scenes/digit_network.py DigitRecognition        # draft, 480p15
    uv run manim -qm scenes/digit_network.py DigitRecognition        # draft, 720p30
    uv run python scenes/studio/inspect_video.py media/videos/digit_network/480p15/DigitRecognition.mp4

Render one beat while working on it (PowerShell):
    $env:BEATS = "activation-functions"; uv run manim -ql scenes/digit_network.py DigitRecognition; Remove-Item Env:BEATS

Beats: digit, network, one-neuron, what-neurons-learn, activation-functions, forward-pass, answer, mistake, learning.
Everything on screen comes from assets/digits/digits_trace.json: real digits (UCI, Alpaydin & Kaynak 1998, CC BY 4.0),
weights learned by training (compute/digits.py), and the forward pass of those weights. Nothing is drawn by hand.
Colors: blue = positive number, orange = negative, gold = look here, violet = the neuron being studied.
"""
from pathlib import Path

import numpy as np
from manim import *

from studio.beats import BeatScene
from studio.digit_network import (
    BAR_MAX,
    BAR_X,
    CENTER_Y,
    GRID_CELL,
    GRID_CENTER,
    DigitNetwork,
    PixelGrid,
    ProbabilityBars,
    weight_gallery,
)
from studio.digits_trace import INK_MAX, DigitsTrace
from studio.equations import typst_number
from studio.plots import CURVE_COLORS, CURVE_NAMES, ActivationComparison
from studio.theme import BG, FOCUS, FONT, GOLD, MUTED, NEG, POS, TEXT, sign_color
from studio.trace import probability_decimals, probability_typst

TRACE_PATH = Path(__file__).resolve().parent.parent / "assets" / "digits" / "digits_trace.json"

# ----------------------------------------------------------------------------------------------
# KNOBS. Positions are scene units: the screen is 14.22 wide and 8 high, (0, 0) is the center, y grows upward.
# ----------------------------------------------------------------------------------------------
BIG_CELL = 0.5  # size of one pixel when the digit is shown large (beat 1)
BIG_CENTER = (0, 0.3)
PANEL_CELL = 0.26  # size of one pixel in the three panels of beat 3
PANEL_Y = 1.4
PANEL_X = (-4.6, -0.9, 2.8)  # pixels x weights = products
SUM_ROW_Y = -1.0
DIM_AMOUNT = 0.88  # how much the rest of the network fades while one neuron is studied
PLOT_CENTER = (-3.0, 0.2)  # the activation-function graph (beat 5)
TABLE_X = (4.1, 6.0)  # centers of the two columns of the activation table
NAME_X = 0.7  # left edge of the row names (sigmoid(z), tanh(z), ReLU(z))
HEAD_Y = 2.2  # header (neuron number and its z)
ROW_Y = {"sigmoid": 1.2, "tanh": 0.4, "relu": -0.4}  # one table row per function
CHART_Y = 0.35  # the two training charts (beat 9)


def saturating(v: float) -> str:
    """Typst for a sigmoid/tanh output: enough decimals that 0.99996 is not shown as 1.00."""
    decimals = probability_decimals(min(abs(v), 1.0))
    shown = f"{abs(v):.{decimals}f}"
    return ("- " if v < 0 and float(shown) != 0 else "") + shown


class DigitRecognition(BeatScene):
    def construct(self):
        self.trace = DigitsTrace.load(TRACE_PATH)
        self.example = self.trace.showcase
        self.focus = self.trace.focus.neuron
        self.beat_digit()
        self.beat_network()
        self.beat_one_neuron()
        self.beat_gallery()
        self.beat_activation_functions()
        self.beat_forward_pass()
        self.beat_answer()
        self.beat_mistake()
        self.beat_learning()

    # ---- small helpers -------------------------------------------------------------------------

    def footer(self) -> VGroup:
        d = self.trace.dataset
        lines = [
            f"Real data: handwritten digits from the UCI repository (Alpaydin & Kaynak, 1998, {d['license']}).",
            f"The weights were learned from {d['n_train']:,} of them. Values rounded for display; probabilities show more digits.",
        ]
        return VGroup(*[Text(t, font=FONT, font_size=17, color=MUTED) for t in lines]).arrange(DOWN, buff=0.08).move_to([0, -3.62, 0])

    def new_network(self, example=None) -> DigitNetwork:
        return DigitNetwork(self.trace, example or self.example)

    # ---- beat 1: a digit is a grid of numbers ------------------------------------------------

    def beat_digit(self):
        self.beat("digit")
        ex = self.example
        title = Text("How does a network read a handwritten digit?", font=FONT, font_size=38, color=TEXT).to_edge(UP, buff=0.5)
        self.big = PixelGrid(ex.pixels, cell=BIG_CELL, vmax=INK_MAX).move_to([BIG_CENTER[0], BIG_CENTER[1], 0])
        numbers = self.big.number_labels(ex.pixels, font_size=22)
        self.foot = self.footer()
        self.play(Write(title), FadeIn(self.foot), run_time=1.2)
        self.play(FadeIn(self.big, scale=0.9), run_time=1.0)
        self.swap_caption("A handwritten digit, stored as a tiny 8 \u00D7 8 picture.")
        self.wait(0.8)
        self.play(FadeIn(numbers), run_time=0.8)
        self.swap_caption("Each pixel is a number: the amount of ink in it, from 0 to 16.")
        self.wait(1.8)
        self.play(FadeOut(numbers), FadeOut(title), run_time=0.6)
        self.swap_caption("The network receives these 64 numbers, scaled to between 0 and 1.")
        self.wait(1.2)

    # ---- beat 2: the network ----------------------------------------------------------------

    def beat_network(self):
        self.beat("network")
        self.net = self.new_network()
        n = self.net
        self.play(ReplacementTransform(self.big, n.grid), FadeIn(n.titles[0]), run_time=1.0)
        self.play(LaggedStart(*[GrowFromCenter(r) for r in n.hidden], lag_ratio=0.06), FadeIn(n.titles[1]), run_time=1.0)
        self.play(LaggedStart(*[Create(f) for f in n.fans], lag_ratio=0.05), run_time=0.8)
        note = Text("each line stands for 64 weights", font=FONT, font_size=18, color=MUTED).move_to([-2.75, -1.75, 0])
        self.play(FadeIn(note), run_time=0.4)
        self.play(LaggedStart(*[GrowFromCenter(r) for r in n.outputs], lag_ratio=0.06), *[FadeIn(l) for l in n.digit_labels], FadeIn(n.titles[2]), run_time=0.9)
        self.play(LaggedStart(*[Create(e) for e in n.out_edges.values()], lag_ratio=0.004), run_time=1.3)
        self.swap_caption("16 hidden neurons look for patterns; 10 output neurons stand for 0 to 9.")
        self.wait(2.0)
        self.play(FadeOut(note), run_time=0.3)

    # ---- beat 3: one neuron = a weight for every pixel ----------------------------------------

    def beat_one_neuron(self):
        self.beat("one-neuron")
        t, ex, f, n = self.trace, self.example, self.focus, self.net
        weights = t.weight_image(f)
        wmax = max(abs(w) for w in weights)

        # The 64 real weights of one neuron, drawn as 64 lines from the pixels to it.
        lines = VGroup(
            *[
                Line(n.grid.cells[k].get_center(), n.hidden[f].get_left(), stroke_color=sign_color(w), stroke_width=0.6 + 3.0 * abs(w) / wmax, stroke_opacity=0.85)
                for k, w in enumerate(weights)
            ]
        )
        halo = Circle(radius=0.2, color=FOCUS, stroke_width=4).move_to(n.hidden[f])
        rest = VGroup(*[r for i, r in enumerate(n.hidden) if i != f], *[l for i, l in enumerate(n.fans) if i != f], *n.out_edges.values(), *n.outputs, *n.digit_labels, n.titles[1], n.titles[2])
        self.play(rest.animate.fade(DIM_AMOUNT), Create(halo), run_time=0.8)
        self.add(lines)
        self.bring_to_front(n.grid)
        self.play(LaggedStart(*[Create(l) for l in lines], lag_ratio=0.012), run_time=1.4)
        self.swap_caption("One hidden neuron has its own weight for every pixel: 64 of them.")
        self.wait(1.6)

        # Unroll the picture: pixels x weights = what each pixel adds.
        pixels = PixelGrid(ex.pixels, cell=PANEL_CELL, vmax=INK_MAX).move_to([PANEL_X[0], PANEL_Y, 0])
        wmap = PixelGrid(weights, cell=PANEL_CELL, mode="signed", vmax=wmax).move_to([PANEL_X[1], PANEL_Y, 0])
        prods = t.products(f, ex)
        pmap = PixelGrid(prods, cell=PANEL_CELL, mode="signed", vmax=max(abs(p) for p in prods)).move_to([PANEL_X[2], PANEL_Y, 0])
        ops = VGroup(
            Text("\u00D7", font=FONT, font_size=44, color=TEXT).move_to([(PANEL_X[0] + PANEL_X[1]) / 2, PANEL_Y, 0]),
            Text("=", font=FONT, font_size=44, color=TEXT).move_to([(PANEL_X[1] + PANEL_X[2]) / 2, PANEL_Y, 0]),
        )

        def label(text, x):
            return Text(text, font=FONT, font_size=20, color=MUTED).move_to([x, PANEL_Y - 1.3, 0])

        labels = VGroup(label("pixels x", PANEL_X[0]), label(f"weights w of neuron {f}", PANEL_X[1]), label("w \u00D7 x: what each pixel adds", PANEL_X[2]))
        self.play(FadeOut(lines), FadeOut(rest), FadeOut(halo), FadeOut(n.hidden[f]), FadeOut(n.fans[f]), FadeOut(n.titles[0]), TransformFromCopy(n.grid, pixels), FadeOut(n.grid), run_time=1.0)
        self.play(FadeIn(labels[0]), run_time=0.3)
        self.play(FadeIn(wmap), FadeIn(labels[1]), FadeIn(ops[0]), run_time=0.9)
        self.swap_caption("The weights as a picture: blue where ink raises the score, orange where it lowers it.")
        self.wait(2.0)
        self.play(FadeIn(pmap), FadeIn(labels[2]), FadeIn(ops[1]), run_time=0.9)
        self.swap_caption("Multiply every pixel by its weight, then add all 64 results and the neuron's bias.")
        self.wait(1.0)

        total, bias, z = t.weighted_sum(f, ex), t.b1[f], ex.z1[f]
        row = MathTypst(
            "z = {{ " + typst_number(total) + " : s }} + {{ " + typst_number(bias) + " : b }} = {{ " + typst_number(z) + " : z }}",
            font_size=44,
            color=TEXT,
        ).move_to([0, SUM_ROW_Y, 0])
        for name, value in (("s", total), ("b", bias), ("z", z)):
            row.select(name).set_color(sign_color(value))
        # The two labels sit on opposite sides of the row so they cannot collide.
        words = Text("sum of the 64 results", font=FONT, font_size=20, color=MUTED).next_to(row.select("s"), UP, buff=0.2)
        bias_word = Text("bias", font=FONT, font_size=20, color=MUTED).next_to(row.select("b"), DOWN, buff=0.2)
        self.play(Write(row), run_time=1.2)
        self.play(FadeIn(words), FadeIn(bias_word), run_time=0.5)
        self.wait(1.2)
        self.neuron_panels = VGroup(pixels, wmap, pmap, ops, labels, row, words, bias_word)

    # ---- beat 4: all the neurons ------------------------------------------------------------

    def beat_gallery(self):
        self.beat("what-neurons-learn")
        t, f = self.trace, self.focus
        maps = weight_gallery(t)
        gallery = VGroup(*maps).move_to([-2.6, 0.6, 0])
        frame = SurroundingRectangle(maps[f], color=FOCUS, buff=0.1, stroke_width=4)
        key = VGroup(
            VGroup(Dot(radius=0.08, color=POS), Text("ink raises the score", font=FONT, font_size=22, color=TEXT)).arrange(RIGHT, buff=0.15),
            VGroup(Dot(radius=0.08, color=NEG), Text("ink lowers the score", font=FONT, font_size=22, color=TEXT)).arrange(RIGHT, buff=0.15),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        side = VGroup(
            Text("All 16 hidden neurons", font=FONT, font_size=30, color=TEXT),
            Text("Each has its own weight picture.", font=FONT, font_size=24, color=MUTED),
            Text("Nobody designed these:\nthey were learned from examples.", font=FONT, font_size=24, color=MUTED, line_spacing=0.9),
            key,
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to([4.2, 0.4, 0])
        self.play(FadeOut(self.neuron_panels), run_time=0.6)
        self.play(LaggedStart(*[FadeIn(m) for m in maps], lag_ratio=0.06), run_time=1.8)
        self.play(Create(frame), FadeIn(side), run_time=0.9)
        self.swap_caption(f"The neuron we just studied is number {f}, marked in violet.")
        self.wait(2.6)
        self.gallery_group = VGroup(gallery, frame, side)

    # ---- beat 5: activation functions -------------------------------------------------------

    def beat_activation_functions(self):
        self.beat("activation-functions")
        t = self.trace
        plot = ActivationComparison(t.curves).move_to([PLOT_CENTER[0], PLOT_CENTER[1], 0])
        names = VGroup(plot.curve_label("relu", 3.9), plot.curve_label("sigmoid", 3.2), plot.curve_label("tanh", -4.3))
        names[2].next_to(plot.axes.c2p(-4.3, -1.0), DOWN, buff=0.1)
        self.play(FadeOut(self.gallery_group), run_time=0.6)
        self.play(FadeIn(plot.axes), FadeIn(plot.ticks), run_time=0.6)
        self.play(LaggedStart(*[Create(c) for c in plot.curves.values()], lag_ratio=0.4), FadeIn(names), run_time=2.2)
        self.swap_caption("Before a neuron passes its sum z on, an activation function reshapes it.")
        self.wait(1.2)

        def column(neuron, x):
            """Readings for one neuron: the gold line and dots on the graph, plus its header and three table cells at fixed positions."""
            z, values = neuron.z, neuron.values
            line, dots = plot.reading(z, values)
            head = VGroup(
                Text(f"neuron {neuron.neuron}", font=FONT, font_size=20, color=MUTED),
                MathTypst("z = " + typst_number(z), font_size=30, color=sign_color(z)),
            ).arrange(DOWN, buff=0.12).move_to([x, HEAD_Y, 0])
            cells = {
                "sigmoid": MathTypst("approx " + saturating(values["sigmoid"]), font_size=26, color=sign_color(values["sigmoid"])),
                "tanh": MathTypst("approx " + saturating(values["tanh"]), font_size=26, color=sign_color(values["tanh"])),
                "relu": MathTypst("= " + typst_number(values["relu"]) if values["relu"] != 0 else "= 0", font_size=26, color=sign_color(values["relu"]) if values["relu"] != 0 else TEXT),
            }
            for name, cell in cells.items():
                cell.move_to([x, ROW_Y[name], 0])
            return line, dots, head, VGroup(*cells.values())

        row_names = VGroup(*[Text(CURVE_NAMES[n] + "(z)", font=FONT, font_size=24, color=CURVE_COLORS[n]).move_to([NAME_X, ROW_Y[n], 0], aligned_edge=LEFT) for n in ("sigmoid", "tanh", "relu")])
        line1, dots1, head1, rows1 = column(t.focus, TABLE_X[0])
        self.play(Create(line1), FadeIn(VGroup(*dots1.values())), FadeIn(head1), run_time=1.0)
        self.play(FadeIn(row_names), LaggedStart(*[FadeIn(r) for r in rows1], lag_ratio=0.3), run_time=1.2)
        self.swap_caption("The same sum, three outputs. This network uses ReLU.")
        self.wait(2.0)

        line2, dots2, head2, rows2 = column(t.quiet, TABLE_X[1])
        self.play(Create(line2), FadeIn(VGroup(*dots2.values())), FadeIn(head2), run_time=0.9)
        self.play(LaggedStart(*[FadeIn(r) for r in rows2], lag_ratio=0.3), run_time=1.0)
        self.swap_caption("Negative sum: ReLU outputs 0. The neuron is switched off for this digit.")
        self.wait(2.2)
        self.swap_caption("Without such a bend, stacked layers would collapse into a single weighted sum.")
        self.wait(2.2)
        self.activation_group = VGroup(plot, names, line1, *dots1.values(), line2, *dots2.values(), head1, head2, row_names, rows1, rows2)

    # ---- beat 6: the whole forward pass -------------------------------------------------------

    def beat_forward_pass(self):
        self.beat("forward-pass")
        ex = self.example
        net = self.net = self.new_network()
        self.play(FadeOut(self.activation_group), run_time=0.6)
        self.play(FadeIn(net), run_time=1.2)
        self.swap_caption("Now the whole pass for this digit.")
        fan_flash = [ShowPassingFlash(f.copy().set_stroke(color=POS, width=4, opacity=1), time_width=0.7) for f in net.fans]
        self.play(LaggedStart(*fan_flash, lag_ratio=0.06), run_time=1.0)
        self.play(LaggedStart(*[net.hidden_animation(i) for i in range(len(net.hidden))], lag_ratio=0.05), run_time=1.2)
        self.swap_caption("Hidden layer: 16 patterns matched against the digit. Gray means switched off (0).")
        self.wait(1.6)
        self.play(LaggedStart(*[ShowPassingFlash(l, time_width=0.6) for l in net.signal_lines()], lag_ratio=0.004), run_time=1.4)
        scores = VGroup(*[MathTypst("z = " + typst_number(z), font_size=22, color=sign_color(z)).move_to([BAR_X, net.output_y(j), 0], aligned_edge=LEFT) for j, z in enumerate(ex.z2)])
        self.play(FadeIn(scores), run_time=0.6)
        self.swap_caption("Output layer: ten scores, one per digit.")
        self.wait(1.4)
        self.bars = ProbabilityBars(net, ex.probabilities)
        ring_fades = [ring.animate.set_stroke(POS).set_fill(POS, opacity=0.55 * p) for ring, p in zip(net.outputs, ex.probabilities)]
        self.play(FadeOut(scores), *[GrowFromEdge(b, LEFT) for b in self.bars.bars], *[FadeIn(nm) for nm in self.bars.numbers], *ring_fades, run_time=1.4)
        self.swap_caption("Softmax turns the scores into probabilities that add up to 1.")
        self.wait(1.8)

    # ---- beat 7: the answer --------------------------------------------------------------------

    def beat_answer(self):
        self.beat("answer")
        ex, net = self.example, self.net
        j = ex.predicted
        box = SurroundingRectangle(VGroup(net.outputs[j], net.digit_labels[j], self.bars.bars[j], self.bars.numbers[j]), color=GOLD, buff=0.04)
        answer = VGroup(
            Text(f"The network says \u201C{j}\u201D,", font=FONT, font_size=28, color=TEXT),
            MathTypst(probability_typst(ex.probabilities[j]), font_size=32, color=TEXT),
            Text(f"the dataset says \u201C{ex.label}\u201D.", font=FONT, font_size=28, color=TEXT),
        ).arrange(RIGHT, buff=0.3).move_to([0, self.caption_y, 0])
        self.play(Create(box), run_time=0.6)
        self.swap_caption(answer)
        self.wait(2.6)
        self.play(FadeOut(box), run_time=0.3)

    # ---- beat 8: it can be wrong ----------------------------------------------------------------

    def beat_mistake(self):
        self.beat("mistake")
        t = self.trace
        ex = t.mistake
        old = Group(self.net, self.bars)
        net = self.new_network(ex).show_hidden_activations()
        bars = ProbabilityBars(net, ex.probabilities)
        _tint_outputs = [ring.set_stroke(POS).set_fill(POS, opacity=0.55 * p) for ring, p in zip(net.outputs, ex.probabilities)]
        self.play(FadeOut(old), run_time=0.6)
        self.play(FadeIn(net), FadeIn(bars), run_time=1.0)
        self.swap_caption("A new digit the network has never seen.")
        self.wait(1.0)
        wrong = SurroundingRectangle(VGroup(net.outputs[ex.predicted], net.digit_labels[ex.predicted], bars.bars[ex.predicted], bars.numbers[ex.predicted]), color=GOLD, buff=0.04)
        truth = SurroundingRectangle(VGroup(net.outputs[ex.label], net.digit_labels[ex.label], bars.bars[ex.label], bars.numbers[ex.label]), color=MUTED, buff=0.04)
        self.play(Create(wrong), Create(truth), run_time=0.7)
        answer = VGroup(
            Text(f"The network answers \u201C{ex.predicted}\u201D,", font=FONT, font_size=28, color=TEXT),
            MathTypst(probability_typst(ex.probabilities[ex.predicted]), font_size=32, color=TEXT),
            Text(f"but the digit is a {ex.label}.", font=FONT, font_size=28, color=TEXT),
        ).arrange(RIGHT, buff=0.3).move_to([0, self.caption_y, 0])
        self.swap_caption(answer)
        self.wait(2.4)
        total = t.dataset["n_test"]
        self.swap_caption(f"On {total:,} new digits it is right {t.test_accuracy * 100:.2f}% of the time: {t.test_errors} mistakes.")
        self.wait(2.4)
        self.mistake_group = Group(net, bars, wrong, truth)

    # ---- beat 9: where the weights come from ----------------------------------------------------

    def beat_learning(self):
        self.beat("learning")
        t = self.trace
        hist = t.history
        epochs = [h["epoch"] for h in hist]

        def chart(x_center, y_range, y_ticks, values, title_text, end_text):
            axes = Axes(x_range=[0, len(epochs), 10], y_range=y_range, x_length=4.6, y_length=2.8, axis_config={"color": MUTED, "include_tip": False, "include_numbers": False}).move_to([x_center, CHART_Y, 0])
            curve = VMobject(color=POS, stroke_width=5).set_points_as_corners([axes.c2p(e, v) for e, v in zip(epochs, values)])
            title = Text(title_text, font=FONT, font_size=22, color=TEXT).next_to(axes, UP, buff=0.25)
            xlabel = Text(f"passes over the training digits (0 to {len(epochs)})", font=FONT, font_size=18, color=MUTED).next_to(axes, DOWN, buff=0.2)
            ticks = VGroup(*[Text(s, font=FONT, font_size=16, color=MUTED).next_to(axes.c2p(0, v), LEFT, buff=0.12) for v, s in y_ticks])
            end = MathTypst(end_text, font_size=26, color=POS).next_to(curve.get_end(), UP, buff=0.15)
            return axes, curve, title, xlabel, end, ticks

        losses, accs = [h["train_loss"] for h in hist], [h["test_accuracy"] for h in hist]
        left = chart(-3.4, [0, 1.4, 0.5], [(0, "0"), (0.5, "0.5"), (1.0, "1.0")], losses, "error on the training digits", f"{losses[-1]:.2f}")
        right = chart(3.4, [0.6, 1.0, 0.1], [(0.6, "60%"), (0.8, "80%"), (1.0, "100%")], accs, "accuracy on new digits", f"{accs[-1] * 100:.1f}%")
        # Start values sit to the right of the curve's first point, where the steep curve leaves room.
        start_loss = MathTypst(f"{losses[0]:.2f}", font_size=26, color=POS).next_to(left[1].get_start(), RIGHT, buff=0.2)
        start_acc = MathTypst(f"{accs[0] * 100:.1f}%", font_size=26, color=POS).next_to(right[1].get_start(), RIGHT, buff=0.2)
        self.play(FadeOut(self.mistake_group), run_time=0.6)
        self.play(*[FadeIn(m) for c in (left, right) for m in (c[0], c[2], c[3], c[5])], run_time=0.8)
        self.swap_caption(f"Learning: guess, compare with the label, nudge every weight a little. {len(epochs)} passes over {t.dataset['n_train']:,} digits.")
        self.play(Create(left[1]), Create(right[1]), FadeIn(start_loss), FadeIn(start_acc), run_time=3.0, rate_func=linear)
        self.play(FadeIn(left[4]), FadeIn(right[4]), run_time=0.6)
        self.wait(1.6)
        self.swap_caption("The same arithmetic as before, just bigger: weighted sums, activations, softmax.")
        self.wait(2.4)
