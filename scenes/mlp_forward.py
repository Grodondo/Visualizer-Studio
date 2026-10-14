"""Forward pass of a small neural network, as an explanatory video (all six beats).

    uv run manim -ql scenes/mlp_forward.py MlpForwardPass          # draft, 480p15
    uv run manim -qm scenes/mlp_forward.py MlpForwardPass          # draft, 720p30 (easier to judge motion)
    uv run manim -ql -s scenes/mlp_forward.py NetworkReference     # every value of the trace, one still
    uv run python scenes/studio/inspect_video.py media/videos/mlp_forward/480p15/MlpForwardPass.mp4

Render only some beats while working on them (PowerShell):
    $env:BEATS = "zoom-neuron"; uv run manim -ql scenes/mlp_forward.py MlpForwardPass; Remove-Item Env:BEATS

Numbers come from assets/mlp/forward_trace.json (regenerate with `uv run python -m compute.export`).
Final delivery (only when requested):  uv run manim -r 1920,1080 --fps 30 scenes/mlp_forward.py MlpForwardPass

The "KNOBS" below are safe to change: they only move, resize, retime or recolor things.
"""
from pathlib import Path

import numpy as np
from manim import *

from studio.beats import BeatScene
from studio.equations import morph, neuron_equations, rest_of, typst_number
from studio.network import CENTER_Y, NODE_RADIUS, NetworkDiagram
from studio.plots import ActivationPlot
from studio.theme import BG, FOCUS, FONT, GOLD, MUTED, NEG, POS, TEXT, sign_color
from studio.trace import ForwardTrace, probability_typst

TRACE_PATH = Path(__file__).resolve().parent.parent / "assets" / "mlp" / "forward_trace.json"

# ----------------------------------------------------------------------------------------------
# KNOBS. Positions are in scene units: the screen is 14.22 wide and 8 high, (0, 0) is the center,
# x grows to the right and y grows upward.
# ----------------------------------------------------------------------------------------------

# Rows of text under the network (y positions).
CAPTION_Y, KEY_Y, FOOTER_Y = -2.55, -3.0, -3.56

# Beat 1: the input numbers start large and shrink into the input nodes.
INPUT_VALUE_SCALE = 1.8

# Beat 3: which neuron we explain, and how the camera frames it.
FOCUS_LAYER, FOCUS_INDEX = 1, 0  # hidden layer 1, first neuron (named "OR" in the trace)
ZOOM_CENTER = (-1.7, 0.2)  # where the camera looks
ZOOM_WIDTH = 10.4  # how much of the scene fits across the screen; smaller = closer
DIM_AMOUNT = 0.9  # how much the rest of the network fades while we look at one neuron (0 = not at all, 1 = gone)
EQUATION_CENTER = (-1.7, -1.75)  # where the equation sits while the camera is zoomed
EQUATION_PANEL = (9.8, 1.1)  # width and height of the dark strip behind the equation
PLOT_CENTER = (1.1, 1.2)  # the tanh graph
PLOT_PANEL = (4.7, 3.3)

# Beat 6: the recap card. Each row is (equation, explanation); x positions are the left edges of the two columns.
RECAP_X = (-6.0, -1.6)
RECAP_TOP_Y, RECAP_ROW_GAP = 1.5, 0.85
RECAP_DIM = 0.88  # how much the network fades behind the recap card
RECAP_CARD = ((0, 0.4), (14.6, 5.2))  # center and size of the dark card the recap sits on (wider than the screen)


def recap_rows(trace: ForwardTrace) -> list[tuple[str | None, str]]:
    """What the viewer should remember. The activation function's name comes from the trace."""
    return [
        ("z = sum_i w_i a_i + b", "weights \u00D7 inputs, plus a bias: the pre-activation z"),
        (f"a = {trace.activation}(z)", "squash z into the activation a"),
        ('p = "softmax"(z)', "at the end, scores become probabilities p"),
        (None, "repeat for every neuron, layer by layer"),
    ]


def answer_caption(trace: ForwardTrace) -> VGroup:
    """The network answers "different", p ≈ 0.99998: the class and its probability both come from the trace."""
    return VGroup(
        Text(f"The network answers \u201C{trace.prediction_label}\u201D,", font=FONT, font_size=28, color=TEXT),
        MathTypst(probability_typst(trace.prediction_probability), font_size=32, color=TEXT),
    ).arrange(RIGHT, buff=0.3).move_to([0, CAPTION_Y, 0])


def footer() -> VGroup:
    # Two lines: the provenance statement and the rounding rule (probabilities use adaptive precision).
    lines = [
        "Schematic network: weights chosen by hand, not trained.",
        "Values rounded for display; probabilities show additional precision.",
    ]
    return VGroup(*[Text(t, font=FONT, font_size=18, color=MUTED) for t in lines]).arrange(DOWN, buff=0.08).move_to([0, FOOTER_Y, 0])


def caption(text: str) -> Text:
    return Text(text, font=FONT, font_size=28, color=TEXT).move_to([0, CAPTION_Y, 0])


def encoding_key() -> VGroup:
    def item(colour, label):
        return VGroup(Dot(radius=0.07, color=colour), Text(label, font=FONT, font_size=20, color=MUTED)).arrange(RIGHT, buff=0.12)

    key = VGroup(item(POS, "positive"), item(NEG, "negative"), Text("thicker line = larger weight", font=FONT, font_size=20, color=MUTED), Text("no line = weight 0", font=FONT, font_size=20, color=MUTED))
    return key.arrange(RIGHT, buff=0.5).move_to([0, KEY_Y, 0])


def backdrop(center, size) -> Rectangle:
    """A dark panel that hides the dimmed network behind an equation or graph so it stays readable."""
    return Rectangle(width=size[0], height=size[1], fill_color=BG, fill_opacity=0.95, stroke_width=0).move_to([center[0], center[1], 0])


class MlpForwardPass(BeatScene):
    def construct(self):
        self.trace = ForwardTrace.load(TRACE_PATH)
        self.diagram = NetworkDiagram(self.trace)
        self.beat_question()
        self.beat_build()
        self.beat_neuron()
        self.beat_back_out()
        self.beat_forward_pass()
        self.beat_result()

    # Beat 1: the question. The input numbers appear exactly where the input nodes will be.
    def beat_question(self):
        self.beat("question")
        d = self.diagram
        inputs = VGroup(*[node.value for node in d.nodes[0]])
        for node in d.nodes[0]:
            node.set_activation()  # colors the value (and the ring, which is drawn in the next beat)
        for value in inputs:
            value.scale(INPUT_VALUE_SCALE)

        title = Text("Two numbers in. One decision out.", font=FONT, font_size=40, color=TEXT).to_edge(UP, buff=0.7)
        box = RoundedRectangle(corner_radius=0.3, width=3.4, height=2.6, stroke_color=MUTED, stroke_width=4).move_to([0, CENTER_Y, 0])
        mark = Text("?", font=FONT, font_size=110, color=TEXT).move_to(box)
        box_label = Text("a neural network", font=FONT, font_size=24, color=MUTED).next_to(box, DOWN, buff=0.25)
        arrows_in = VGroup(*[Arrow(n.value.get_right() + RIGHT * 0.25, box.get_left() + UP * dy, buff=0, color=MUTED, stroke_width=5) for n, dy in zip(d.nodes[0], (0.5, -0.5))])
        answer = Text("different\nor same?", font=FONT, font_size=34, color=TEXT, line_spacing=0.9).move_to([4.6, CENTER_Y, 0])
        arrow_out = Arrow(box.get_right(), answer.get_left() + LEFT * 0.15, buff=0, color=MUTED, stroke_width=5)
        self.question_group = VGroup(title, box, mark, box_label, arrows_in, answer, arrow_out)

        self.play(Write(title), run_time=1.2)
        self.play(FadeIn(d.input_names), FadeIn(inputs), run_time=0.8)
        self.play(*[GrowArrow(a) for a in arrows_in], FadeIn(box), FadeIn(mark), FadeIn(box_label), run_time=1.0)
        self.play(GrowArrow(arrow_out), FadeIn(answer), run_time=0.8)
        self.wait(1.2)

    # Beat 2: the question box opens up into layers and connections, left to right.
    def beat_build(self):
        self.beat("build-network")
        d = self.diagram
        foot = footer()
        self.play(FadeOut(self.question_group), FadeIn(foot), run_time=0.6)
        self.play(
            *[Create(node.ring) for node in d.nodes[0]],
            *[value.animate.scale(1 / INPUT_VALUE_SCALE) for value in (n.value for n in d.nodes[0])],
            FadeIn(d.layer_titles[0]),
            run_time=0.8,
        )
        for layer in range(1, self.trace.last_layer + 1):
            # Units first, then the connections that join them to the previous layer.
            self.play(LaggedStart(*[GrowFromCenter(n.ring) for n in d.nodes[layer]], lag_ratio=0.2), FadeIn(d.layer_titles[layer]), run_time=0.6)
            self.play(LaggedStart(*[Create(e) for e in d.layer_edges(layer)], lag_ratio=0.1), run_time=1.0)
        self.play(*[FadeIn(label) for label in d.output_labels], run_time=0.6)
        self.build_notes = VGroup(caption("Layers of simple units, joined by weighted connections"), encoding_key())
        self.play(FadeIn(self.build_notes), run_time=1.0)
        self.wait(1.5)

    # Beat 3: zoom into one neuron and follow its arithmetic: weights x inputs, plus bias, then tanh.
    def beat_neuron(self):
        self.beat("zoom-neuron")
        d, trace = self.diagram, self.trace
        node = d.nodes[FOCUS_LAYER][FOCUS_INDEX]
        sources = d.nodes[FOCUS_LAYER - 1]
        bias_value = trace.biases[FOCUS_LAYER - 1][FOCUS_INDEX]
        z_value = trace.pre_activations[FOCUS_LAYER - 1][FOCUS_INDEX]
        a_value = trace.activations[FOCUS_LAYER][FOCUS_INDEX]
        eq = neuron_equations(trace, FOCUS_LAYER, FOCUS_INDEX)
        terms = eq.term_labels  # ["w1", "x1", "w2", "x2", "b"]

        # Everything that is not part of this neuron's story gets dimmed (and restored in the next beat).
        keep = [*d.edges_into(FOCUS_LAYER, FOCUS_INDEX), node.ring, *[s.ring for s in sources], *[s.value for s in sources], *d.input_names]
        is_kept = lambda m: any(m is k for k in keep)
        self.background = VGroup(
            *[e for e in d.edges.values() if not is_kept(e)],
            *[n.ring for layer in d.nodes for n in layer if not is_kept(n.ring)],
            *[label.text for label in d.output_labels],
        )
        self.background.save_state()
        # The layer names would sit on top of the zoomed neuron, so they leave and come back.
        self.play(self.zoom_to(ZOOM_CENTER, ZOOM_WIDTH), self.background.animate.fade(DIM_AMOUNT), FadeOut(self.build_notes), FadeOut(d.layer_titles), run_time=1.3)

        # Mark the neuron and label what goes into it: one weight per line, plus the bias.
        self.halo = Circle(radius=NODE_RADIUS + 0.14, color=FOCUS, stroke_width=4).move_to(node.ring)
        weights = d.weight_labels(FOCUS_LAYER, FOCUS_INDEX)
        bias = MathTypst("b = " + typst_number(bias_value), font_size=30, color=sign_color(bias_value)).next_to(node.ring, UP, buff=0.2)
        self.play(Create(self.halo), run_time=0.5)
        self.play(*[GrowFromCenter(w) for w in weights], FadeIn(bias), run_time=0.8)

        # The rule, in symbols. Each term is boxed in gold together with the matching part of the diagram.
        panel = backdrop(EQUATION_CENTER, EQUATION_PANEL)
        for equation in (eq.symbolic, eq.substituted, eq.products, eq.result):
            equation.move_to([EQUATION_CENTER[0], EQUATION_CENTER[1], 0])
        self.play(FadeIn(panel), Write(eq.symbolic), run_time=1.0)
        for k in range(eq.n_inputs):
            term = VGroup(*eq.symbolic.select(terms[2 * k]), *eq.symbolic.select(terms[2 * k + 1]))
            box = SurroundingRectangle(term, color=GOLD, buff=0.1)
            self.play(Create(box), Indicate(weights[k], color=GOLD), Indicate(sources[k].value, color=GOLD), run_time=0.8)
            self.play(FadeOut(box), run_time=0.2)
        bias_box = SurroundingRectangle(eq.symbolic.select("b"), color=GOLD, buff=0.1)
        self.play(Create(bias_box), Indicate(bias, color=GOLD), run_time=0.8)
        self.play(FadeOut(bias_box), run_time=0.2)

        # Replace each symbol by its real number, then multiply, then add up.
        morph(self, [(eq.symbolic.select(t), eq.substituted.select(t)) for t in terms] + [(rest_of(eq.symbolic, terms), rest_of(eq.substituted, terms))], run_time=1.1)
        self.wait(0.3)
        products = [(VGroup(*eq.substituted.select(terms[2 * k]), *eq.substituted.select(terms[2 * k + 1])), eq.products.select(f"p{k + 1}")) for k in range(eq.n_inputs)]
        morph(self, products + [(eq.substituted.select("b"), eq.products.select("b")), (rest_of(eq.substituted, terms), rest_of(eq.products, eq.product_labels))], run_time=1.1)
        self.wait(0.3)
        everything = VGroup(*[g for label in eq.product_labels for g in eq.products.select(label)])
        morph(self, [(everything, eq.result.select("z")), (rest_of(eq.products, eq.product_labels), rest_of(eq.result, ["z"]))], run_time=1.0)
        z_box = SurroundingRectangle(eq.result.select("z"), color=GOLD, buff=0.1)
        self.play(Create(z_box), run_time=0.5)
        self.wait(0.4)

        # The activation function squashes z into a. The graph is the same tanh the model used.
        plot = ActivationPlot().move_to([PLOT_CENTER[0] + 0.1, PLOT_CENTER[1] - 0.1, 0])
        plot_panel = backdrop(PLOT_CENTER, PLOT_PANEL)
        self.play(FadeIn(plot_panel), FadeIn(plot.axes), FadeIn(plot.ticks), FadeIn(plot.title), run_time=0.6)
        self.play(Create(plot.curve), run_time=0.8)
        z_dot = plot.point_on_axis(z_value)
        z_label = plot.value_label(z_value, z_dot, DOWN)
        self.play(TransformFromCopy(eq.result.select("z"), z_label), FadeIn(z_dot), run_time=0.8)
        up, across = plot.guides(z_value)
        a_dot = plot.point_on_curve(z_value)
        a_label = plot.value_label(a_value, a_dot, DR)
        self.play(Create(up), run_time=0.6)
        self.play(FadeIn(a_dot), Create(across), FadeIn(a_label), run_time=0.6)
        # The result flies back into the neuron, which lights up in its sign color.
        self.play(TransformFromCopy(a_label, node.prepare_value()), node.ring_animation(), run_time=1.2)
        self.wait(0.8)

        self.neuron_overlays = Group(panel, eq.result, z_box, *weights, bias, plot_panel, plot, z_dot, z_label, up, across, a_dot, a_label)

    # Beat 4: back to the whole network, with the first neuron lit.
    def beat_back_out(self):
        self.beat("back-out")
        d, trace = self.diagram, self.trace
        node = d.nodes[FOCUS_LAYER][FOCUS_INDEX]
        self.play(FadeOut(self.neuron_overlays), FadeOut(self.halo), Restore(self.background), FadeIn(d.layer_titles), self.zoom_reset(), run_time=1.4)
        role = trace.neuron_label(FOCUS_LAYER, FOCUS_INDEX)
        if role:
            self.play(FadeIn(Text(role, font=FONT, font_size=20, color=MUTED).next_to(node.ring, UP, buff=0.08)), run_time=0.5)
        self.caption_now = caption("Every neuron repeats this recipe.")
        self.play(Write(self.caption_now), run_time=1.0)
        self.wait(1.5)

    def swap_caption(self, text, run_time: float = 0.6):
        """Replace the caption under the network: the old one fades out while the new one fades in."""
        new = text if isinstance(text, Mobject) else caption(text)
        old, self.caption_now = self.caption_now, new
        self.play(*([FadeOut(old)] if old else []), FadeIn(new), run_time=run_time)


    # Beat 5: the whole forward pass, layer by layer, with the same recipe the viewer just learned.
    def beat_forward_pass(self):
        self.beat("forward-pass")
        self.swap_caption("Layer by layer, the signal flows forward.")
        for layer in range(1, self.trace.last_layer):
            # Layer 1 skips the neuron already explained in beat 3.
            skip = {FOCUS_INDEX} if layer == FOCUS_LAYER else set()
            self.compute_layer(layer, [i for i in range(self.trace.layer_sizes[layer]) if i not in skip])
        out = self.trace.last_layer
        self.output_z = self.compute_layer(out, list(range(self.trace.layer_sizes[out])))
        self.swap_caption("Softmax turns the two scores into probabilities.")
        self.wait(1.0)

    def compute_layer(self, layer: int, indices: list[int]) -> list:
        """One layer of the forward pass: signals flash along the edges, z appears, then the activation lights the node.

        Returns the z labels still on screen (only the output layer's, which turn into probabilities in the next beat).
        """
        d, trace = self.diagram, self.trace
        is_output = layer == trace.last_layer
        self.play(LaggedStart(*[ShowPassingFlash(line, time_width=0.7) for i in indices for line in d.signal_lines(layer, i)], lag_ratio=0.1), run_time=1.0)
        z_labels = [d.pre_activation_label(layer, i) for i in indices]
        self.play(*[FadeIn(z) for z in z_labels], run_time=0.5)
        if is_output:
            self.play(*[d.nodes[layer][i].ring_animation() for i in indices], run_time=0.6)
            return z_labels
        self.play(*[TransformFromCopy(z, d.nodes[layer][i].prepare_value()) for z, i in zip(z_labels, indices)], *[d.nodes[layer][i].ring_animation() for i in indices], run_time=0.9)
        self.play(*[FadeOut(z) for z in z_labels], run_time=0.3)
        return []

    # Beat 6: the answer, then a recap of the recipe.
    def beat_result(self):
        self.beat("result")
        d, trace = self.diagram, self.trace
        winner = d.output_labels[trace.prediction_index]
        self.play(*[FadeOut(z) for z in self.output_z], *[label.prob.animate.set_opacity(1) for label in d.output_labels], run_time=0.8)
        box = SurroundingRectangle(VGroup(winner.text, winner.prob), color=GOLD, buff=0.12)
        answer = answer_caption(trace)
        self.play(Create(box), run_time=0.5)
        self.swap_caption(answer)
        self.wait(1.2)

        # Recap card: the network fades back and the equations of the recipe appear one by one.
        network = VGroup(d, box)
        rows = VGroup()
        for k, (equation, words) in enumerate(recap_rows(trace)):
            y = RECAP_TOP_Y - k * RECAP_ROW_GAP
            row = VGroup()
            if equation:
                row.add(MathTypst(equation, font_size=38, color=TEXT).move_to([RECAP_X[0], y, 0], aligned_edge=LEFT))
            row.add(Text(words, font=FONT, font_size=26, color=MUTED).move_to([RECAP_X[1], y, 0], aligned_edge=LEFT))
            rows.add(row)
        self.play(network.animate.fade(RECAP_DIM), FadeOut(self.caption_now), FadeIn(backdrop(*RECAP_CARD)), run_time=0.8)
        self.play(LaggedStart(*[FadeIn(r, shift=RIGHT * 0.3) for r in rows], lag_ratio=0.45), run_time=2.4)
        self.wait(1.6)
        self.caption_now = caption("A neural network is arithmetic, repeated.")
        self.play(Write(self.caption_now), run_time=1.0)
        self.wait(1.5)


class NetworkReference(Scene):
    """Every node and probability in its computed state: a still for checking the encodings against the trace."""

    def construct(self):
        self.camera.background_color = BG
        trace = ForwardTrace.load(TRACE_PATH)
        diagram = NetworkDiagram(trace).show_all_activations()
        self.add(diagram, footer(), encoding_key())
