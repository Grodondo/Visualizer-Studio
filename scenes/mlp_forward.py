"""Forward pass of a small neural network, as an explanatory video (work in progress: beats 1-2 of 6).

    uv run manim -ql scenes/mlp_forward.py MlpForwardPass          # draft, 480p15
    uv run manim -ql -s scenes/mlp_forward.py NetworkReference     # every value of the trace, one still
    uv run python scenes/studio/inspect_video.py media/videos/mlp_forward/480p15/MlpForwardPass.mp4

Numbers come from assets/mlp/forward_trace.json (regenerate with `uv run python -m compute.export`).
Final delivery (only when requested):  uv run manim -r 1920,1080 --fps 30 scenes/mlp_forward.py MlpForwardPass
"""
from pathlib import Path

from manim import *

from studio.beats import BeatScene
from studio.network import CENTER_Y, NetworkDiagram
from studio.theme import BG, FONT, MUTED, NEG, POS, TEXT, sign_color
from studio.trace import ForwardTrace

TRACE_PATH = Path(__file__).resolve().parent.parent / "assets" / "mlp" / "forward_trace.json"

# Vertical positions of the text rows under the network.
CAPTION_Y, KEY_Y, FOOTER_Y = -2.55, -3.0, -3.56
# The input numbers are the subject of beat 1, so they start large and shrink into the input nodes.
INPUT_VALUE_SCALE = 1.8


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


class MlpForwardPass(BeatScene):
    def construct(self):
        self.trace = ForwardTrace.load(TRACE_PATH)
        self.diagram = NetworkDiagram(self.trace)
        self.beat_question()
        self.beat_build()

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
        self.play(FadeIn(caption("Layers of simple units, joined by weighted connections")), FadeIn(encoding_key()), run_time=1.0)
        self.wait(1.5)


class NetworkReference(Scene):
    """Every node and probability in its computed state: a still for checking the encodings against the trace."""

    def construct(self):
        self.camera.background_color = BG
        trace = ForwardTrace.load(TRACE_PATH)
        diagram = NetworkDiagram(trace).show_all_activations()
        self.add(diagram, footer(), encoding_key())
