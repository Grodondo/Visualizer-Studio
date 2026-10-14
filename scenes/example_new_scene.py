"""A starter scene for learning: copy this file, rename the class, and change things.

    uv run manim -ql scenes/example_new_scene.py MyFirstScene

It reuses the real numbers from assets/mlp/forward_trace.json and the NetworkDiagram from scenes/studio.
Try: change the title, change which layer lights up (diagram.nodes[2]), add a pause with self.wait(2).
"""
from pathlib import Path

from manim import *

from studio.beats import BeatScene
from studio.network import NetworkDiagram
from studio.theme import FONT, TEXT
from studio.trace import ForwardTrace

TRACE_PATH = Path(__file__).resolve().parent.parent / "assets" / "mlp" / "forward_trace.json"


class MyFirstScene(BeatScene):
    def construct(self):
        trace = ForwardTrace.load(TRACE_PATH)
        diagram = NetworkDiagram(trace)

        # Beat 1: the empty network. Rings and lines only: the numbers have not been "computed" yet.
        self.beat("empty-network")
        title = Text("My first scene", font=FONT, font_size=40, color=TEXT).move_to([0, 3.5, 0])  # above the layer names (y = 2.95)
        all_rings = [node.ring for layer in diagram.nodes for node in layer]
        self.play(Write(title))
        self.play(*[Create(r) for r in all_rings], *[Create(e) for e in diagram.edges.values()], FadeIn(diagram.layer_titles))
        self.wait(1)

        # Beat 2: light up the input layer and hidden layer 1 with their real values from the trace.
        self.beat("light-up")
        self.play(*[animation for node in [*diagram.nodes[0], *diagram.nodes[1]] for animation in node.reveal()], run_time=1.5)
        self.wait(1)
