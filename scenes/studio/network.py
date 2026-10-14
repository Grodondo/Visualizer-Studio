"""A static neural-network diagram drawn from a ForwardTrace.

Every number on screen comes from the trace; nothing here computes a forward pass.
Encodings (tweak the constants below):
  node ring + tint   sign of the activation (blue +, orange -); tint strength follows |activation|
  node number        the activation, rounded to DEFAULT_DECIMALS, in the same sign color
  edge color         sign of the weight
  edge thickness     |weight|; a weight of exactly 0 draws no edge (it contributes nothing)
"""
from __future__ import annotations

import numpy as np
from manim import (
    DOWN,
    DR,
    LEFT,
    RIGHT,
    UP,
    Circle,
    DecimalNumber,
    FadeIn,
    Line,
    MathTypst,
    Text,
    VGroup,
)

from .equations import typst_number
from .theme import FONT, MUTED, TEXT, sign_color
from .trace import DEFAULT_DECIMALS, ForwardTrace, probability_typst, typst_label

# Layout, in scene units (the frame is 14.22 wide and 8 high).
FIRST_LAYER_X = -5.2
LAST_LAYER_X = 4.4
NODE_SPACING = 1.2
NODE_RADIUS = 0.42
CENTER_Y = 0.15
TITLE_Y = 2.95

# Encoding.
RING_WIDTH = 4
TINT_MAX = 0.35  # light enough that the number, drawn in the same sign color, stays readable on top
VALUE_FONT_SIZE = 26
EDGE_WIDTH_MIN, EDGE_WIDTH_MAX = 1.0, 6.0
EDGE_FULL_WEIGHT = 4.0  # |weight| drawn at maximum thickness
EDGE_OPACITY = 0.85
NAME_FONT_SIZE = 28
TITLE_FONT_SIZE = 22


class Node(VGroup):
    """A ring plus its activation number. `blank` shows an empty ring (value not yet known to the viewer)."""

    def __init__(self, center, activation: float, show_value: bool = True):
        super().__init__()
        self.activation = activation
        self.ring = Circle(radius=NODE_RADIUS, stroke_width=RING_WIDTH).move_to(center)
        self.add(self.ring)
        self.value = None
        if show_value:
            self.value = DecimalNumber(activation, num_decimal_places=DEFAULT_DECIMALS, mob_class=MathTypst, font_size=VALUE_FONT_SIZE)
            self.value.move_to(center)
            self.add(self.value)
        self.set_blank()

    def set_blank(self) -> "Node":
        self.ring.set_stroke(MUTED).set_fill(MUTED, opacity=0)
        if self.value is not None:
            self.value.set_opacity(0)
        return self

    def set_activation(self) -> "Node":
        colour = sign_color(self.activation)
        self.ring.set_stroke(colour).set_fill(colour, opacity=TINT_MAX * min(1.0, abs(self.activation)))
        if self.value is not None:
            self.value.set_color(colour).set_opacity(1)
        return self

    def _tint(self) -> float:
        return TINT_MAX * min(1.0, abs(self.activation))

    def ring_animation(self):
        """Animate the ring from blank to its sign color and tint (use inside play())."""
        colour = sign_color(self.activation)
        return self.ring.animate.set_stroke(colour).set_fill(colour, opacity=self._tint())

    def prepare_value(self) -> DecimalNumber:
        """Color the number and make it visible, without adding it to the scene yet."""
        self.value.set_color(sign_color(self.activation)).set_opacity(1)
        return self.value

    def reveal(self) -> list:
        """Animations that light the node up: ring tint plus its number fading in."""
        return [self.ring_animation(), FadeIn(self.prepare_value())] if self.value is not None else [self.ring_animation()]


class OutputLabel(VGroup):
    """Class name beside an output node, with its rounded probability underneath."""

    def __init__(self, node: Node, name: str, probability: float):
        super().__init__()
        self.text = Text(name, font=FONT, font_size=NAME_FONT_SIZE, color=TEXT)
        self.prob = MathTypst(probability_typst(probability), font_size=30, color=sign_color(probability))
        self.add(self.text, self.prob)
        self.arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        self.next_to(node.ring, RIGHT, buff=0.3)
        self.prob.set_opacity(0)  # the probability is revealed at the end of the forward pass


class NetworkDiagram(VGroup):
    def __init__(self, trace: ForwardTrace):
        super().__init__()
        self.trace = trace
        sizes = trace.layer_sizes
        last = trace.last_layer
        xs = np.linspace(FIRST_LAYER_X, LAST_LAYER_X, len(sizes))

        def position(layer: int, index: int) -> np.ndarray:
            offset = (index - (sizes[layer] - 1) / 2) * NODE_SPACING
            return np.array([xs[layer], CENTER_Y - offset, 0.0])

        self.nodes: list[list[Node]] = [
            [Node(position(l, i), trace.activations[l][i], show_value=(l != last)) for i in range(n)] for l, n in enumerate(sizes)
        ]

        self.edges: dict[tuple[int, int, int], Line] = {}
        for l in range(1, len(sizes)):
            for to in range(sizes[l]):
                for src in range(sizes[l - 1]):
                    w = trace.weights[l - 1][to][src]
                    if w == 0:
                        continue
                    a, b = self.nodes[l - 1][src].ring.get_center(), self.nodes[l][to].ring.get_center()
                    direction = (b - a) / np.linalg.norm(b - a)
                    edge = Line(a + direction * NODE_RADIUS, b - direction * NODE_RADIUS, stroke_color=sign_color(w), stroke_opacity=EDGE_OPACITY)
                    edge.set_stroke(width=EDGE_WIDTH_MIN + (EDGE_WIDTH_MAX - EDGE_WIDTH_MIN) * min(1.0, abs(w) / EDGE_FULL_WEIGHT))
                    edge.weight = w
                    self.edges[(l, to, src)] = edge

        self.layer_titles = VGroup(
            *[
                Text("Input" if l == 0 else "Output" if l == last else f"Hidden layer {l}", font=FONT, font_size=TITLE_FONT_SIZE, color=MUTED).move_to([xs[l], TITLE_Y, 0])
                for l in range(len(sizes))
            ]
        )
        self.input_names = VGroup(
            *[MathTypst(typst_label(trace.input_labels[i]), font_size=34, color=TEXT).next_to(self.nodes[0][i].ring, LEFT, buff=0.3) for i in range(sizes[0])]
        )
        self.output_labels = [
            OutputLabel(self.nodes[last][i], trace.neuron_labels[last][i], trace.activations[last][i]) for i in range(sizes[last])
        ]

        # Edges first so rings and numbers sit on top of the line ends.
        self.add(*self.edges.values(), *[n for layer in self.nodes for n in layer], self.layer_titles, self.input_names, *self.output_labels)

    def layer_edges(self, layer: int) -> list[Line]:
        """Edges that end in `layer`, in a stable order."""
        return [e for (l, _, _), e in self.edges.items() if l == layer]

    def edges_into(self, layer: int, index: int) -> list[Line]:
        return [e for (l, to, _), e in self.edges.items() if l == layer and to == index]

    def weight_labels(self, layer: int, index: int, font_size: float = 30) -> list[DecimalNumber]:
        """The weight of every edge into one neuron, as a number sitting beside its line.

        The first edge's label goes above its line and the others to the lower right, so labels never sit on a line.
        """
        labels = []
        for k, edge in enumerate(self.edges_into(layer, index)):
            number = DecimalNumber(edge.weight, num_decimal_places=DEFAULT_DECIMALS, mob_class=MathTypst, font_size=font_size, color=sign_color(edge.weight))
            number.next_to(edge.point_from_proportion(0.5), UP if k == 0 else DR, buff=0.12)
            labels.append(number)
        return labels

    def signal_lines(self, layer: int, index: int) -> list[Line]:
        """Bright copies of the edges into one neuron, colored by the sign of the signal each one carries.

        The signal of an edge is weight x incoming activation (trace.contributions). Feed these to ShowPassingFlash.
        """
        lines = []
        for (l, to, src), edge in self.edges.items():
            if l == layer and to == index:
                signal = self.trace.contributions[layer - 1][to][src]
                lines.append(edge.copy().set_stroke(color=sign_color(signal), width=edge.get_stroke_width() + 4, opacity=1))
        return lines

    def pre_activation_label(self, layer: int, index: int) -> MathTypst:
        """'z = 2.40' for one neuron: above a hidden node, or where the probability will appear for an output node."""
        z = self.trace.pre_activations[layer - 1][index]
        is_output = layer == self.trace.last_layer
        label = MathTypst("z = " + typst_number(z), font_size=30 if is_output else 22, color=sign_color(z))
        if is_output:
            label.move_to(self.output_labels[index].prob, aligned_edge=LEFT)
        else:
            label.next_to(self.nodes[layer][index].ring, UP, buff=0.08)
        return label

    def show_all_activations(self) -> "NetworkDiagram":
        """Instantly show every node in its computed state (used for reference stills and tests)."""
        for layer in self.nodes:
            for node in layer:
                node.set_activation()
        for label in self.output_labels:
            label.prob.set_opacity(1)
        return self
