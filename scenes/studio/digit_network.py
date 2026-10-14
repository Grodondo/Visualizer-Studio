"""Drawing pieces for the digit-recognition video: pixel images, weight maps, the 64-16-10 network, probability bars.

Every number drawn here comes from a DigitsTrace. Encodings (tweak the constants below):
  image pixels      ink gray, black (no ink) to white (full ink): a picture, not a signed quantity
  weight maps       blue = positive weight, orange = negative, brighter = larger |weight|
  hidden neurons    blue ring + tint when active (ReLU output above 0), gray hollow ring when switched off (output 0)
  hidden -> output  line color = sign of the weight, thickness = |weight|
  probability bars  length = probability
"""
from __future__ import annotations

import numpy as np
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Circle,
    Line,
    MathTypst,
    Rectangle,
    Square,
    Text,
    VGroup,
    interpolate_color,
    ManimColor,
)

from .digits_trace import INK_MAX, DigitExample, DigitsTrace
from .theme import BG, FONT, MUTED, NEG, POS, TEXT, sign_color
from .trace import format_probability

# ---- layout of the overview (scene units; the screen is 14.22 wide and 8 high) -----------------
GRID_CENTER = (-5.0, 0.35)
GRID_CELL = 0.34
HIDDEN_X, HIDDEN_SPACING, HIDDEN_RADIUS = -0.9, 0.3, 0.12
OUTPUT_X, OUTPUT_SPACING, OUTPUT_RADIUS = 2.6, 0.42, 0.17
CENTER_Y = 0.35
BAR_X, BAR_MAX = 3.55, 1.9  # bars start here and are at most this long
TITLE_Y = 2.85

# ---- encodings ----------------------------------------------------------------------------------
TINT_MAX = 0.55
RING_WIDTH = 2.5
EDGE_WIDTH_MIN, EDGE_WIDTH_MAX = 0.4, 3.0
EDGE_OPACITY_MIN, EDGE_OPACITY_MAX = 0.12, 0.9
FAN_OPACITY = 0.35


class PixelGrid(VGroup):
    """An 8x8 picture drawn as 64 squares, row by row from the top left.

    mode "ink":    gray from black (0) to white (vmax)
    mode "signed": blue for positive values, orange for negative, brighter for larger |value| (zero stays dark)
    """

    def __init__(self, values, cell: float = GRID_CELL, mode: str = "ink", vmax: float | None = None):
        super().__init__()
        self.cell, self.mode = cell, mode
        self.values = [float(v) for v in values]
        self.vmax = vmax if vmax else (max(abs(v) for v in self.values) or 1.0)
        self.cells = []
        for k, v in enumerate(self.values):
            row, col = divmod(k, 8)
            square = Square(side_length=cell, stroke_width=0.6, stroke_color=BG, stroke_opacity=0.9)
            square.move_to([(col - 3.5) * cell, (3.5 - row) * cell, 0])
            self._style(square, v)
            self.cells.append(square)
        self.add(*self.cells)
        self.outline = Square(side_length=8 * cell, stroke_color=MUTED, stroke_width=2, fill_opacity=0)
        self.add(self.outline)

    def _style(self, square, v: float) -> None:
        level = min(1.0, abs(v) / self.vmax)
        if self.mode == "ink":
            square.set_fill(interpolate_color(ManimColor(BG), ManimColor("#ffffff"), level), opacity=1)
        else:
            square.set_fill(sign_color(v), opacity=0.06 + 0.94 * level)

    def brightness(self, k: int) -> float:
        return min(1.0, abs(self.values[k]) / self.vmax)

    def number_labels(self, strings, font_size: float = 16) -> VGroup:
        """A small number on every square (dark on bright squares, light on dark ones)."""
        labels = VGroup()
        for k, text in enumerate(strings):
            colour = BG if self.brightness(k) > 0.55 and self.mode == "ink" else TEXT
            labels.add(Text(str(text), font=FONT, font_size=font_size, color=colour).move_to(self.cells[k].get_center()))
        return labels


def weight_gallery(trace: DigitsTrace, cell: float = 0.13, gap: float = 0.24) -> list[PixelGrid]:
    """One signed weight map per hidden neuron, all on the same brightness scale, laid out 4 x 4 around the origin."""
    vmax = max(abs(w) for row in trace.w1 for w in row)
    maps = [PixelGrid(trace.weight_image(i), cell=cell, mode="signed", vmax=vmax) for i in range(len(trace.w1))]
    side = 8 * cell + gap
    for i, m in enumerate(maps):
        row, col = divmod(i, 4)
        m.move_to([(col - 1.5) * side, (1.5 - row) * side, 0])
    return maps


def style_hidden(ring: Circle, activation: float, top: float) -> None:
    """Ring and tint of a hidden neuron: gray and hollow when switched off, blue and brighter the more active."""
    if activation <= 0:
        ring.set_stroke(MUTED, opacity=0.9).set_fill(MUTED, opacity=0)
    else:
        ring.set_stroke(POS, opacity=1).set_fill(POS, opacity=0.08 + (TINT_MAX - 0.08) * min(1.0, activation / top))


class DigitNetwork(VGroup):
    """Image -> 16 hidden neurons -> 10 outputs, for one example digit."""

    def __init__(self, trace: DigitsTrace, example: DigitExample):
        super().__init__()
        self.trace, self.example = trace, example
        n_hidden, n_out = len(trace.w1), len(trace.w2)
        self.grid = PixelGrid(example.pixels, vmax=INK_MAX).move_to([GRID_CENTER[0], GRID_CENTER[1], 0])

        def hidden_y(i):
            return CENTER_Y + (i - (n_hidden - 1) / 2) * -HIDDEN_SPACING

        def output_y(j):
            return CENTER_Y + (j - (n_out - 1) / 2) * -OUTPUT_SPACING

        self.hidden = [Circle(radius=HIDDEN_RADIUS, stroke_width=RING_WIDTH).move_to([HIDDEN_X, hidden_y(i), 0]) for i in range(n_hidden)]
        self.outputs = [Circle(radius=OUTPUT_RADIUS, stroke_width=RING_WIDTH).move_to([OUTPUT_X, output_y(j), 0]) for j in range(n_out)]
        for ring in (*self.hidden, *self.outputs):
            ring.set_stroke(MUTED).set_fill(BG, opacity=1)
        self.digit_labels = [Text(str(j), font=FONT, font_size=24, color=TEXT).next_to(self.outputs[j], RIGHT, buff=0.14) for j in range(n_out)]

        # One thin line per hidden neuron stands for its 64 incoming weights (the real 64 lines are drawn when we zoom in on one).
        start = self.grid.get_right() + RIGHT * 0.08
        self.fans = [Line(start, ring.get_left(), stroke_color=MUTED, stroke_width=1, stroke_opacity=FAN_OPACITY) for ring in self.hidden]

        wmax = max(abs(w) for row in trace.w2 for w in row)
        self.out_edges: dict[tuple[int, int], Line] = {}
        for j, out in enumerate(self.outputs):
            for i, ring in enumerate(self.hidden):
                w = trace.w2[j][i]
                level = abs(w) / wmax
                edge = Line(ring.get_right(), out.get_left(), stroke_color=sign_color(w))
                edge.set_stroke(width=EDGE_WIDTH_MIN + (EDGE_WIDTH_MAX - EDGE_WIDTH_MIN) * level, opacity=EDGE_OPACITY_MIN + (EDGE_OPACITY_MAX - EDGE_OPACITY_MIN) * level)
                edge.weight = w
                self.out_edges[(j, i)] = edge

        self.titles = VGroup(
            *[
                Text(text, font=FONT, font_size=22, color=MUTED).move_to([x, TITLE_Y, 0])
                for text, x in (("Image: 64 pixels", GRID_CENTER[0]), ("Hidden layer: 16 neurons", HIDDEN_X), ("Output: 10 digits", OUTPUT_X + 0.4))
            ]
        )
        self.add(self.grid, *self.fans, *self.out_edges.values(), *self.hidden, *self.outputs, *self.digit_labels, self.titles)

    def hidden_top(self) -> float:
        return max(self.example.a1) or 1.0

    def show_hidden_activations(self) -> "DigitNetwork":
        """Instantly show every hidden neuron in its computed state for this example."""
        for ring, a in zip(self.hidden, self.example.a1):
            style_hidden(ring, a, self.hidden_top())
        return self

    def hidden_animation(self, i: int):
        """Animate hidden neuron i to its computed state (use inside play())."""
        target = self.hidden[i].copy()
        style_hidden(target, self.example.a1[i], self.hidden_top())
        return self.hidden[i].animate.become(target)

    def signal_lines(self) -> list[Line]:
        """Bright copies of the hidden->output edges that carry something: colored by the sign of weight x activation."""
        lines = []
        for (j, i), edge in self.out_edges.items():
            signal = self.trace.signal(self.example, j, i)
            if signal != 0:
                lines.append(edge.copy().set_stroke(color=sign_color(signal), width=edge.get_stroke_width() + 2.5, opacity=1))
        return lines

    def output_y(self, j: int) -> float:
        return float(self.outputs[j].get_center()[1])


def bar_number_typst(p: float) -> str:
    """'approx 0.999' for ordinary probabilities, 'approx 3 times 10^(-9)' for tiny ones (a row of zeros would read as exactly 0)."""
    if p == 0:
        return "= 0"
    if p >= 1e-4:
        return "approx " + format_probability(p)
    mantissa, exponent = f"{p:.0e}".split("e")
    return f"approx {mantissa} times 10^({int(exponent)})"


class ProbabilityBars(VGroup):
    """Ten horizontal bars next to the output neurons: length = probability, with the rounded number on the right."""

    def __init__(self, diagram: DigitNetwork, probabilities):
        super().__init__()
        self.bars, self.numbers = [], []
        for j, p in enumerate(probabilities):
            y = diagram.output_y(j)
            bar = Rectangle(width=max(BAR_MAX * p, 0.02), height=0.22, fill_color=POS, fill_opacity=0.85, stroke_width=0)
            bar.move_to([BAR_X + bar.width / 2, y, 0])
            number = MathTypst(bar_number_typst(p), font_size=22, color=sign_color(p)).move_to([BAR_X + BAR_MAX + 0.1, y, 0], aligned_edge=LEFT)
            self.bars.append(bar)
            self.numbers.append(number)
        self.add(*self.bars, *self.numbers)
