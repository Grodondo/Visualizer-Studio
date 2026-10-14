"""A small activation-function plot (axes, curve, tick labels) that avoids Manim's LaTeX-backed axis numbers."""
from __future__ import annotations

import numpy as np
from manim import Axes, DashedLine, DecimalNumber, DOWN, LEFT, UP, Dot, MathTypst, VGroup, VMobject, Text

from .theme import FONT, GOLD, MUTED, TEXT, sign_color
from .trace import DEFAULT_DECIMALS

X_LIMIT = 4.0  # the plot shows x from -4 to 4
Y_LIMIT = 1.25


class ActivationPlot(VGroup):
    def __init__(self, function=np.tanh, name: str = "tanh", width: float = 3.6, height: float = 2.2):
        super().__init__()
        self.function = function
        self.axes = Axes(
            x_range=[-X_LIMIT, X_LIMIT, 1],
            y_range=[-Y_LIMIT, Y_LIMIT, 0.5],
            x_length=width,
            y_length=height,
            axis_config={"color": MUTED, "include_tip": False, "include_numbers": False},
        )
        self.curve = self.axes.plot(function, x_range=[-X_LIMIT, X_LIMIT], color=TEXT, stroke_width=4)
        self.ticks = VGroup(
            *[MathTypst(s, font_size=24, color=MUTED).next_to(self.axes.c2p(x, 0), DOWN, buff=0.1) for s, x in (("-4", -4), ("4", 4))],
            *[MathTypst(s, font_size=24, color=MUTED).next_to(self.axes.c2p(0, y), LEFT, buff=0.1) for s, y in (("-1", -1), ("1", 1))],
        )
        # The top-left corner of the plot is empty for tanh (the curve is at the bottom left), so the name sits there.
        self.title = Text(name, font=FONT, font_size=24, color=MUTED).move_to(self.axes.c2p(-2.6, 1.0))
        self.add(self.axes, self.curve, self.ticks, self.title)

    def point_on_axis(self, z: float) -> Dot:
        return Dot(self.axes.c2p(z, 0), radius=0.07, color=sign_color(z))

    def point_on_curve(self, z: float) -> Dot:
        a = float(self.function(z))
        return Dot(self.axes.c2p(z, a), radius=0.07, color=sign_color(a))

    def guides(self, z: float) -> tuple[DashedLine, DashedLine]:
        """Dashed line up from z to the curve, then across to the vertical axis where the output is read."""
        a = float(self.function(z))
        up = DashedLine(self.axes.c2p(z, 0), self.axes.c2p(z, a), color=GOLD, stroke_width=3)
        across = DashedLine(self.axes.c2p(z, a), self.axes.c2p(0, a), color=GOLD, stroke_width=3)
        return up, across

    def value_label(self, value: float, at, direction, font_size: float = 28) -> DecimalNumber:
        return DecimalNumber(value, num_decimal_places=DEFAULT_DECIMALS, mob_class=MathTypst, font_size=font_size, color=sign_color(value)).next_to(at, direction, buff=0.12)


# Curve colors for the activation-function comparison. They are kept apart from blue/orange (sign) and gold/violet (focus).
CURVE_COLORS = {"sigmoid": "#35d6c4", "tanh": "#ff7eb6", "relu": "#ffffff"}
CURVE_NAMES = {"sigmoid": "sigmoid", "tanh": "tanh", "relu": "ReLU"}


class ActivationComparison(VGroup):
    """Sigmoid, tanh and ReLU on one graph, drawn from sample points exported by compute/digits.py."""

    def __init__(self, curves: dict, x_limit: float = 6.0, y_range=(-1.5, 6.5), width: float = 6.2, height: float = 4.2, highlight: str = "relu"):
        super().__init__()
        self.axes = Axes(
            x_range=[-x_limit, x_limit, 2],
            y_range=[y_range[0], y_range[1], 1],
            x_length=width,
            y_length=height,
            axis_config={"color": MUTED, "include_tip": False, "include_numbers": False},
        )
        self.curves = {}
        for name in CURVE_COLORS:
            points = [self.axes.c2p(x, y) for x, y in zip(curves["x"], curves[name])]
            thick = 7 if name == highlight else 4
            self.curves[name] = VMobject(color=CURVE_COLORS[name], stroke_width=thick).set_points_as_corners(points)
        self.ticks = VGroup(
            *[MathTypst(s, font_size=22, color=MUTED).next_to(self.axes.c2p(x, 0), DOWN, buff=0.1) for s, x in (("-4", -4), ("4", 4))],
            *[MathTypst(s, font_size=22, color=MUTED).next_to(self.axes.c2p(0, y), LEFT, buff=0.1) for s, y in (("-1", -1), ("1", 1), ("4", 4))],
        )
        self.add(self.axes, self.ticks, *self.curves.values())

    def curve_label(self, name: str, x: float, font_size: float = 24) -> Text:
        """The function's name, in its curve color, placed beside the curve at input x."""
        y = {"sigmoid": 1.0, "tanh": 1.0, "relu": x}[name]
        return Text(CURVE_NAMES[name], font=FONT, font_size=font_size, color=CURVE_COLORS[name]).next_to(self.axes.c2p(x, y), UP, buff=0.12)

    def reading(self, z: float, values: dict):
        """A gold dashed line at input z and one dot per function where the line meets its curve."""
        line = DashedLine(self.axes.c2p(z, self.axes.y_range[0]), self.axes.c2p(z, self.axes.y_range[1]), color=GOLD, stroke_width=3)
        dots = {name: Dot(self.axes.c2p(z, values[name]), radius=0.08, color=CURVE_COLORS[name]) for name in CURVE_COLORS}
        return line, dots
