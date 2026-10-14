"""Smoke test: confirms Typst equations render without a LaTeX install.

Run: uv run manim -ql -s scenes/smoke_typst.py TypstSmoke
"""
from manim import *

# Typst text is black by default, which vanishes on Manim's black background.
EQ_COLOR = WHITE


class TypstSmoke(Scene):
    def construct(self):
        eq = MathTypst("a = sigma(w x + b)", color=EQ_COLOR)
        self.add(eq)
