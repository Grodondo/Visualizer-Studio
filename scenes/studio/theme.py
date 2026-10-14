"""Visual language shared by every scene. Change a value here and the whole video follows.

The colors mirror web/src/ui/theme.css so a video and the interactive playground look related.
Keep the two in step by hand; there are only a few values.
"""
from manim import ManimColor

# Frame
BG = "#0b0e13"
TEXT = "#e8edf3"
MUTED = "#a3b0bd"

# Meaning of color (never reused for anything else):
#   blue / orange = the sign of a number (weights, inputs, biases, sums, activations)
#   violet        = the one object the viewer should look at (selected neuron)
#   gold          = an important step or result being pointed at
POS = "#4ea8ff"
NEG = "#ff9a3d"
FOCUS = "#b69cff"
GOLD = "#f5c542"

# Typography. Segoe UI ships with Windows; Manim falls back to the system default if it is missing.
FONT = "Segoe UI"
SIZE_LABEL = 24
SIZE_CAPTION = 26


def sign_color(value: float) -> ManimColor:
    """Blue for zero or positive, orange for negative."""
    return ManimColor(POS if value >= 0 else NEG)
