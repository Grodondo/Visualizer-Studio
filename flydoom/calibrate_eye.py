"""How the eye's brightness threshold was chosen (run once; prints the table the eye.py docstring quotes).

Uses ViZDoom's object LABELS buffer to know which pixels are fireballs. This happens ONLY here, offline, on the
calibration seeds, to choose one number (BRIGHT_LUMINANCE). No controller ever receives labels.

    uv run --group flydoom python -m flydoom.calibrate_eye
"""
from __future__ import annotations

import numpy as np
import vizdoom as vzd

from .game import SCENARIO, make_game
from .trials import CALIBRATION_SEEDS

THRESHOLDS = (130, 150, 170, 190)


def fireball_luminance(seed: int):
    game = make_game(seed)
    game.close()
    game.set_labels_buffer_enabled(True)
    game.init()
    game.new_episode()
    rng = np.random.default_rng(seed)
    fb, other = [], []
    while not game.is_episode_finished():
        s = game.get_state()
        lum = s.screen_buffer.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
        ids = [l.value for l in s.labels if l.object_name == "DoomImpBall"]
        m = np.isin(s.labels_buffer, ids)
        if m.any():
            fb.append(lum[m])
            other.append(lum[~m])
        game.make_action([int(rng.integers(0, 2)), int(rng.integers(0, 2))])
    game.close()
    return np.concatenate(fb), np.concatenate(other)


def main():
    parts = [fireball_luminance(s) for s in CALIBRATION_SEEDS]
    fb = np.concatenate([p[0] for p in parts])
    other = np.concatenate([p[1] for p in parts])
    print(f"{SCENARIO}, calibration seeds {CALIBRATION_SEEDS}: {fb.size:,} fireball pixels, {other.size:,} other pixels")
    for th in THRESHOLDS:
        print(f"  luminance > {th}: {np.mean(fb > th):6.1%} of fireball pixels, {np.mean(other > th):.4%} of other pixels")


if __name__ == "__main__":
    main()
