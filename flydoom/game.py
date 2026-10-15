"""The game: ViZDoom 1.3.1, scenario "take_cover", graphics from Freedoom (freedoom2.wad, BSD-3-Clause).

take_cover: a room; imps at the far wall throw fireballs at the player, who can only step LEFT or RIGHT.
There is no shooting. The episode ends when the player dies, or at EPISODE_CAP_TICS (we add the cap; the scenario has none).

Why this scenario: it is a pure "see a threat coming, get out of the way" problem, the same problem the fly's
looming-escape circuit evolved for (Video 2). It has only 2 buttons, so the action mapping is simple and inspectable.

Determinism: the game is seeded with game.set_seed(seed) and run in synchronous PLAYER mode, one action per tic.
tests/test_flydoom.py checks that the same seed + the same actions give bit-identical frames, and that a recorded
.lmp demo replays to the same health trace.

What a controller is allowed to see: ONLY the RGB screen of the current tic (and its own previous action).
Health, positions, object labels and every other game variable are used for SCORING only, never passed to a controller.
The labels buffer is switched on so the scorer knows when each fireball first comes into view (EpisodeLog.fireballs).
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import vizdoom as vzd

SCENARIO = "take_cover"
TICRATE = 35  # Doom tics per second
EPISODE_CAP_TICS = 60 * TICRATE  # 60 s cap so every trial is bounded
SCREEN = vzd.ScreenResolution.RES_320X240
HFOV_DEG = 90.0  # ViZDoom's default horizontal field of view (used by the eye to turn pixels into degrees)
ACTIONS = {"none": [0, 0], "left": [1, 0], "right": [0, 1]}  # buttons: MOVE_LEFT, MOVE_RIGHT
ACTION_NAMES = list(ACTIONS)


def scenario_path(name: str) -> str:
    return os.path.join(os.path.dirname(vzd.__file__), "scenarios", name)


def make_game(seed: int, *, visible: bool = False, spectator: bool = False, resolution=SCREEN) -> vzd.DoomGame:
    """A configured, initialised game. spectator=True lets a human play with the keyboard in a window."""
    game = vzd.DoomGame()
    game.load_config(scenario_path(f"{SCENARIO}.cfg"))
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_screen_resolution(resolution)
    game.set_window_visible(visible)
    game.set_episode_timeout(EPISODE_CAP_TICS)
    game.set_mode(vzd.Mode.SPECTATOR if spectator else vzd.Mode.PLAYER)
    game.set_labels_buffer_enabled(True)  # for scoring only (see the module docstring)
    game.set_seed(seed)
    game.init()
    return game


@dataclass
class EpisodeLog:
    seed: int
    controller: str
    actions: list = field(default_factory=list)  # action name per tic
    health: list = field(default_factory=list)  # health at the START of each tic (scoring only)
    frame_hash: str = ""  # SHA-256 over every frame, to prove two runs were identical
    tics: int = 0
    died: bool = False
    frames: list | None = None  # RGB frames, only when recording for the video
    eye: list | None = None  # an observer eye's output per tic [size L, size R, expansion L, expansion R], for the timeline
    fireballs: list = field(default_factory=list)  # per tic: [[object id, screen x of centre (px), visible area (px)], ...]
    final_health: float = 0.0

    @property
    def survival_s(self) -> float:
        return self.tics / TICRATE


def fireballs_in_view(state) -> list:
    """Ground truth for scoring: every imp fireball visible this tic (from ViZDoom's labels; never given to a controller)."""
    out = []
    for lab in state.labels:
        if lab.object_name == "DoomImpBall":
            area = int((state.labels_buffer == lab.value).sum())
            if area:
                out.append([int(lab.object_id), round(lab.x + lab.width / 2, 1), area])
    return out


def run_episode(controller, seed: int, *, keep_frames: bool = False, record_lmp: Path | None = None) -> EpisodeLog:
    """Play one episode. controller.reset() is called first; controller.act(rgb) returns an action name."""
    from .eye import LoomingEye  # here, not at the top: eye.py imports constants from this module

    game = make_game(seed)
    controller.reset(seed)
    observer = LoomingEye()  # not given to the controller; defines stimulus events identically for everyone
    log = EpisodeLog(seed=seed, controller=controller.name, frames=[] if keep_frames else None, eye=[])
    digest = hashlib.sha256()
    game.new_episode(str(record_lmp) if record_lmp else "")
    while not game.is_episode_finished():
        state = game.get_state()
        rgb = state.screen_buffer
        digest.update(rgb.tobytes())
        log.health.append(float(game.get_game_variable(vzd.GameVariable.HEALTH)))
        action = controller.act(rgb)
        assert action in ACTIONS, action
        log.actions.append(action)
        if keep_frames:
            log.frames.append(rgb.copy())
        log.eye.append(observer.see(rgb).as_list())  # timeline only: the same eye watches every controller
        log.fireballs.append(fireballs_in_view(state))
        game.make_action(ACTIONS[action])
    log.tics = len(log.actions)
    log.died = bool(game.is_player_dead())
    log.final_health = float(game.get_game_variable(vzd.GameVariable.HEALTH))
    log.frame_hash = digest.hexdigest()
    game.close()
    return log


def replay_lmp(path: Path, seed: int, *, keep_frames: bool = False) -> EpisodeLog:
    """Replay a recorded .lmp demo (e.g. a human episode) at the standard resolution, re-rendering every frame.
    Returns what happened: the actions taken each tic, health, frames."""
    from .eye import LoomingEye  # here, not at the top: eye.py imports constants from this module

    game = make_game(seed)
    observer = LoomingEye()
    log = EpisodeLog(seed=seed, controller="replay", frames=[] if keep_frames else None, eye=[])
    digest = hashlib.sha256()
    game.replay_episode(str(path))
    while not game.is_episode_finished():
        state = game.get_state()
        digest.update(state.screen_buffer.tobytes())
        log.health.append(float(game.get_game_variable(vzd.GameVariable.HEALTH)))
        log.eye.append(observer.see(state.screen_buffer).as_list())
        log.fireballs.append(fireballs_in_view(state))
        if keep_frames:
            log.frames.append(state.screen_buffer.copy())
        game.advance_action()
        last = game.get_last_action()
        log.actions.append(action_name(last))
    log.tics = len(log.actions)
    log.died = bool(game.is_player_dead())
    log.final_health = float(game.get_game_variable(vzd.GameVariable.HEALTH))
    log.frame_hash = digest.hexdigest()
    game.close()
    return log


def action_name(buttons) -> str:
    left, right = (float(b) > 0 for b in buttons[:2])
    if left and not right:
        return "left"
    if right and not left:
        return "right"
    return "none"  # neither, or both pressed (they cancel)
