"""Metrics. Every controller, the human included, is scored identically.

Defined before the first test run, except where marked REVISED (after the first run, before any tuning on test seeds):
  survival_s        tics survived / 35, capped at 60 s (the cap is ours; see game.EPISODE_CAP_TICS)
  hits              tics on which health went down, the fatal hit included (REVISED: v1 missed the fatal hit)
  damage            health lost in total, the fatal hit included
  move_fraction     share of tics with LEFT or RIGHT pressed
  switches_per_s    changes of action (none/left/right) per second: how jittery the control is
  threat events     REVISED. Ground truth: the first tic on which each new imp fireball is visible (game labels, used
                    for scoring only). v1 used the looming eye at the reflex's own threshold, which made the reflex's
                    reaction time zero by construction.
  reaction_s        per event where the player was standing still on the tic before: time from the fireball's first
                    appearance to the first LEFT or RIGHT press, within REACTION_WINDOW_S (otherwise "no response")
  away_fraction     of those first presses, the share that stepped AWAY from the fireball's side of the screen
  eye_delay_s       per event: time from the fireball's appearance until the shared looming eye's expansion exceeds
                    EVENT_DPS on any side (how long the SENSOR takes; every eye-based controller inherits this delay)
Over episodes: median and the 25-75 % range (episodes are few, so no means and standard errors).

Limitations, said up front: times are quantised to one tic (28.6 ms); controllers that never stand still (constant)
have no measurable reaction time; fireballs often overlap in time, so an "event" may be answered for another fireball;
"away" is a crude direction score, because stepping either way can dodge.
"""
from __future__ import annotations

import numpy as np

from .game import TICRATE

EVENT_DPS = 20.0  # degrees per second of expansion
EVENT_REFRACTORY_S = 0.3
REACTION_WINDOW_S = 1.0


def threat_events(fireballs: list, screen_width: int = 320) -> list[tuple[int, str]]:
    """(tic, side) for the first tic each new fireball is visible; side = its half of the screen."""
    seen, events = set(), []
    for t, balls in enumerate(fireballs):
        for oid, x, _area in balls:
            if oid not in seen:
                seen.add(oid)
                events.append((t, "left" if x < screen_width / 2 else "right"))
    return events


def eye_delays(fireballs: list, eye: list) -> list[float]:
    exp = np.array(eye)[:, 2:4].max(axis=1) if len(eye) else np.zeros(0)
    window = int(round(REACTION_WINDOW_S * TICRATE))
    out = []
    for t, _side in threat_events(fireballs):
        hit = next((dt for dt in range(window + 1) if t + dt < len(exp) and exp[t + dt] > EVENT_DPS), None)
        if hit is not None:
            out.append(hit / TICRATE)
    return out


def reactions(actions: list, fireballs: list) -> dict:
    window = int(round(REACTION_WINDOW_S * TICRATE))
    events = threat_events(fireballs)
    times, away, no_response, moving = [], [], 0, 0
    for t, side in events:
        if t > 0 and actions[t - 1] != "none":
            moving += 1  # already moving: no reaction time can be measured
            continue
        first = next((dt for dt in range(window + 1) if t + dt < len(actions) and actions[t + dt] != "none"), None)
        if first is None:
            no_response += 1
            continue
        times.append(first / TICRATE)
        away.append(actions[t + first] != side)  # fireball on the left -> stepping right is "away"
    return {"reaction_s": times, "away": away, "no_response": no_response, "already_moving": moving, "events": len(events)}


def episode_metrics(log) -> dict:
    health = np.array(log.health + [log.final_health])
    drops = np.diff(health)
    acts = log.actions
    r = reactions(acts, log.fireballs)
    return {
        "seed": log.seed, "controller": log.controller, "tics": log.tics, "survival_s": round(log.survival_s, 4),
        "died": log.died, "hits": int((drops < 0).sum()), "damage": float(-drops[drops < 0].sum()),
        "move_fraction": round(float(np.mean([a != "none" for a in acts])), 4),
        "switches_per_s": round(sum(a != b for a, b in zip(acts, acts[1:])) / max(log.survival_s, 1e-9), 4),
        "events": r["events"], "already_moving": r["already_moving"], "no_response": r["no_response"],
        "reaction_s": [round(v, 4) for v in r["reaction_s"]], "away": r["away"],
        "eye_delay_s": [round(v, 4) for v in eye_delays(log.fireballs, log.eye)], "frame_hash": log.frame_hash,
    }


def summarise(rows: list[dict]) -> dict:
    """Median and 25-75 % range of each number over episodes; reaction times pooled over all events."""
    def q(values):
        v = np.array(values, float)
        return None if v.size == 0 else [round(float(x), 4) for x in np.percentile(v, [25, 50, 75])]

    pooled = [x for r in rows for x in r["reaction_s"]]
    away = [x for r in rows for x in r["away"]]
    return {
        "episodes": len(rows), "survival_s_q": q([r["survival_s"] for r in rows]), "hits_q": q([r["hits"] for r in rows]),
        "move_fraction_q": q([r["move_fraction"] for r in rows]), "switches_per_s_q": q([r["switches_per_s"] for r in rows]),
        "reaction_s_q": q(pooled), "reactions_measured": len(pooled),
        "eye_delay_s_q": q([x for r in rows for x in r["eye_delay_s"]]), "damage_q": q([r["damage"] for r in rows]),
        "away_fraction": None if not away else round(float(np.mean(away)), 3),
        "events": sum(r["events"] for r in rows), "already_moving": sum(r["already_moving"] for r in rows),
        "no_response": sum(r["no_response"] for r in rows), "reached_cap": sum(not r["died"] for r in rows),
    }
