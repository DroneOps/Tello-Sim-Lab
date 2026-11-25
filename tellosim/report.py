"""What really happened in a simulated flight (the simulator knows; the drone never tells you)."""
from __future__ import annotations

import math

import numpy as np

from .objects import FloorLine, Gate, LandingPad, compass


def gates_crossed(world, path) -> list[tuple[str, bool]]:
    """For each gate: did the path (x, y, z points) go through its opening?"""
    out = []
    for g in [o for o in world.objects if isinstance(o, Gate)]:
        c, n = np.array(g.position, float), compass(g.facing_deg)
        right = np.cross(np.array([0, 0, 1.0]), n)
        through = False
        for a, b in zip(path, path[1:]):
            a, b = np.array(a, float), np.array(b, float)
            da, db = (a - c) @ n, (b - c) @ n
            if da < 0 <= db:
                q = a + (b - a) * (-da / (db - da)) - c
                x, z = q @ right, q[2]
                inside = math.hypot(x, z) < g.width_cm / 2 if g.shape == "circle" else \
                    abs(x) < g.width_cm / 2 and abs(z) < g.height_cm / 2
                through = through or inside
        out.append((g.name or str(g.color), through))
    return out


def line_distance(world, path):
    """(median, max) distance in cm from the path to the nearest FloorLine, or None."""
    lines = [o for o in world.objects if isinstance(o, FloorLine) and len(o.points) > 1]
    if not lines or not path:
        return None
    segs = [(np.array(a, float), np.array(b, float)) for ln in lines for a, b in zip(ln.points, ln.points[1:])]
    d = []
    for p in path:
        q = np.array(p[:2], float)
        d.append(min(np.linalg.norm(q - (a + np.clip((q - a) @ (b - a) / max((b - a) @ (b - a), 1e-9), 0, 1) * (b - a)))
                     for a, b in segs))
    return float(np.median(d)), float(max(d))


def summary(sim, path) -> list[str]:
    d = sim.drone
    lines = [f"flight time {d.flight_time:.0f} s, battery {d.bat:.0f}%, crash: {d.crash or 'none'}"]
    if path:
        length = sum(math.dist(a[:2], b[:2]) for a, b in zip(path, path[1:]))
        x, y, z, yaw = sim.pose()
        lines.append(f"flew {length:.0f} cm; ended at x={x:.0f} y={y:.0f} z={z:.0f} cm, yaw {yaw:.0f}; "
                     f"highest {max(p[2] for p in path):.0f} cm")
    for name, ok in gates_crossed(sim.world, path):
        lines.append(f"gate {name}: {'flown through' if ok else 'NOT flown through'}")
    ld = line_distance(sim.world, [p for p in path if p[2] > 20])
    if ld:
        lines.append(f"distance to the line: median {ld[0]:.0f} cm, max {ld[1]:.0f} cm")
    if d.mode == "ground" and d.flight_time > 0 and d.touchdown:
        t, tx, ty = d.touchdown  # where the pad was when the drone touched down
        for pad in [o for o in sim.world.objects if isinstance(o, LandingPad)]:
            ox, oy = sim.world.offset(pad, t)
            x, y = tx - ox, ty - oy
            dist = math.hypot(x - pad.position[0], y - pad.position[1])
            lines.append(f"landing pad {pad.name or ''}: touched down {dist:.0f} cm from its center"
                         f" ({'on it' if pad.supports(x, y) else 'off it'}), at {t:.0f} s".replace("pad :", "pad:"))
    return lines
