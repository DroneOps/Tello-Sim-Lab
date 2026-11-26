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


def results(sim, path) -> dict:
    """What really happened, as plain data (for --report-json and for batches of runs).

    Keys: flight_time_s, battery, crash (None if none), path_cm, end (x, y, z, yaw), highest_cm,
    gates [{name, through}], line_distance_cm {median, max} or None, and landings
    [{pad, distance_cm, on_pad, t_s}] if the drone touched down on a map with landing pads."""
    d = sim.drone
    out = {"flight_time_s": round(float(d.flight_time), 2), "battery": round(float(d.bat), 1),
           "crash": d.crash or None, "path_cm": None, "end": None, "highest_cm": None,
           "gates": [], "line_distance_cm": None, "landings": []}
    if path:
        x, y, z, yaw = sim.pose()
        out["path_cm"] = round(sum(math.dist(a[:2], b[:2]) for a, b in zip(path, path[1:])), 1)
        out["end"] = {"x": round(float(x), 1), "y": round(float(y), 1), "z": round(float(z), 1),
                      "yaw": round(float(yaw), 1)}
        out["highest_cm"] = round(float(max(p[2] for p in path)), 1)
    out["gates"] = [{"name": name, "through": ok} for name, ok in gates_crossed(sim.world, path)]
    ld = line_distance(sim.world, [p for p in path if p[2] > 20])
    if ld:
        out["line_distance_cm"] = {"median": round(ld[0], 1), "max": round(ld[1], 1)}
    if d.mode == "ground" and d.flight_time > 0 and d.touchdown:
        t, tx, ty = d.touchdown  # where the pad was when the drone touched down
        for pad in [o for o in sim.world.objects if isinstance(o, LandingPad)]:
            ox, oy = sim.world.offset(pad, t)
            x, y = tx - ox, ty - oy
            out["landings"].append({"pad": pad.name or "", "t_s": round(float(t), 2),
                                    "distance_cm": round(math.hypot(x - pad.position[0], y - pad.position[1]), 1),
                                    "on_pad": bool(pad.supports(x, y))})
    return out


def summary(sim, path) -> list[str]:
    """The same results as text lines (what tellosim.run prints at the end)."""
    r = results(sim, path)
    lines = [f"flight time {r['flight_time_s']:.0f} s, battery {r['battery']:.0f}%, crash: {r['crash'] or 'none'}"]
    if r["end"]:
        e = r["end"]
        lines.append(f"flew {r['path_cm']:.0f} cm; ended at x={e['x']:.0f} y={e['y']:.0f} z={e['z']:.0f} cm, "
                     f"yaw {e['yaw']:.0f}; highest {r['highest_cm']:.0f} cm")
    for g in r["gates"]:
        lines.append(f"gate {g['name']}: {'flown through' if g['through'] else 'NOT flown through'}")
    if r["line_distance_cm"]:
        ld = r["line_distance_cm"]
        lines.append(f"distance to the line: median {ld['median']:.0f} cm, max {ld['max']:.0f} cm")
    for ln in r["landings"]:
        lines.append(f"landing pad {ln['pad']}: touched down {ln['distance_cm']:.0f} cm from its center"
                     f" ({'on it' if ln['on_pad'] else 'off it'}), at {ln['t_s']:.0f} s".replace("pad :", "pad:"))
    return lines
