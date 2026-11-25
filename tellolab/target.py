"""Getting over (or in front of) a visual target and staying there: a marker, a pad, a gate.

Input: where the target is relative to the drone, in cm (forward, right), from your detector
(e.g. a marker pose with solvePnP). Output: rc (lr, fb).

The command brakes with the target's velocity: with the ~170 ms video delay and the drone's
inertia, a plain P overshot and oscillated ±15–20 cm (lab, Nov 9). The velocity is computed
without the part that is only due to the drone turning (a target 10 cm away seemed to move
~6 cm/s while turning, and the brake pushed sideways).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from .util import with_min, wrap_deg


@dataclass
class TargetCenteringConfig:
    """Defaults: lab Tellos centering over floor tags through a mirror (config/mission_espejo.json)."""
    kp: float = 0.6
    kd: float = 1.0  # rc per cm/s: brakes depending on how fast it approaches
    min_rc: int = 10
    max_rc: int = 15
    deadband_rc: float = 5.0
    tol_cm: float = 12.0  # "over the target" when closer than this...
    vel_tol_cm_s: float = 8.0  # ...and moving slower than this
    window_s: float = 0.5  # velocity estimated over this window


class TargetCentering:
    """``lr, fb, settled = c.update(fwd_cm, right_cm, t_frame, yaw_deg)``."""

    def __init__(self, cfg: TargetCenteringConfig | None = None) -> None:
        self.cfg = cfg or TargetCenteringConfig()
        self.hist: deque = deque(maxlen=60)

    def reset(self) -> None:
        """When you switch to another target."""
        self.hist.clear()

    def velocity(self, t_frame: float, fwd: float, right: float, yaw: float | None = None) -> tuple[float, float]:
        """Target velocity relative to the drone (cm/s), one sample per frame. With yaw, every
        sample is rotated to the current heading first, so turning alone gives zero velocity."""
        h = self.hist
        if not h or t_frame > h[-1][0]:
            h.append((t_frame, fwd, right, yaw))
        pts = [p for p in h if t_frame - p[0] <= self.cfg.window_s]
        if len(pts) < 4 or pts[-1][0] - pts[0][0] < 0.25:
            return 0.0, 0.0
        t, fs, rs = [p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts]
        if yaw is not None and all(p[3] is not None for p in pts):
            fs, rs = [], []
            for _, f, r, y in pts:
                th = math.radians(wrap_deg(yaw - y))
                fs.append(f * math.cos(th) + r * math.sin(th))
                rs.append(-f * math.sin(th) + r * math.cos(th))
        return float(np.polyfit(t, fs, 1)[0]), float(np.polyfit(t, rs, 1)[0])

    def axis(self, err: float, vel: float, max_rc: float | None = None) -> int:
        """Command for one axis. Far and still: push towards the target; otherwise the velocity
        subtracts (if it already approaches fast enough it does not push; if it overshoots, it
        pushes back)."""
        c = self.cfg
        top = c.max_rc if max_rc is None else max_rc
        u = c.kp * err + c.kd * vel
        if abs(err) > c.tol_cm and abs(vel) < c.vel_tol_cm_s and u * err > 0:
            return with_min(u, c.min_rc, top)
        if abs(u) < c.deadband_rc:
            return 0
        return with_min(u, c.min_rc, top)

    def update(self, fwd: float, right: float, t_frame: float, yaw: float | None = None) -> tuple[int, int, bool]:
        c = self.cfg
        vf, vr = self.velocity(t_frame, fwd, right, yaw)
        settled = abs(fwd) <= c.tol_cm and abs(right) <= c.tol_cm and abs(vf) <= c.vel_tol_cm_s \
            and abs(vr) <= c.vel_tol_cm_s
        return self.axis(right, vr), self.axis(fwd, vf), settled
