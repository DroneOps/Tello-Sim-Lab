"""Holding a position: one PID per axis, with an integral for constant drift and a safety cutoff.

It takes the position from any source: :class:`tellolab.floor_flow.FloorFlow` (a mirror),
a motion-capture system (OptiTrack), a marker pose, or the simulator's true pose.

Why the integral: lab Tello #7 drifts ~12 cm/s to its right. A PD only pushes when there is
already an error, so it stayed ~26 cm off; the integral learns the constant push. With
``i_start`` (in rc) a known drift is compensated from takeoff (Tello #7: 4.5 to the right).

Why the cutoff: if the measurement comes out reversed (a wrong mirror orientation, a wrong
sign), the correction pushes the drone away. If, while correcting, it moves away more than
``runaway_cm`` in 1.5 s, or gets farther than ``limit_cm``, the hold switches itself off for the
rest of the flight.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from .util import with_min


@dataclass
class PositionHoldConfig:
    """Defaults: tuned for the lab Tellos with the floor seen through a mirror (AprilTag-Detection,
    config/mission_espejo.json). Symptoms and what to change: docs/controllers.md."""
    kp: float = 0.5  # rc per cm away from where it should be
    kd: float = 0.8  # rc per cm/s it is moving away
    ki: float = 0.15  # rc per cm·s
    i_max: float = 15.0  # cap of the integral, in rc
    i_start: tuple = (0.0, 0.0)  # integral at takeoff (right, forward), in rc: a known drift
    min_rc: int = 10
    max_rc: int = 20
    deadband_rc: float = 4.0  # commands smaller than this: send nothing (the Tello brakes by itself)
    runaway_cm: float = 35.0
    limit_cm: float = 150.0


class PositionHold:
    """``lr, fb, why_off = hold.update(pos, vel, now)`` with pos/vel = (right, forward) relative to
    where it should be (cm, cm/s). Add lr, fb to your own commands.

    ``along_track=True`` (flying a leg): only holds sideways, i.e. stays on the line; forward is
    yours. After ``why_off`` is not None, it returns (0, 0) for the rest of the flight."""

    def __init__(self, cfg: PositionHoldConfig | None = None) -> None:
        self.cfg = cfg or PositionHoldConfig()
        self.i = np.array(self.cfg.i_start, float)
        self.t: float | None = None
        self.off_reason: str | None = None
        self._away: deque = deque()

    def new_flight(self) -> None:
        self.i[:] = self.cfg.i_start
        self.off_reason = None
        self._away.clear()

    def new_leg(self) -> None:
        """When the target point or line changes: the cutoff starts counting again. The integral
        stays: the drift belongs to the drone."""
        self._away.clear()

    def _axis(self, k: int, err: float, vel: float, dt: float) -> int:
        c = self.cfg
        pd = c.kp * err + c.kd * vel
        total = pd + self.i[k]
        if dt > 0 and not (abs(total) >= c.max_rc and total * err > 0):  # anti-windup
            self.i[k] = max(-c.i_max, min(c.i_max, self.i[k] + c.ki * err * dt))
        u = -(pd + self.i[k])
        return 0 if abs(u) < c.deadband_rc else with_min(u, c.min_rc, c.max_rc)

    def update(self, pos, vel, now: float, along_track: bool = False) -> tuple[int, int, str | None]:
        c = self.cfg
        if self.off_reason:
            return 0, 0, self.off_reason
        (pr, pf), (vr, vf) = pos, vel
        away = abs(pr) if along_track else math.hypot(pr, pf)
        h = self._away
        if h and (h[-1][2] != along_track or now - h[-1][0] > 0.5):
            h.clear()
        h.append((now, away, along_track))
        while now - h[0][0] > 1.5:
            h.popleft()
        moved_away = away - min(x[1] for x in h)
        if moved_away > c.runaway_cm or away > c.limit_cm:
            self.off_reason = (f"moved away {moved_away:.0f} cm in 1.5 s while correcting" if moved_away > c.runaway_cm
                               else f"{away:.0f} cm away")
            return 0, 0, self.off_reason
        dt = now - self.t if self.t is not None and now - self.t < 0.5 else 0.0
        self.t = now
        lr = self._axis(0, pr, vr, dt)
        fb = 0 if along_track else self._axis(1, pf, vf, dt)
        return lr, fb, None
