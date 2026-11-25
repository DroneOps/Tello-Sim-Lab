"""Holding a heading (yaw) and turning to a heading.

On the lab Tellos rc 10–15 barely turns (yaw deadband ~14): corrections need at least rc 20.
After a turn the drone keeps rotating a bit (asked to stop at 88°, it reached 101°), so a
heading hold right after turning is what lands it on the right angle.
"""
from __future__ import annotations

from dataclasses import dataclass

from .util import with_min, wrap_deg


@dataclass
class HeadingConfig:
    """Defaults: lab Tellos (config/mission_tags.json and flight.json)."""
    kp: float = 1.0  # rc per degree when holding
    min_rc: int = 20
    max_rc: int = 30
    tol_deg: float = 3.0
    turn_kp: float = 2.0  # rc per degree when turning (turn_to)
    turn_max_rc: int = 60  # rc 60 ≈ 44°/s on the real Tello


class HeadingHold:
    """``yaw_rc = heading.update(state["yaw"])`` keeps ``target_deg``. ``turn_to(deg)`` changes the
    target and uses the faster turning gains until it gets there."""

    def __init__(self, cfg: HeadingConfig | None = None, target_deg: float | None = None) -> None:
        self.cfg = cfg or HeadingConfig()
        self.target_deg = target_deg
        self.turning = False

    def turn_to(self, deg: float) -> None:
        self.target_deg = wrap_deg(deg)
        self.turning = True

    def turn_by(self, deg: float, yaw_now: float) -> None:
        self.turn_to(yaw_now + deg)

    def error(self, yaw: float) -> float:
        return 0.0 if self.target_deg is None else wrap_deg(self.target_deg - yaw)

    def update(self, yaw: float) -> int:
        c = self.cfg
        if self.target_deg is None:
            self.target_deg = yaw
        err = self.error(yaw)
        if abs(err) <= c.tol_deg:
            self.turning = False
            return 0
        if self.turning:
            return with_min(c.turn_kp * err, c.min_rc, c.turn_max_rc)
        return with_min(c.kp * err, c.min_rc, c.max_rc)
