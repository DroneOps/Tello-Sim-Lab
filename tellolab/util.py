"""Small helpers used by every controller."""
from __future__ import annotations

import math
import time


def altitude_cm(state: dict) -> int:
    """Height above the floor in cm, from the ToF sensor (it reads ~10 on the ground).

    Do not use the Tello's own 'h': on the lab Tellos it read 10 cm while flying at 76 and -30 on
    the ground. If the ToF is out of range (>= 1000, e.g. too high or over a hole) it falls back
    to 'h'."""
    tof = state.get("tof", 0)
    return tof - 10 if 0 < tof < 1000 else state.get("h", 0)


def wrap_deg(a: float) -> float:
    """Angle to (-180, 180]."""
    return (a + 180.0) % 360.0 - 180.0


def clamp(v: float, lim: float) -> int:
    return int(max(-lim, min(lim, v)))


def with_min(value: float, minimum: float, limit: float) -> int:
    """Proportional command with a floor: the Tello ignores small rc values (deadband), so a
    nonzero command is raised to at least ``minimum`` and capped at ``limit``."""
    if value == 0:
        return 0
    return clamp(math.copysign(max(abs(value), minimum), value), limit)


def now() -> float:
    return time.monotonic()
