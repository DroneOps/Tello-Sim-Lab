"""Reading the Tello state safely (works with djitellopy and tellosim)."""
from __future__ import annotations

import math
import time


class Telemetry:
    """The latest state dict (~10 Hz) and how old it is.

    djitellopy replaces the state dict object on every packet, so a new object means a new
    packet. Use the age to detect a lost link: SafetyMonitor lands after 1 s without state."""

    def __init__(self, tello, clock=time.monotonic) -> None:
        self._tello = tello
        self._clock = clock
        self._last: dict | None = None
        self._t = -math.inf

    def snapshot(self) -> tuple[dict, float]:
        """(state, age in seconds). An empty dict and inf before the first packet."""
        s = self._tello.get_own_udp_object()["state"]
        t = self._clock()
        if not s:
            return {}, math.inf
        if s is not self._last:
            self._last, self._t = s, t
        return s, t - self._t
