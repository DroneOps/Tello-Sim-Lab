"""Flips that do not end on the floor.

What we learned flipping the lab Tellos:
- the Tello refuses flips below ~50 % battery;
- after the rotation it FALLS ~50 cm/s for 1–3 s and its own height 'h' does not notice: from
  117 cm it fell to 39, and one flip touched the floor. So: climb to ~130 cm first, and right
  after the rotation climb at full power until it stops falling (brake), stay there a moment
  (hold: if you descend right away it falls again), then go down gently to where you were
  (settle). If it starts falling fast again at any point, brake again;
- the "ok" reply is not proof (replies can be shifted): the flip is confirmed when pitch or
  roll went past 60° and came back level.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .altitude import GoToAltitude, GoToAltitudeConfig
from .util import altitude_cm


@dataclass
class SafeFlipConfig:
    """Defaults: lab Tellos (config/flight.json)."""
    min_battery: float = 50.0
    min_alt_cm: float = 130.0  # climb here before flipping
    climb_timeout_s: float = 12.0
    full_up_rc: int = 40  # "climb at full power" while braking the fall
    hold_s: float = 2.0  # once stopped, do not descend for this long
    brake_vz: float = -8.0  # cm/s: at this vertical speed or above, it stopped falling
    fall_vz: float = -25.0  # cm/s: faster than this is not descending on purpose: brake again
    min_flip_s: float = 0.6  # the real rotation takes ~0.5 s
    max_flip_s: float = 3.0  # give up waiting for the rotation
    level_deg: float = 15.0
    settle_hold_s: float = 1.0
    recover_timeout_s: float = 12.0
    goto: GoToAltitudeConfig = field(default_factory=GoToAltitudeConfig)


class SafeFlip:
    """Usage in a 20 Hz loop (``send`` is ``tello.send_command_without_return``)::

        flip = SafeFlip()
        why_not = flip.start("b", state)            # None if it will flip
        while not flip.finished:
            ud = flip.update(state, time.monotonic(), tello.send_command_without_return)
            tello.send_rc_control(0, 0, ud, 0)      # add your own position hold to lr/fb
        print(flip.result)

    Phases: climb -> flip -> brake -> hold -> settle -> done (or skipped).
    """

    def __init__(self, cfg: SafeFlipConfig | None = None) -> None:
        self.cfg = cfg or SafeFlipConfig()
        self.goto = GoToAltitude(self.cfg.goto)
        self.phase = "idle"
        self.result = ""

    @property
    def finished(self) -> bool:
        return self.phase in ("done", "skipped", "idle")

    def start(self, direction: str, state: dict, now: float | None = None) -> str | None:
        import time
        now = time.monotonic() if now is None else now
        bat = state.get("bat", 0)
        if bat < self.cfg.min_battery:
            self.phase, self.result = "skipped", f"flip skipped: battery {bat}% < {self.cfg.min_battery:.0f}%"
            return self.result
        self.direction = direction
        self.return_alt = altitude_cm(state)
        self.phase, self.t_phase = "climb", now
        self.peak_tilt, self.min_alt, self.t_level, self.t_arrest = 0.0, math.inf, None, None
        self.hist: deque = deque()
        self.result = ""
        return None

    def _go(self, phase: str, now: float) -> None:
        self.phase, self.t_phase = phase, now

    def update(self, state: dict, now: float, send) -> int:
        c = self.cfg
        alt = altitude_cm(state)
        el = now - self.t_phase
        if self.phase == "climb":
            ud, _ = self.goto.update(c.min_alt_cm + 5, alt)
            if alt >= c.min_alt_cm - c.goto.tol_cm:
                send(f"flip {self.direction}")
                self._go("flip", now)
                return 0
            if el > c.climb_timeout_s:
                self.phase, self.result = "skipped", f"flip skipped: could not reach {c.min_alt_cm:.0f} cm (at {alt})"
                return 0
            return ud
        if self.phase == "flip":  # no rc while it rotates
            self.peak_tilt = max(self.peak_tilt, abs(state.get("pitch", 0)), abs(state.get("roll", 0)))
            level = abs(state.get("pitch", 0)) <= c.level_deg and abs(state.get("roll", 0)) <= c.level_deg
            if self.peak_tilt >= 60 and level and el >= c.min_flip_s:
                self.t_level = self.t_level or now
                if now - self.t_level >= 0.3:
                    self._go("brake", now)
            else:
                self.t_level = None
            if el > c.max_flip_s:
                self._go("brake", now)
            return 0
        # recovery
        tof = state.get("tof", 0)
        if 0 < tof < 300:  # the ToF reads garbage (400–600) while upside down
            self.hist.append((now, alt))
            self.min_alt = min(self.min_alt, alt)
        while self.hist and now - self.hist[0][0] > 0.5:
            self.hist.popleft()
        vz = np.polyfit([t for t, _ in self.hist], [a for _, a in self.hist], 1)[0] if len(self.hist) >= 4 else None
        if self.phase in ("hold", "settle") and vz is not None and vz < c.fall_vz:
            self._go("brake", now)
            self.t_arrest = None
        if self.phase == "brake":
            stopped = vz is not None and vz >= c.brake_vz
            self.t_arrest = (self.t_arrest or now) if stopped else None
            if (self.t_arrest is not None and now - self.t_arrest >= 0.3) or alt > c.min_alt_cm + 30:
                self.arrest_alt = alt
                self._go("hold", now)
            else:
                return c.full_up_rc
        if self.phase == "hold":
            if el < c.hold_s:
                return c.full_up_rc if alt < self.arrest_alt - c.goto.tol_cm else 0
            self._go("settle", now)
            self.t_on = None
        if self.phase == "settle":
            ud, on = self.goto.update(self.return_alt, alt)
            self.t_on = (self.t_on or now) if on else None
            if (self.t_on is not None and now - self.t_on >= c.settle_hold_s) or el > c.recover_timeout_s:
                done = self.peak_tilt >= 60
                self.phase = "done"
                self.result = (f"flip {'done' if done else 'NOT detected'}: max tilt {self.peak_tilt:.0f}°, "
                               f"lowest {self.min_alt:.0f} cm after it")
                return 0
            return max(ud, -c.goto.min_rc)  # go down slowly
        return 0
