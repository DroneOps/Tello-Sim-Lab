"""Safety watchdogs: when to refuse a takeoff and when to land, and why.

Every check comes from something that happened in the lab:
- telemetry or video lost (WiFi): land instead of flying blind;
- low battery, overheating (the Tello heats 60 -> 85 °C in ~5 min on the ground with video on);
- takeoff that collapses: it climbed ~40 cm and came back to the floor while its own height said
  it was climbing (a clip covering the downward sensors);
- the drone does not obey: asked to climb at full power it went down from 70 to 22 cm and
  forward by itself; asked to descend it climbed. A miscalibrated IMU or a damaged propeller:
  no controller can fix that, land;
- on the floor without asking (e.g. it sank after a flip).

Never switch a watchdog off to make a test "pass": fix the drone instead.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .util import altitude_cm


@dataclass
class SafetyConfig:
    """Defaults: lab Tellos (config/flight.json and mission_tags.json)."""
    battery_min_takeoff: int = 30
    battery_land: int = 15
    temp_max_takeoff_c: int = 80
    temp_land_c: int = 90
    telemetry_lost_s: float = 1.0
    video_lost_s: float = 3.0
    floor_alt_cm: float = 3.0
    floor_s: float = 1.0
    takeoff_min_height_cm: float = 30.0
    takeoff_collapse_cm: float = 15.0
    disobey_ud: int = 20  # asking up/down with at least this...
    disobey_s: float = 0.7  # ...for this long...
    disobey_cm: float = 15.0  # ...and moving this much the other way: land
    disobey_step_cm: float = 10.0  # bigger jumps between readings (a box, a false reading) do not count
    max_flight_s: float | None = None


class SafetyMonitor:
    def __init__(self, cfg: SafetyConfig | None = None) -> None:
        self.cfg = cfg or SafetyConfig()
        self.reset()

    def reset(self) -> None:
        self._obey: deque = deque()
        self._t_floor: float | None = None
        self._takeoff_max = 0.0
        self._t_air: float | None = None

    def preflight(self, state: dict, state_age: float, video_age: float = 0.0) -> list[str]:
        """Reasons not to take off (empty list = go)."""
        c, out = self.cfg, []
        if state_age > c.telemetry_lost_s:
            out.append(f"no telemetry ({state_age:.1f} s)")
        if video_age > 1.0:
            out.append(f"no video ({video_age:.1f} s)")
        if state.get("bat", 0) < c.battery_min_takeoff:
            out.append(f"battery {state.get('bat', 0)}% < {c.battery_min_takeoff}%")
        if state.get("temph", 0) >= c.temp_max_takeoff_c:
            out.append(f"temperature {state.get('temph', 0)} C >= {c.temp_max_takeoff_c} C: cool it down")
        return out

    def check_takeoff(self, state: dict) -> str | None:
        """During takeoff (before the Tello's ok): did it collapse back to the floor?"""
        c = self.cfg
        alt = altitude_cm(state)
        self._takeoff_max = max(self._takeoff_max, alt)
        if self._takeoff_max >= c.takeoff_min_height_cm and alt <= c.takeoff_collapse_cm:
            return (f"takeoff collapsed (climbed to {self._takeoff_max:.0f} and came back to {alt} cm): "
                    f"check that nothing covers the downward sensors")
        return None

    def check(self, state: dict, state_age: float, now: float, video_age: float = 0.0, ud_sent: float = 0.0,
              check_disobey: bool = True) -> str | None:
        """In the air: a reason to land now, or None. Call it every loop. Pass
        check_disobey=False during a flip (the flip has its own altitude control)."""
        c = self.cfg
        self._t_air = self._t_air or now
        if state_age > c.telemetry_lost_s:
            return f"telemetry lost {state_age:.1f} s"
        bat, temp = state.get("bat", 0), state.get("temph", 0)
        if 0 < bat < c.battery_land:
            return f"low battery {bat}%"
        if temp >= c.temp_land_c:
            return f"temperature {temp} C"
        if video_age > c.video_lost_s:
            return f"no video {video_age:.1f} s"
        alt = altitude_cm(state)
        if 0 < state.get("tof", 0) < 1000 and alt <= c.floor_alt_cm:
            self._t_floor = self._t_floor or now
            if now - self._t_floor > c.floor_s:
                return "on the floor without asking"
        else:
            self._t_floor = None
        if c.max_flight_s and now - self._t_air > c.max_flight_s:
            return "maximum flight time"
        if check_disobey:
            why = self._disobeys(alt, ud_sent, now)
            if why:
                return f"the drone does not obey: {why}; check the IMU calibration and the propellers"
        else:
            self._obey.clear()
        return None

    def _disobeys(self, alt: float, ud: float, now: float) -> str | None:
        c = self.cfg
        self._obey.append((now, alt, ud))
        while now - self._obey[0][0] > c.disobey_s:
            self._obey.popleft()
        if len(self._obey) < 10 or now - self._obey[0][0] < 0.9 * c.disobey_s:
            return None
        alts = [a for _, a, _ in self._obey]
        moved = sum(d for d in np.diff(alts) if abs(d) <= c.disobey_step_cm)
        uds = [u for _, _, u in self._obey]
        if min(uds) >= c.disobey_ud and moved <= -c.disobey_cm:
            return f"asked to climb and went down ({moved:+.0f} cm in {c.disobey_s:.1f} s)"
        if max(uds) <= -c.disobey_ud and moved >= c.disobey_cm:
            return f"asked to descend and went up ({moved:+.0f} cm in {c.disobey_s:.1f} s)"
        return None
