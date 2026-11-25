"""Holding and changing altitude with the ToF sensor.

Why it is needed: the Tello sinks by itself when flying forward (lab: 10–13 cm/s, from 60 to
33 cm), and with a downward mirror it holds its height worse. The ToF also gives isolated false
readings (lab: 81 -> 14 -> 83 cm in 0.4 s) that a naive controller follows.
"""
from __future__ import annotations

from dataclasses import dataclass

from .util import with_min


@dataclass
class AltitudeFilterConfig:
    spike_cm: float = 20.0  # a jump this big that lasts less than spike_s is a false reading
    spike_s: float = 0.6
    max_rate_cm_s: float = 30.0  # a real sink is slower; a step (flying over a box) is followed at this rate


class AltitudeFilter:
    """ToF altitude without isolated spikes, and without changes faster than max_rate_cm_s."""

    def __init__(self, cfg: AltitudeFilterConfig | None = None) -> None:
        self.cfg = cfg or AltitudeFilterConfig()
        self.reset()

    def reset(self) -> None:
        self.value: float | None = None
        self._t: float | None = None
        self._t_spike: float | None = None

    def update(self, alt_cm: float, now: float) -> float:
        c = self.cfg
        dt = min(0.2, now - self._t) if self._t is not None else 0.0
        self._t = now
        if self.value is None:
            self.value = float(alt_cm)
            return self.value
        if abs(alt_cm - self.value) > c.spike_cm:
            self._t_spike = self._t_spike or now
            if now - self._t_spike < c.spike_s:
                return self.value  # isolated jump: ignore it while it does not last
        else:
            self._t_spike = None
        step = c.max_rate_cm_s * dt
        self.value += max(-step, min(step, alt_cm - self.value))
        return self.value


@dataclass
class AltitudeHoldConfig:
    """Defaults: tuned on the lab Tellos (AprilTag-Detection, config/mission_tags.json)."""
    kp: float = 1.2  # rc per cm of error
    ki: float = 0.5  # rc per cm·s: learns the push needed not to sink...
    i_max: float = 15.0  # ...up to this (with 40 it got stuck climbing at full power)
    min_rc: int = 20  # rc below ~20 barely moves the Tello vertically
    max_rc: int = 30  # rc 40 up climbed ~40 cm/s and overshot from 45 to 89 cm
    tol_cm: float = 4.0
    filter: AltitudeFilterConfig | None = None


class AltitudeHold:
    """PI altitude hold on the filtered ToF height.

    Typical use in a 20 Hz loop::

        hold = AltitudeHold(target_cm=100)
        ud = hold.update(altitude_cm(state), time.monotonic(), ud_request=my_ud)
        tello.send_rc_control(lr, fb, ud, yaw)

    ``ud_request != 0`` means "I want to climb/descend now": it is passed through and the height
    where you stop becomes the new target. ``ud_request == 0`` means "hold".
    """

    def __init__(self, cfg: AltitudeHoldConfig | None = None, target_cm: float | None = None) -> None:
        self.cfg = cfg or AltitudeHoldConfig()
        self.filter = AltitudeFilter(self.cfg.filter)
        self.target_cm = target_cm
        self.reset()

    def reset(self) -> None:
        """Call after something that moves the altitude on purpose (a flip recovery, landing)."""
        self.i = 0.0
        self._t: float | None = None
        self.filter.reset()

    def update(self, alt_cm: float, now: float, ud_request: float = 0) -> int:
        c = self.cfg
        alt = self.filter.update(alt_cm, now)
        dt = min(0.2, now - self._t) if self._t is not None else 0.0
        self._t = now
        if ud_request != 0:
            self.target_cm = alt  # moving on purpose: hold wherever it stops
            return int(ud_request)
        if self.target_cm is None:
            self.target_cm = alt
        err = self.target_cm - alt
        self.i = max(-c.i_max, min(c.i_max, self.i + c.ki * err * dt))
        u = c.kp * err + self.i
        if abs(u) >= c.min_rc or abs(err) > c.tol_cm:
            return with_min(u, c.min_rc, c.max_rc)
        return 0


@dataclass
class GoToAltitudeConfig:
    """Defaults from the lab (config/flight.json): climbing is fast, descending slow."""
    kp: float = 1.0
    min_rc: int = 15
    max_rc: int = 40
    tol_cm: float = 5.0
    min_alt_cm: float = 30.0
    max_alt_cm: float = 180.0


class GoToAltitude:
    """rc to reach a height: ``ud, done = goto.update(target_cm, altitude_cm(state))``."""

    def __init__(self, cfg: GoToAltitudeConfig | None = None) -> None:
        self.cfg = cfg or GoToAltitudeConfig()

    def update(self, target_cm: float, alt_cm: float) -> tuple[int, bool]:
        c = self.cfg
        target = min(max(target_cm, c.min_alt_cm), c.max_alt_cm)
        err = target - alt_cm
        on_target = abs(err) <= c.tol_cm
        return (0 if on_target else with_min(c.kp * err, c.min_rc, c.max_rc)), on_target
