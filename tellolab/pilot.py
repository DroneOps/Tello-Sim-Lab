"""A stabilized Tello in a few lines.

    from tellosim import Tello          # or: from djitellopy import Tello
    from tellolab import Pilot

    tello = Tello(); tello.connect(); tello.streamon()
    pilot = Pilot(tello, frame_read=tello.get_frame_read())
    pilot.takeoff(altitude_cm=100)      # preflight checks, takeoff, climb to 100 cm and hold
    pilot.set_rc(0, 20, 0, 0)           # your commands; 0 on an axis = hold it
    ...
    pilot.land()

What it does every 1/20 s:
- vertical: your ud if not 0; otherwise AltitudeHold keeps the height (the Tello sinks by itself);
- yaw: your yaw if not 0; otherwise HeadingHold keeps the heading you were at;
- lr/fb: yours, plus PositionHold if you gave it a ``position`` source and both are 0;
- SafetyMonitor: lands with the reason in ``pilot.reason`` (low battery, overheating, telemetry
  or video lost, the drone does not obey, on the floor...).
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from .altitude import AltitudeHold, AltitudeHoldConfig, GoToAltitude
from .flip import SafeFlip, SafeFlipConfig
from .heading import HeadingConfig, HeadingHold
from .position import PositionHold, PositionHoldConfig
from .safety import SafetyConfig, SafetyMonitor
from .telemetry import Telemetry
from .util import altitude_cm


@dataclass
class PilotConfig:
    rc_hz: float = 20.0
    rc_limit: int = 100
    altitude: AltitudeHoldConfig = field(default_factory=AltitudeHoldConfig)
    heading: HeadingConfig = field(default_factory=HeadingConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    position: PositionHoldConfig = field(default_factory=PositionHoldConfig)
    flip: SafeFlipConfig = field(default_factory=SafeFlipConfig)


class Pilot:
    def __init__(self, tello, cfg: PilotConfig | None = None, frame_read=None, position=None,
                 threaded: bool = True, clock=time.monotonic, log_path=None) -> None:
        """position: optional callable returning ((right, forward), (v_right, v_forward)) of the
        drone relative to where it should hover, in cm and cm/s, or None if unknown right now.
        threaded=False: call ``tick()`` yourself (tests in simulated time).
        log_path: write a FlightLog CSV of every control step (for tools/fit_model.py)."""
        self.t = tello
        self.cfg = cfg or PilotConfig()
        self.tel = Telemetry(tello, clock)
        self.alt = AltitudeHold(self.cfg.altitude)
        self.heading = HeadingHold(self.cfg.heading)
        self.safety = SafetyMonitor(self.cfg.safety)
        self.pos_hold = PositionHold(self.cfg.position)
        self.goto = GoToAltitude()
        self.position = position
        self.frame_read = frame_read
        self.clock = clock
        self.threaded = threaded
        self.wanted = (0, 0, 0, 0)
        self.sent = (0, 0, 0, 0)
        self.flying = False
        self.reason: str | None = None
        self._flip: SafeFlip | None = None
        self._climb_to: float | None = None
        self._last_frame, self._t_frame = None, clock()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.log = None
        if log_path:
            from .log import FlightLog
            self.log = FlightLog(tello, log_path, clock=clock)

    # ------------------------------------------------------------------ commands
    def takeoff(self, altitude_cm: float | None = None) -> None:
        state, age = self.tel.snapshot()
        problems = self.safety.preflight(state, age, self.video_age())
        if problems:
            raise RuntimeError("not taking off: " + "; ".join(problems))
        self.safety.reset()
        self.t.takeoff()
        self.flying, self.reason = True, None
        self.pos_hold.new_flight()
        self.alt.reset()
        self.alt.target_cm = None
        self.heading.target_deg = None
        self._climb_to = altitude_cm
        if self.threaded and self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def set_rc(self, lr: int = 0, fb: int = 0, ud: int = 0, yaw: int = 0) -> None:
        """What you want (rc -100..100). 0 on an axis = hold it."""
        with self._lock:
            self.wanted = (int(lr), int(fb), int(ud), int(yaw))

    def hover(self) -> None:
        self.set_rc(0, 0, 0, 0)

    def set_altitude(self, cm: float) -> None:
        self._climb_to = cm

    def turn_to(self, deg: float) -> None:
        self.heading.turn_to(deg)

    def turn_by(self, deg: float) -> None:
        self.heading.turn_by(deg, self.tel.snapshot()[0].get("yaw", 0))

    def flip(self, direction: str = "b") -> str:
        """Safe flip (blocks until it is over). Returns the result text."""
        f = SafeFlip(self.cfg.flip)
        why_not = f.start(direction, self.tel.snapshot()[0], self.clock())
        if why_not:
            return why_not
        self._flip = f
        while not f.finished and self.flying:
            self._wait(0.05)
        self._flip = None
        self.alt.reset()
        self.alt.target_cm = f.return_alt
        return f.result

    def land(self, reason: str = "") -> None:
        if not self.flying:
            return
        self.flying = False
        self.reason = self.reason or reason or None
        self.t.send_rc_control(0, 0, 0, 0)
        self.t.land()
        if self.log:
            self.log.close()
            self.log = None

    def kill(self) -> None:
        """Motors off: the drone FALLS. Only if it is heading at a person."""
        self.flying = False
        self.t.emergency()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)

    # ------------------------------------------------------------------ loop
    def video_age(self) -> float:
        if self.frame_read is None:
            return 0.0
        f = self.frame_read.frame
        now = self.clock()
        if f is not self._last_frame:
            self._last_frame, self._t_frame = f, now
        return now - self._t_frame

    def _wait(self, seconds: float) -> None:
        if self.threaded:
            time.sleep(seconds)
        else:
            sim = getattr(self.t, "sim", None)
            steps = max(1, int(round(seconds * self.cfg.rc_hz)))
            for _ in range(steps):
                sim.advance(1.0 / self.cfg.rc_hz)
                self.tick()

    def _loop(self) -> None:
        period = 1.0 / self.cfg.rc_hz
        while not self._stop.is_set() and self.flying:
            t0 = time.monotonic()
            self.tick()
            time.sleep(max(0.0, period - (time.monotonic() - t0)))
        self._thread = None

    def tick(self) -> None:
        """One control step: read the state, check safety, compute and send rc."""
        if not self.flying:
            return
        now = self.clock()
        state, age = self.tel.snapshot()
        with self._lock:
            lr, fb, ud, yaw = self.wanted
        flipping = self._flip is not None and not self._flip.finished
        why = self.safety.check(state, age, now, self.video_age(), self.sent[2], check_disobey=not flipping)
        if why:
            print(f"[pilot] landing: {why}")
            self.reason = why
            self.land()
            return
        alt = altitude_cm(state)
        if flipping:
            ud = self._flip.update(state, now, self.t.send_command_without_return)
            lr = fb = yaw = 0
        else:
            if ud == 0 and self._climb_to is not None:
                ud, done = self.goto.update(self._climb_to, alt)
                if done:
                    self.alt.target_cm, self._climb_to = self._climb_to, None
                    ud = 0
            ud = self.alt.update(alt, now, ud_request=ud)
            if yaw == 0:
                yaw = self.heading.update(state.get("yaw", 0))
            else:
                self.heading.target_deg = state.get("yaw", 0)
            if self.position is not None and lr == 0 and fb == 0:
                pv = self.position()
                if pv is not None:
                    dlr, dfb, off = self.pos_hold.update(pv[0], pv[1], now)
                    if off:
                        print(f"[pilot] position hold OFF: {off}")
                    lr, fb = lr + dlr, fb + dfb
        lim = self.cfg.rc_limit
        self.sent = tuple(int(max(-lim, min(lim, v))) for v in (lr, fb, ud, yaw))
        self.t.send_rc_control(*self.sent)
        if self.log:
            self.log.write(self.sent)
