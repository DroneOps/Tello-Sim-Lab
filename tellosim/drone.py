"""Physics of the simulated Tello: what it does with each command, and what its sensors read.

Ported from the AprilTag-Detection simulator (sim/sim_world.py, SimTello), which was fitted to
real flights of the lab Tellos. The behaviors that surprised us in the lab are all here:

- rc has a deadband and a lag; forward + sideways at once is slower and sinks;
- it sinks a little when flying forward;
- it drifts by itself (randomly, and some drones always to one side);
- it slides ~30 cm to a random side during a hard turn, and the turn keeps going a bit after
  releasing the stick;
- takeoff climbs to a random 65–74 cm, overshoots ~9 cm, and only obeys rc after its "ok"
  (4.5–7.8 s);
- after a flip it FALLS ~50 cm/s for 1–3 s unless you climb hard;
- the ToF reads height + 10 cm (and garbage mid-flip); its own height 'h' is unreliable.

Coordinates: x east, y north, z up (cm); yaw in compass degrees (clockwise from +y).
"""
from __future__ import annotations

import collections
import math
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from .model import DroneModel
from .world import World

RADIUS_CM = 10.0  # for collisions (a Tello is ~18 cm across without guards)


def wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def stick(rc: float, deadband: float, gain: float) -> float:
    """Steady speed (cm/s or deg/s) that an rc value gives, with its deadband."""
    return math.copysign(gain * (abs(rc) - deadband), rc) if abs(rc) > deadband else 0.0


class RealClock:
    def now(self) -> float:
        return time.monotonic()


class SimClock:
    """Simulated time, advanced by hand (tests, faster-than-real-time runs)."""

    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t


@dataclass
class Faults:
    """Things that go wrong, all off by default. See docs/faults.md.

    takeoff: "normal", "high" (climbs by itself to ~141 cm and answers ok at 10 s), "collapse"
      (climbs ~40 cm and comes back to the floor while its 'h' says it is climbing; never ok),
      "no_ok" (flies normally but never answers ok: blocking takeoff() times out).
    imu_bad: (from_s, forward_cm_s, up_cm_s): from that time it moves by itself, whatever you ask.
    tof_glitches: [(from_s, to_s, value_cm)]: the ToF reads that value (lab: 81 -> 14 -> 83).
    wind_cm_s: constant wind (x, y) in world coordinates.
    speed_factor: the same rc moves more or less (lab: rc 15 gave 12–20 cm/s between flights).
    video_dropouts: [(from_s, to_s)]: the video freezes (WiFi trouble).
    """
    takeoff: str = "normal"
    imu_bad: tuple | None = None
    tof_glitches: list = field(default_factory=list)
    wind_cm_s: tuple = (0.0, 0.0)
    speed_factor: float = 1.0
    video_dropouts: list = field(default_factory=list)


class SimDrone:
    def __init__(self, world: World, model: DroneModel, seed: int = 0, faults: Faults | None = None,
                 clock=None, battery: float = 90.0, temperature_c: float | None = None,
                 crash_stops: bool = True) -> None:
        self.world, self.m = world, model
        self.faults = faults or Faults()
        self.clock = clock or RealClock()
        self.rng = np.random.default_rng(seed)
        self.crash_stops = crash_stops
        self.lock = threading.RLock()
        self.t0 = self.clock.now()
        # true pose and motion
        self.x, self.y, self.yaw = world.start
        self.z = 0.0
        self.v = np.zeros(3)  # body frame: forward, right, up (cm/s)
        self.v_world = np.zeros(3)
        self.acc_world = np.zeros(3)
        self.yaw_rate, self.yaw_carry = 0.0, 0.0
        self.wind = np.zeros(2)
        self._turn_dir = None
        self.mode = "ground"  # ground, takeoff, air, land, falling
        self.rc = (0, 0, 0, 0)
        self._rc_hist: collections.deque = collections.deque([(-1e9, (0, 0, 0, 0))])
        # takeoff
        self.t_takeoff, self._goal, self._ok_t, self._peaked = None, None, None, False
        self.takeoff_yaw = self.yaw
        # flip
        self.t_flip, self.flip_dir, self.flips, self.fall_end, self.vz_fall = -1e9, "b", 0, -1e9, 0.0
        self.pitch, self.roll, self.tof_garbage = 0, 0, None
        # automatic moves (move_*/rotate_* commands): target in world coordinates
        self.auto_target: np.ndarray | None = None
        self.auto_yaw: float | None = None
        self.auto_speed = model.move_speed_cm_s
        # sensors, battery, temperature
        self.h_bias = 0.0
        self.bat = float(battery)
        self.temp = model.temp_start_c if temperature_c is None else float(temperature_c)
        self.flight_time = 0.0
        self.state: dict = {}
        self.responses: list = []
        self.sent: list = []
        self.crash: str | None = None
        self.min_z_airborne = math.inf
        self.history: collections.deque = collections.deque(maxlen=400)  # (t, x, y, z, yaw)
        self._k = 0
        self._last = self.clock.now()
        self.history.append((self._last, self.x, self.y, self.z, self.yaw))
        self._update_state(self._last)

    # ------------------------------------------------------------------ commands
    def set_rc(self, rc, now: float | None = None) -> None:
        now = self.clock.now() if now is None else now
        rc = tuple(int(max(-100, min(100, v))) for v in rc)
        with self.lock:
            self.rc = rc
            self._rc_hist.append((now, rc))
            if any(rc):  # rc cancels an automatic move, like the real Tello
                self.auto_target, self.auto_yaw = None, None

    def _rc_now(self, now: float):
        lim = now - self.m.command_delay_s
        h = self._rc_hist
        while len(h) > 1 and h[1][0] <= lim:
            h.popleft()
        return h[0][1] if h[0][0] <= lim else (0, 0, 0, 0)

    def command(self, c: str) -> str | None:
        """A Tello SDK command. Returns the immediate reply ("ok", "error") or None when the
        reply comes later (takeoff: after it settles)."""
        now = self.clock.now()
        with self.lock:
            self.sent.append(c)
            parts = c.split()
            name = parts[0] if parts else ""
            if name == "takeoff":
                if self.mode != "ground":
                    return "error"
                if self.bat < 10:
                    return "error"
                self.mode, self.t_takeoff, self._goal, self._peaked = "takeoff", now, None, False
                self.takeoff_yaw = self.yaw
                return None
            if name == "land":
                if self.mode in ("air", "takeoff"):
                    self.mode = "land"
                    self.auto_target = self.auto_yaw = None
                return None
            if name == "emergency":
                if self.mode != "ground":
                    self.mode = "falling"
                return "ok"
            if name == "flip":
                if self.mode != "air" or self.bat < self.m.flip_min_battery:
                    return "error"  # the real Tello refuses flips with low battery
                self.t_flip, self.flips, self.flip_dir = now, self.flips + 1, parts[1] if len(parts) > 1 else "b"
                lo, hi = self.m.flip_fall_s
                self.fall_end = now + 1.0 + float(self.rng.uniform(lo, hi))
                self.vz_fall = -30.0
                return "ok"
            if name in ("forward", "back", "left", "right", "up", "down"):
                if self.mode != "air":
                    return "error"
                d = float(parts[1])
                ps = math.radians(self.yaw)
                fwd, rgt = np.array([math.sin(ps), math.cos(ps), 0]), np.array([math.cos(ps), -math.sin(ps), 0])
                vec = {"forward": fwd, "back": -fwd, "right": rgt, "left": -rgt,
                       "up": np.array([0, 0, 1.0]), "down": np.array([0, 0, -1.0])}[name] * d
                self.auto_target = np.array([self.x, self.y, self.z]) + vec
                return None
            if name in ("cw", "ccw"):
                if self.mode != "air":
                    return "error"
                self.auto_yaw = wrap_deg(self.yaw + (1 if name == "cw" else -1) * float(parts[1]))
                return None
            if name == "go" and len(parts) >= 5:
                if self.mode != "air":
                    return "error"
                gx, gy, gz, speed = (float(v) for v in parts[1:5])  # Tello frame: x forward, y left, z up
                ps = math.radians(self.yaw)
                fwd, rgt = np.array([math.sin(ps), math.cos(ps), 0]), np.array([math.cos(ps), -math.sin(ps), 0])
                self.auto_target = np.array([self.x, self.y, self.z]) + gx * fwd - gy * rgt + np.array([0, 0, gz])
                self.auto_speed = speed
                return None
            if name == "speed" and len(parts) == 2:
                self.auto_speed = float(parts[1])
                return "ok"
            return "ok"

    def busy(self) -> bool:
        """An automatic move/rotation is still running."""
        return self.auto_target is not None or self.auto_yaw is not None

    # ------------------------------------------------------------------ physics
    def step(self, dt: float | None = None, now: float | None = None) -> None:
        """One physics step. Real time: called by the drone thread every 20 ms. Simulated time:
        called by Tello.sim.advance()."""
        with self.lock:
            now = self.clock.now() if now is None else now
            dt = min(0.1, now - self._last) if dt is None else dt
            self._last = now
            if dt <= 0:
                return
            self._step(dt, now)

    def _drift(self, turning: bool) -> tuple[float, float]:
        m = self.m
        fw, rt = m.drift_forward_cm_s, m.drift_right_cm_s
        if turning and m.turn_drift_cm_s:
            if self._turn_dir is None:
                self._turn_dir = float(self.rng.uniform(0, 2 * math.pi))
            fw += m.turn_drift_cm_s * math.cos(self._turn_dir)
            rt += m.turn_drift_cm_s * math.sin(self._turn_dir)
        elif not turning:
            self._turn_dir = None
        return fw, rt

    def _move_world(self, fw: float, rt: float, dt: float) -> None:
        ps = math.radians(self.yaw)
        wx, wy = self.faults.wind_cm_s
        self.x += (fw * math.sin(ps) + rt * math.cos(ps) + self.wind[0] + wx) * dt
        self.y += (fw * math.cos(ps) - rt * math.sin(ps) + self.wind[1] + wy) * dt

    def _takeoff(self, dt: float, now: float) -> None:
        m, el, mode = self.m, now - self.t_takeoff, self.faults.takeoff
        if mode == "collapse":
            goal, ok_t = (40.0 if el < 2.5 else 0.0), None
        elif mode == "high":
            goal, ok_t = 141.0, 10.0
        else:
            if self._goal is None:
                lo, hi = m.takeoff_height_cm
                self._goal = float(self.rng.uniform(lo, hi)) if hi > lo else float(lo)
                a, b = m.takeoff_ok_s
                self._ok_t = float(self.rng.uniform(a, b)) if b > a else float(a)
            self._peaked = self._peaked or self.z >= self._goal + m.takeoff_peak_cm - 0.5
            goal = self._goal if self._peaked else self._goal + m.takeoff_peak_cm
            ok_t = self._ok_t
        speed = m.takeoff_climb_cm_s if goal > self.z else 15.0
        self.z = goal if abs(goal - self.z) <= speed * dt else self.z + math.copysign(speed * dt, goal - self.z)
        if self.z > 10:  # off the ground: the drift moves it too (lab: Tello 7 went into the net)
            self.wind = 0.98 * self.wind + self.rng.normal(0, m.random_drift_cm_s * 0.2, 2)
            self._move_world(*self._drift(False), dt)
        if mode == "collapse" and el > 2.5 and self.z <= 0.5:
            self.mode = "ground"
            return
        if ok_t is not None and el >= ok_t:
            self.mode = "air"
            if mode != "no_ok":
                self.responses.append("ok")

    def _auto(self, dt: float, now: float) -> None:
        """The Tello's own position control for move_*/rotate_*/go."""
        if self.auto_yaw is not None:
            err = wrap_deg(self.auto_yaw - self.yaw)
            rate = math.copysign(min(abs(err) / max(dt, 1e-3), self.m.rotate_speed_deg_s), err)
            self.yaw = wrap_deg(self.yaw + rate * dt)
            self.yaw_rate = rate
            if abs(err) < 0.5:
                self.auto_yaw, self.yaw_rate = None, 0.0
                self.responses.append("ok")
        if self.auto_target is not None:
            d = self.auto_target - np.array([self.x, self.y, self.z])
            dist = float(np.linalg.norm(d))
            if dist < 2.0:
                self.auto_target = None
                self.responses.append("ok")
            else:
                vw = d / dist * min(self.auto_speed, dist / max(self.m.response_s, 0.2))
                self.wind = 0.98 * self.wind + self.rng.normal(0, self.m.random_drift_cm_s * 0.2, 2)
                self.x += (vw[0] + 0.3 * self.wind[0]) * dt
                self.y += (vw[1] + 0.3 * self.wind[1]) * dt
                self.z = max(0.0, self.z + vw[2] * dt)

    def _step(self, dt: float, now: float) -> None:
        m, f = self.m, self.faults
        rc = self._rc_now(now)
        x0, y0, z0 = self.x, self.y, self.z
        if self.mode == "takeoff":
            self._takeoff(dt, now)
        elif self.mode == "land":
            self.z = max(self.world.surface_height(self.x, self.y), self.z - m.land_speed_cm_s * dt)
            if self.z <= self.world.surface_height(self.x, self.y) + 0.5:
                self.mode, self.z = "ground", self.world.surface_height(self.x, self.y)
                self.responses.append("ok")
        elif self.mode == "falling":
            self.z = max(0.0, self.z - 300 * dt)
            if self.z <= 0:
                self.mode = "ground"
        elif self.mode == "air" and now - self.t_flip < 1.0:
            # flip: one turn in ~1 s, drops ~20 cm, slides ~30 cm, ends a bit rotated; the ToF
            # reads garbage while upside down (lab: 154, 454, 567)
            frac = (now - self.t_flip) / 1.0
            axis_pitch = self.flip_dir in ("f", "b")
            ang = int(wrap_deg(360 * frac))
            self.pitch, self.roll = (ang, 0) if axis_pitch else (0, ang)
            self.tof_garbage = int(self.rng.uniform(150, 600)) if 0.2 < frac < 0.8 else None
            self.z = max(0.0, self.z - 20 * dt)
            ps = math.radians(self.yaw)
            sgn = {"f": 1, "b": -1}.get(self.flip_dir, 0)
            sgr = {"r": 1, "l": -1}.get(self.flip_dir, 0)
            self.x += (sgn * math.sin(ps) + sgr * math.cos(ps)) * 30 * dt
            self.y += (sgn * math.cos(ps) - sgr * math.sin(ps)) * 30 * dt
            self.yaw = wrap_deg(self.yaw + 6 * dt)
        elif self.mode == "air" and now < self.fall_end:
            # after the flip the Tello FALLS (lab: from 130 cm to the floor) and its own height
            # does not notice; climbing hard (rc up) stops it in ~1 s
            self.pitch, self.roll, self.tof_garbage = 0, 0, None
            climb = 1.25 * max(0.0, rc[2] - 3) if now - self.t_flip >= 1.2 and rc[2] > 0 else 0.0
            self.vz_fall += ((-m.flip_fall_cm_s + climb) - self.vz_fall) * min(1, dt / 0.3)
            self.z = max(0.0, self.z + self.vz_fall * dt)
        elif self.mode == "air" and self.busy():
            self._auto(dt, now)
        elif self.mode == "air":
            lr, fb, ud, yr = rc
            g = m.forward_gain * f.speed_factor
            want = [stick(fb, m.deadband, g), stick(lr, m.deadband, g),
                    stick(ud, m.vertical_deadband, m.up_gain if ud > 0 else m.down_gain)]
            if want[0] and want[1]:  # forward and sideways at once: slower, and it sinks
                want[0], want[1] = want[0] * m.combo_factor, want[1] * m.combo_factor
                want[2] -= m.combo_sink_cm_s
            if want[0]:
                want[2] -= m.forward_sink_cm_s
            k = min(1.0, dt / max(m.response_s, 1e-3))
            kz = min(1.0, dt / max(m.vertical_response_s, 1e-3))
            self.v = np.array([self.v[0] + (want[0] - self.v[0]) * k, self.v[1] + (want[1] - self.v[1]) * k,
                               self.v[2] + (want[2] - self.v[2]) * kz])
            w = stick(yr, m.yaw_deadband, m.yaw_gain)
            self.yaw_rate += (w - self.yaw_rate) * min(1.0, dt / max(m.yaw_response_s, 1e-3))
            self.yaw_carry += (m.yaw_carry * w - self.yaw_carry) * min(1.0, dt / max(m.yaw_carry_s, 1e-3))
            self.yaw = wrap_deg(self.yaw + (self.yaw_rate + self.yaw_carry) * dt)
            self.wind = 0.98 * self.wind + self.rng.normal(0, m.random_drift_cm_s * 0.2, 2)
            fw, rt = self._drift(abs(self.yaw_rate + self.yaw_carry) > 15)
            self._move_world(fw + self.v[0], rt + self.v[1], dt)
            self.z = max(0.0, self.z + self.v[2] * dt)
            if f.imu_bad and now - self.t0 >= f.imu_bad[0]:
                ps = math.radians(self.yaw)
                self.x += f.imu_bad[1] * math.sin(ps) * dt
                self.y += f.imu_bad[1] * math.cos(ps) * dt
                self.z = max(0.0, self.z + f.imu_bad[2] * dt)
        # contact with the floor, boxes, walls...
        if self.mode in ("air", "takeoff") and not (self.mode == "takeoff" and self.z < 15):
            hit = self.world.collision((self.x, self.y, self.z), RADIUS_CM)
            if hit and self.crash is None:
                self.crash = hit
                if self.crash_stops:
                    self.mode = "falling"
            floor = self.world.surface_height(self.x, self.y)
            if self.mode == "air" and self.z <= floor + 0.5:
                self.mode, self.z = "ground", floor  # touched down: the Tello stops its motors
                self.crash = self.crash or "touched the floor"
        if self.mode == "air":
            self.min_z_airborne = min(self.min_z_airborne, self.z)
        # bookkeeping
        if self.mode != "ground":
            self.temp += m.temp_c_per_min / 60 * dt
            self.flight_time += dt
            self.bat -= m.battery_pct_per_min / 60 * dt
            if self.bat < 5 and self.mode == "air":
                self.mode = "land"  # the real Tello lands by itself with an empty battery
        else:
            self.yaw_rate, self.v[:] = 0.0, 0.0
        vw = np.array([self.x - x0, self.y - y0, self.z - z0]) / dt
        self.acc_world += ((vw - self.v_world) / dt - self.acc_world) * 0.2
        self.v_world += (vw - self.v_world) * 0.3
        self.history.append((now, self.x, self.y, self.z, self.yaw))
        self._k += 1
        if self._k % 5 == 0:  # the real Tello sends its state at ~10 Hz
            self._update_state(now)

    # ------------------------------------------------------------------ sensors
    def _update_state(self, now: float) -> None:
        m, f = self.m, self.faults
        floor = self.world.surface_height(self.x, self.y)
        tof = int(self.z - floor + 10 + self.rng.normal(0, m.tof_noise_cm)) if self.z > floor + 0.5 else 10
        if self.tof_garbage is not None:
            tof = self.tof_garbage
        for g0, g1, gv in f.tof_glitches:
            if g0 <= now - self.t0 < g1:
                tof = int(gv)
        if m.height_bias_max_cm > 0:  # 'h' wanders: do not trust it (lab: read 10 while the ToF said 76)
            self.h_bias = float(np.clip(0.995 * self.h_bias + self.rng.normal(0, 1.5), -m.height_bias_max_cm,
                                        m.height_bias_max_cm))
        h = 0 if self.mode == "ground" else self.z + self.h_bias
        if self.mode == "takeoff" and f.takeoff == "collapse":
            h = 60 + 10 * (now - self.t_takeoff)  # lab: 'h' climbed while the ToF fell
        ps = math.radians(self.takeoff_yaw)  # speeds in the frame of the takeoff heading, dm/s
        fwd0, rgt0 = np.array([math.sin(ps), math.cos(ps)]), np.array([math.cos(ps), -math.sin(ps)])
        vx, vy, vz = float(self.v_world[:2] @ fwd0), float(self.v_world[:2] @ rgt0), float(self.v_world[2])
        ax, ay = float(self.acc_world[:2] @ fwd0), float(self.acc_world[:2] @ rgt0)
        self.state = {
            "mid": -1, "x": -100, "y": -100, "z": -100, "mpry": "0,0,0",
            "pitch": int(self.pitch), "roll": int(self.roll), "yaw": int(round(self.yaw)),
            "vgx": int(round(vx / 10)), "vgy": int(round(vy / 10)), "vgz": int(round(-vz / 10)),
            "templ": int(round(self.temp - 2)), "temph": int(round(self.temp)),
            "tof": max(10, tof), "h": int(round(max(h, 0) / 10) * 10) if self.mode != "ground" else 0,
            "bat": int(max(0, self.bat)), "baro": round(150.0 + self.z / 100 + float(self.rng.normal(0, 0.05)), 2),
            "time": int(self.flight_time),
            "agx": round(ax / 0.981, 2), "agy": round(ay / 0.981, 2), "agz": round(-1000 - self.acc_world[2] / 0.981, 2),
        }

    def pose(self) -> tuple[float, float, float, float]:
        """True pose (x, y, z, yaw): what the real drone never tells you."""
        with self.lock:
            return self.x, self.y, self.z, self.yaw

    def pose_at(self, t: float) -> tuple[float, float, float, float]:
        """True pose at time t (for the delayed video)."""
        with self.lock:
            hist = list(self.history)
        if not hist or t >= hist[-1][0]:
            return hist[-1][1:] if hist else self.pose()
        for (ta, *a), (tb, *b) in zip(hist, hist[1:]):
            if ta <= t <= tb:
                s = (t - ta) / max(tb - ta, 1e-9)
                yaw = a[3] + wrap_deg(b[3] - a[3]) * s
                return (a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s, a[2] + (b[2] - a[2]) * s, yaw)
        return hist[0][1:]
