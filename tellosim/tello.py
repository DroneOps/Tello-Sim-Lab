"""A drop-in replacement for ``djitellopy.Tello`` that flies in the simulator.

    # from djitellopy import Tello
    from tellosim import Tello

    tello = Tello()                     # or Tello(world="line_follower", model="lab_tello7")
    tello.connect()
    tello.streamon()
    frame_read = tello.get_frame_read()
    tello.takeoff()
    tello.send_rc_control(0, 15, 0, 0)
    img = frame_read.frame              # RGB, 960x720, ~170 ms late, like the real one
    tello.land()
    tello.end()

Everything the sim knows and the real drone doesn't is under ``tello.sim`` (true pose,
crashes, the world, a viewer). See docs/how-it-works.md.
"""
from __future__ import annotations

import logging
import threading
import time

import numpy as np

from .camera import Camera
from .drone import Faults, RealClock, SimClock, SimDrone
from .model import load_model
from .render import Renderer
from .world import apply_glare, as_fog, load_map

LOGGER = logging.getLogger("tellosim")
if not LOGGER.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] tellosim: %(message)s"))
    LOGGER.addHandler(_h)
    LOGGER.setLevel(logging.INFO)


class TelloException(Exception):
    """Same name as djitellopy's exception, so ``except TelloException`` works the same."""


class BackgroundFrameRead:
    """Like djitellopy's frame reader: ``.frame`` is the latest RGB image."""

    def __init__(self, tello: Tello, with_queue: bool = False, max_queue_len: int = 32) -> None:
        self._tello = tello
        self.stopped = False
        self.with_queue, self.max_queue_len = with_queue, max_queue_len
        h, w = tello.sim.camera.height, tello.sim.camera.width
        self._frame = np.zeros((h, w, 3), np.uint8)
        self._t_frame = -1.0

    @property
    def frame(self) -> np.ndarray:
        s = self._tello.sim
        if not s.realtime:  # simulated time: draw it now (as it looked video_delay_s ago)
            now = s.clock.now()
            if now != self._t_frame:
                img = s.capture(now)
                if img is not None:
                    self._frame = img
                self._t_frame = now
        return self._frame

    def _set(self, img: np.ndarray) -> None:
        self._frame = img

    def start(self) -> BackgroundFrameRead:
        return self

    def stop(self) -> None:
        self.stopped = True


class Sim:
    """The simulator behind a Tello: world, drone, camera, renderer and their threads."""

    def __init__(self, world="lab_grid", model="lab_healthy", camera="lab_tello", camera_mode=None,
                 seed: int = 0, faults: Faults | dict | None = None, battery: float = 90.0,
                 temperature_c: float | None = None, realtime: bool = True, crash_stops: bool = True,
                 fog=None, glare=None, **camera_overrides) -> None:
        self.world = load_map(world)
        if fog is not None:  # fog or smoke: overrides the map's (fog=False: clear air)
            self.world.fog = as_fog(fog)
        if glare is not None:  # laminated markers reflecting the lamps (see world.apply_glare)
            apply_glare(self.world, glare)
        self.model = load_model(model)
        overrides = {**self.world.camera, **camera_overrides}  # the map suggests, the caller decides
        if camera_mode:
            overrides["mode"] = camera_mode
        self.camera = Camera.load(camera, **overrides)
        self.realtime = realtime
        self.clock = RealClock() if realtime else SimClock()
        if isinstance(faults, dict):
            faults = Faults(**faults)
        self.drone = SimDrone(self.world, self.model, seed=seed, faults=faults, clock=self.clock,
                              battery=battery, temperature_c=temperature_c, crash_stops=crash_stops)
        self.renderer = Renderer(self.world, self.camera, seed=seed)
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._readers: list[BackgroundFrameRead] = []
        self.video_on = False
        self._reported_crash = False
        self._viewers: list = []
        if realtime:
            self._start(self._physics_loop)

    # ------------------------------------------------------------------ public helpers
    def pose(self) -> tuple[float, float, float, float]:
        """True (x, y, z, yaw) in cm and compass degrees."""
        return self.drone.pose()

    @property
    def crash(self) -> str | None:
        return self.drone.crash

    def advance(self, seconds: float, dt: float = 0.02) -> None:
        """Simulated-time mode: run the physics for this many seconds (fast)."""
        if self.realtime:
            time.sleep(seconds)
            return
        n = max(1, int(round(seconds / dt)))
        for _ in range(n):
            self.clock.t += dt
            self.drone.step(dt, self.clock.t)
        self._check_crash()

    def capture(self, now: float | None = None) -> np.ndarray | None:
        """The RGB image the camera delivers at time ``now`` (it shows the world as it was
        video_delay_s earlier). None during a video dropout."""
        now = self.clock.now() if now is None else now
        el = now - self.drone.t0
        if any(a <= el < b for a, b in self.drone.faults.video_dropouts):
            return None
        seen = now - self.model.video_delay_s
        x, y, z, yaw = self.drone.pose_at(seen)
        bgr_img = self.renderer.render(x, y, z, yaw, t=seen - self.drone.t0)
        return bgr_img[..., ::-1].copy()  # djitellopy gives RGB

    def world_changed(self) -> None:
        """Call after adding, removing or moving objects while the simulation runs."""
        self.world._texture = None
        self.renderer.invalidate()

    def viewer(self, show: bool = True, record: str | None = None, fps: float = 10.0):
        """Map from above + camera, in a window and/or recorded to an mp4. See viewer.py."""
        from .viewer import Viewer
        v = Viewer(self, show=show, record=record, fps=fps)
        self._viewers.append(v)
        return v.start()

    def close(self) -> None:
        for v in getattr(self, "_viewers", []):
            v.stop()
        self._stop.set()
        for t in self._threads:
            t.join(timeout=1.0)

    # ------------------------------------------------------------------ threads
    def _start(self, fn) -> None:
        t = threading.Thread(target=fn, daemon=True)
        t.start()
        self._threads.append(t)

    def _physics_loop(self) -> None:
        while not self._stop.wait(0.02):
            self.drone.step()
            self._check_crash()

    def _video_loop(self) -> None:
        period = 1.0 / self.camera.fps
        nxt = time.monotonic()
        while not self._stop.is_set():
            if self.video_on and self._readers:
                img = self.capture()
                if img is not None:
                    for r in self._readers:
                        r._set(img)
            nxt += period
            time.sleep(max(0.0, nxt - time.monotonic()))
            if time.monotonic() - nxt > 1.0:
                nxt = time.monotonic()

    def _check_crash(self) -> None:
        if self.drone.crash and not self._reported_crash:
            self._reported_crash = True
            LOGGER.warning(f"CRASH: {self.drone.crash} at x={self.drone.x:.0f} y={self.drone.y:.0f} "
                           f"z={self.drone.z:.0f} cm")


class Tello:
    """Simulated Tello with djitellopy's interface. Extra keyword arguments choose the
    simulation: world (map name, YAML path or World), model (drone model), camera (calibration),
    camera_mode ("forward" or "mirror"), seed, faults, battery, realtime. Any other keyword is a
    camera field (e.g. mirror_tilt_deg=-8)."""

    RESPONSE_TIMEOUT = 7
    TAKEOFF_TIMEOUT = 20
    FRAME_GRAB_TIMEOUT = 5
    LOGGER = LOGGER

    def __init__(self, host: str = "192.168.10.1", retry_count: int = 3, vs_udp=None, **sim_options) -> None:
        self.address = (host, 8889)
        self.retry_count = retry_count
        self.sim = Sim(**sim_options)
        self.is_flying = False
        self.stream_on = False
        self._connected = False
        self._udp = {"state": self.sim.drone.state, "responses": self.sim.drone.responses}
        self._frame_read: BackgroundFrameRead | None = None
        self._last_rc_t = 0.0

    # ------------------------------------------------------------------ connection and video
    def connect(self, wait_for_state: bool = True) -> None:
        self._connected = True
        self.send_control_command("command")

    def streamon(self) -> None:
        self.stream_on = True
        self.sim.video_on = True
        if self.sim.realtime and not any(t.name == "video" for t in self.sim._threads):
            t = threading.Thread(target=self.sim._video_loop, daemon=True, name="video")
            t.start()
            self.sim._threads.append(t)

    def streamoff(self) -> None:
        self.stream_on = False
        self.sim.video_on = False

    def get_frame_read(self, with_queue: bool = False, max_queue_len: int = 32) -> BackgroundFrameRead:
        if not self.stream_on:
            LOGGER.warning("get_frame_read() before streamon(): starting the stream (the real Tello would not)")
            self.streamon()
        if self._frame_read is None:
            self._frame_read = BackgroundFrameRead(self, with_queue, max_queue_len)
            self.sim._readers.append(self._frame_read)
            if self.sim.realtime:  # first frame, like djitellopy waiting for the stream
                t_end = time.monotonic() + self.FRAME_GRAB_TIMEOUT
                while self._frame_read._frame.max() == 0 and time.monotonic() < t_end:
                    time.sleep(0.02)
        return self._frame_read

    def get_udp_video_address(self) -> str:
        return "udp://@0.0.0.0:11111"

    def end(self) -> None:
        if self.is_flying:
            self.land()
        self.streamoff()
        if self._frame_read:
            self._frame_read.stop()
        self.sim.close()

    def __del__(self) -> None:
        try:
            self.sim.close()
        except Exception:
            pass

    # ------------------------------------------------------------------ low level
    def get_own_udp_object(self) -> dict:
        """{'state': latest state dict, 'responses': replies not read yet} (djitellopy internals)."""
        self._udp["state"] = self.sim.drone.state
        return self._udp

    def _wait_reply(self, timeout: float, until=None) -> str | None:
        t_end = self.sim.clock.now() + timeout
        while self.sim.clock.now() < t_end:
            r = self.sim.drone.responses
            if until is None or until():
                if r:
                    return r.pop(0)
            if self.sim.realtime:
                time.sleep(0.02)
            else:
                self.sim.advance(0.02)
        return None

    def send_command_without_return(self, command: str) -> None:
        reply = self.sim.drone.command(command)
        if reply is not None:
            self.sim.drone.responses.append(reply)

    def send_command_with_return(self, command: str, timeout: int = RESPONSE_TIMEOUT) -> str:
        q = self._read(command)
        if q is not None:
            return q
        self.sim.drone.responses.clear()
        reply = self.sim.drone.command(command)
        if reply is not None:
            return reply
        r = self._wait_reply(timeout)
        return r if r is not None else "Aborting command '{}'. Did not receive a response after {} seconds".format(
            command, timeout)

    def send_control_command(self, command: str, timeout: int = RESPONSE_TIMEOUT) -> bool:
        r = self.send_command_with_return(command, timeout)
        if "ok" in r.lower():
            return True
        raise TelloException(f"Command '{command}' was unsuccessful for {self.retry_count} tries. "
                             f"Latest response:\t'{r}'")

    def send_read_command(self, command: str) -> str:
        return self.send_command_with_return(command)

    def _read(self, command: str) -> str | None:
        s = self.sim.drone.state
        table = {"battery?": str(s["bat"]), "height?": f"{s['h'] // 10}dm", "tof?": f"{s['tof'] * 10}mm",
                 "temp?": f"{s['templ']}~{s['temph']}C", "time?": f"{s['time']}s", "speed?": "10.0",
                 "wifi?": "90", "baro?": str(s["baro"]), "attitude?": f"pitch:{s['pitch']};roll:{s['roll']};yaw:{s['yaw']};",
                 "acceleration?": f"agx:{s['agx']};agy:{s['agy']};agz:{s['agz']};",
                 # a standard Tello (not EDU) answers like this; the lab Tellos did
                 "sdk?": "unknown command: sdk?", "sn?": "unknown command: sn?"}
        return table.get(command)

    # ------------------------------------------------------------------ flying
    def takeoff(self) -> None:
        self.sim.drone.responses.clear()
        reply = self.sim.drone.command("takeoff")
        if reply == "error":
            raise TelloException("Command 'takeoff' was unsuccessful. Latest response:\t'error'")
        r = self._wait_reply(self.TAKEOFF_TIMEOUT)
        if r is None or "ok" not in r:
            raise TelloException("Command 'takeoff' was unsuccessful. Latest response:\t'"
                                 f"{r or 'no response (timeout)'}'")
        self.is_flying = True

    def land(self) -> None:
        self.sim.drone.responses.clear()
        self.sim.drone.command("land")
        self._wait_reply(15, until=lambda: self.sim.drone.mode == "ground")
        self.is_flying = False

    def emergency(self) -> None:
        self.sim.drone.command("emergency")
        self.is_flying = False

    def send_rc_control(self, left_right_velocity: int, forward_backward_velocity: int, up_down_velocity: int,
                        yaw_velocity: int) -> None:
        self.sim.drone.set_rc((left_right_velocity, forward_backward_velocity, up_down_velocity, yaw_velocity))

    def send_keepalive(self) -> None:
        self.send_control_command("keepalive")

    def _blocking_move(self, command: str, timeout: float = 30) -> None:
        self.sim.drone.responses.clear()
        reply = self.sim.drone.command(command)
        if reply == "error":
            raise TelloException(f"Command '{command}' was unsuccessful. Latest response:\t'error'")
        if reply is None:
            r = self._wait_reply(timeout, until=lambda: not self.sim.drone.busy())
            if r is None:
                raise TelloException(f"Command '{command}' timed out")

    def move(self, direction: str, x: int) -> None:
        self._blocking_move(f"{direction} {x}")

    def move_up(self, x: int) -> None:
        self.move("up", x)

    def move_down(self, x: int) -> None:
        self.move("down", x)

    def move_left(self, x: int) -> None:
        self.move("left", x)

    def move_right(self, x: int) -> None:
        self.move("right", x)

    def move_forward(self, x: int) -> None:
        self.move("forward", x)

    def move_back(self, x: int) -> None:
        self.move("back", x)

    def rotate_clockwise(self, x: int) -> None:
        self._blocking_move(f"cw {x}")

    def rotate_counter_clockwise(self, x: int) -> None:
        self._blocking_move(f"ccw {x}")

    def go_xyz_speed(self, x: int, y: int, z: int, speed: int) -> None:
        self._blocking_move(f"go {x} {y} {z} {speed}")

    def set_speed(self, x: int) -> None:
        self.send_control_command(f"speed {x}")

    def flip(self, direction: str) -> None:
        r = self.sim.drone.command(f"flip {direction}")
        if r != "ok":
            raise TelloException(f"Command 'flip {direction}' was unsuccessful. Latest response:\t'{r}'")
        if self.sim.realtime:
            time.sleep(1.0)
        else:
            self.sim.advance(1.0)

    def flip_left(self) -> None:
        self.flip("l")

    def flip_right(self) -> None:
        self.flip("r")

    def flip_forward(self) -> None:
        self.flip("f")

    def flip_back(self) -> None:
        self.flip("b")

    # ------------------------------------------------------------------ state
    def get_current_state(self) -> dict:
        return dict(self.sim.drone.state)

    def get_state_field(self, key: str):
        s = self.sim.drone.state
        if key not in s:
            raise TelloException(f"Could not get state property: {key}")
        return s[key]

    def get_battery(self) -> int:
        return self.get_state_field("bat")

    def get_height(self) -> int:
        """The Tello's own height estimate 'h' (cm). Unreliable on the lab Tellos: use
        get_distance_tof() - 10 for the real height."""
        return self.get_state_field("h")

    def get_distance_tof(self) -> int:
        return self.get_state_field("tof")

    def get_barometer(self) -> float:
        return self.get_state_field("baro") * 100

    def get_temperature(self) -> float:
        return (self.get_state_field("templ") + self.get_state_field("temph")) / 2

    def get_lowest_temperature(self) -> int:
        return self.get_state_field("templ")

    def get_highest_temperature(self) -> int:
        return self.get_state_field("temph")

    def get_flight_time(self) -> int:
        return self.get_state_field("time")

    def get_yaw(self) -> int:
        return self.get_state_field("yaw")

    def get_pitch(self) -> int:
        return self.get_state_field("pitch")

    def get_roll(self) -> int:
        return self.get_state_field("roll")

    def get_speed_x(self) -> int:
        return self.get_state_field("vgx")

    def get_speed_y(self) -> int:
        return self.get_state_field("vgy")

    def get_speed_z(self) -> int:
        return self.get_state_field("vgz")

    def get_acceleration_x(self) -> float:
        return self.get_state_field("agx")

    def get_acceleration_y(self) -> float:
        return self.get_state_field("agy")

    def get_acceleration_z(self) -> float:
        return self.get_state_field("agz")

    def query_battery(self) -> int:
        return int(self.send_read_command("battery?"))

    def query_sdk_version(self) -> str:
        return self.send_read_command("sdk?")

    def query_serial_number(self) -> str:
        return self.send_read_command("sn?")
