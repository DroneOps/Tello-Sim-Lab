"""Tests of the simulator. They use simulated time (realtime=False), so they run in seconds."""
import math

import cv2
import numpy as np
import pytest

import tellosim
from tellosim import AprilTag, Box, Faults, Tello, TelloException, World, load_map, load_model
from tellosim.drone import SimClock, SimDrone


def detect(img_bgr, dictionary):
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
    _, ids, _ = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters()).detectMarkers(
        cv2.cvtColor(np.ascontiguousarray(img_bgr), cv2.COLOR_BGR2GRAY))
    return [] if ids is None else sorted(int(i) for i in ids.ravel())


def place(tello, x, y, z, yaw=0.0):
    """Teleport the simulated drone (tests only)."""
    d = tello.sim.drone
    with d.lock:
        d.x, d.y, d.z, d.yaw = x, y, z, yaw
        d.history.clear()
        d.history.append((tello.sim.clock.now(), x, y, z, yaw))


@pytest.fixture
def flying():
    def make(**kw):
        t = Tello(realtime=False, seed=kw.pop("seed", 1), **kw)
        t.connect()
        t.streamon()
        t.takeoff()
        return t
    return make


# ----------------------------------------------------------------------------- models and maps
def test_models_load_and_inherit():
    healthy, t7 = load_model("lab_healthy"), load_model("lab_tello7")
    assert t7.drift_right_cm_s == pytest.approx(11.7)
    assert t7.forward_gain == healthy.forward_gain  # inherited from the base
    with pytest.raises(ValueError):
        load_model({"no_such_field": 1})


def test_builtin_maps_load():
    for name in tellosim.builtin_maps():
        w = load_map(name)
        assert w.texture.ndim == 3
    with pytest.raises(ValueError):
        load_map({"objects": [{"type": "spaceship"}]})


# ----------------------------------------------------------------------------- physics
def run(drone, clock, seconds, rc=(0, 0, 0, 0)):
    drone.set_rc(rc, clock.t)
    for _ in range(int(round(seconds / 0.02))):
        clock.t += 0.02
        drone.step(0.02, clock.t)


def test_takeoff_height_and_ok():
    for seed in range(5):
        clk = SimClock()
        d = SimDrone(load_map("lab_grid"), load_model("lab_healthy"), seed=seed, clock=clk)
        d.command("takeoff")
        run(d, clk, 10)
        assert d.mode == "air" and "ok" in d.responses
        assert 64 <= d.z <= 75  # lab: settles between 65 and 74 cm


def test_rc_speed_matches_model():
    clk = SimClock()
    m = load_model("ideal")
    d = SimDrone(load_map("lab_grid"), m, seed=0, clock=clk)
    d.command("takeoff")
    run(d, clk, 5)
    run(d, clk, 2, (0, 15, 0, 0))  # reach the steady speed
    y0 = d.y
    run(d, clk, 4, (0, 15, 0, 0))
    assert (d.y - y0) / 4 == pytest.approx(m.forward_gain * (15 - m.deadband), rel=0.05)


def test_flip_falls_unless_climbing():
    def flip(climb):
        clk = SimClock()
        d = SimDrone(load_map("lab_grid"), load_model("lab_healthy"), seed=2, clock=clk)
        d.command("takeoff")
        run(d, clk, 9)
        run(d, clk, 6, (0, 0, 60, 0))  # climb before flipping, like a safe flip
        z0, lowest = d.z, d.z
        assert d.command("flip b") == "ok"
        d.set_rc((0, 0, 100 if climb else 0, 0), clk.t)
        for _ in range(200):  # 4 s
            clk.t += 0.02
            d.step(0.02, clk.t)
            lowest = min(lowest, d.z)
        return z0 - lowest
    assert flip(climb=False) > flip(climb=True) + 20


def test_flip_refused_with_low_battery():
    clk = SimClock()
    d = SimDrone(load_map("lab_grid"), load_model("lab_healthy"), clock=clk, battery=40)
    d.command("takeoff")
    run(d, clk, 9)
    assert d.command("flip f") == "error"


def test_box_under_drone_changes_tof_and_crash():
    w = World(objects=[Box(position=(0, 60), size=(60, 60, 40))])
    clk = SimClock()
    d = SimDrone(w, load_model("ideal"), clock=clk)
    d.command("takeoff")
    run(d, clk, 5)
    tof_free = d.state["tof"]
    d.x, d.y = 0.0, 60.0
    run(d, clk, 0.5)
    assert d.state["tof"] < tof_free - 25  # the ToF sees the top of the box
    d.x, d.y, d.z = 0.0, -5.0, 20.0  # low, in front of the box...
    run(d, clk, 0.1)
    assert d.crash is None
    d.y = 25.0  # ...and into its side
    run(d, clk, 0.1)
    assert d.crash and "box" in d.crash


# ----------------------------------------------------------------------------- djitellopy interface
def test_api_like_djitellopy(flying):
    t = flying()
    assert t.is_flying
    frame = t.get_frame_read().frame
    assert frame.shape == (720, 960, 3) and frame.dtype == np.uint8
    for getter in ("get_battery", "get_height", "get_distance_tof", "get_yaw", "get_speed_x",
                   "get_temperature", "get_flight_time"):
        assert isinstance(getattr(t, getter)(), (int, float))
    x0, y0, z0, _ = t.sim.pose()
    t.move_forward(50)
    x1, y1, _, _ = t.sim.pose()
    assert math.hypot(x1 - x0, y1 - y0) == pytest.approx(50, abs=12)
    yaw0 = t.get_yaw()
    t.rotate_clockwise(90)
    t.sim.advance(0.3)
    assert ((t.get_yaw() - yaw0) % 360) == pytest.approx(90, abs=6)
    t.land()
    assert not t.is_flying and t.sim.drone.mode == "ground"
    t.end()


def test_frames_are_rgb():
    w = World(objects=[tellosim.ColorPanel(position=(0, 150, 60), width_cm=200, height_cm=200, color="red",
                                           facing_deg=180)])
    t = Tello(world=w, realtime=False)
    t.connect()
    t.streamon()
    t.takeoff()
    f = t.get_frame_read().frame.astype(int)
    center = f[300:420, 400:560].reshape(-1, 3).mean(axis=0)
    assert center[0] > center[2] + 80  # red in the R channel: RGB like djitellopy


def test_takeoff_without_ok_raises():
    t = Tello(realtime=False, faults=Faults(takeoff="no_ok"))
    t.connect()
    with pytest.raises(TelloException):
        t.takeoff()


def test_standard_tello_answers():
    t = Tello(realtime=False)
    assert t.query_sdk_version().startswith("unknown command")  # standard Tello, like the lab's


# ----------------------------------------------------------------------------- rendering
def test_apriltag_seen_through_mirror(flying):
    t = flying(world="apriltag_course")
    assert t.sim.camera.mode == "mirror"  # the map asks for it
    place(t, 0, 78, 95)
    img = cv2.flip(t.sim.capture()[..., ::-1], 0)  # flip back what the mirror flips
    assert detect(img, "DICT_APRILTAG_25h9") == [1]
    raw = t.sim.capture()[..., ::-1]
    assert detect(raw, "DICT_APRILTAG_25h9") == []  # mirrored tags do not decode


def test_worn_tag_still_rendered():
    w = World(objects=[AprilTag(id=3, position=(0, 0), wear="lab")])
    t = Tello(world=w, realtime=False, camera_mode="mirror")
    place(t, 0, -2, 90)
    img = cv2.flip(t.sim.capture()[..., ::-1], 0)
    assert detect(img, "DICT_APRILTAG_25h9") == [3]


def test_aruco_on_wall_forward_camera(flying):
    t = flying(world="aruco_board")
    place(t, 0, 60, 110)
    assert set(detect(t.sim.capture()[..., ::-1], "DICT_4X4_50")) >= {0, 1, 2, 3}


def test_line_visible_from_mirror():
    t = Tello(world="line_follower", realtime=False, camera_mode="mirror")
    place(t, 0, 50, 80)  # above the straight part of the line
    gray = cv2.cvtColor(t.sim.capture()[..., ::-1], cv2.COLOR_BGR2GRAY)
    dark = (gray < 70).mean()
    assert 0.02 < dark < 0.3


def test_gate_color_visible(flying):
    t = flying(world="color_gates")
    place(t, 0, 0, 100)
    hsv = cv2.cvtColor(t.sim.capture()[..., ::-1], cv2.COLOR_BGR2HSV)
    orange = cv2.inRange(hsv, (5, 120, 120), (25, 255, 255))
    assert orange.mean() / 255 > 0.01


def test_gate_leg_color():
    from tellosim.objects import Gate, bgr
    polys = Gate(position=(0, 300, 100), color="green", leg_color="gray").primitives()
    assert [q.color for q in polys[-2:]] == [bgr("gray")] * 2
    assert polys[0].color == bgr("green")
