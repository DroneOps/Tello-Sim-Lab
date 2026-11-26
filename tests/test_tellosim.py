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


# ----------------------------------------------------------------------------- moving objects, fog, map checks
def test_motion_back_and_forth_at_constant_speed():
    from tellosim.objects import Motion
    m = Motion(direction_deg=90, distance_cm=200, speed_cm_s=10, start_s=10)
    assert np.allclose(m.offset(5), 0) and np.allclose(m.offset(15), (50, 0), atol=1e-6)
    assert np.allclose(m.offset(20), (100, 0), atol=1e-6) and np.allclose(m.offset(40), (-100, 0), atol=1e-6)
    assert np.allclose(m.velocity(17), (10, 0), atol=1e-6) and np.allclose(m.velocity(23), (-10, 0), atol=1e-6)
    once = Motion(direction_deg=0, distance_cm=50, speed_cm_s=5, back_and_forth=False)
    assert np.allclose(once.offset(5), (0, 25), atol=1e-6) and np.allclose(once.offset(60), (0, 50), atol=1e-6)


def test_moving_pad_is_seen_where_it_is():
    t = Tello(world="tmr_landing", realtime=False)
    r = t.sim.renderer

    def marker_px(x, time):
        img = cv2.flip(r.render(x, -20, 80, 0, t=time), 0)
        found = [c for i, c in tellolab_detect(img) if i == 0]
        return found[0].mean(axis=0) if found else None
    still = marker_px(0, 0.0)
    assert still is not None
    assert marker_px(0, 15.0) is None  # 15 s: it moved 50 cm to the right, out of view
    moved = marker_px(50, 15.0)  # ...and it is there
    assert moved is not None and np.linalg.norm(moved - still) < 3


def tellolab_detect(img):
    from tellolab.vision import detect_markers
    return detect_markers(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), "DICT_6X6_100")


def test_landed_drone_rides_a_moving_platform():
    from tellosim import LandingPad
    w = World(objects=[LandingPad(position=(0, 100), size_cm=60, height_cm=15,
                                  motion={"direction_deg": 90, "distance_cm": 400, "speed_cm_s": 10})])
    clk = SimClock()
    d = SimDrone(w, load_model("ideal"), clock=clk)
    d.command("takeoff")
    run(d, clk, 6)
    d.x, d.y = w.offset(w.objects[0], clk.t - d.t0)[0], 100.0  # over the pad
    d.command("land")
    run(d, clk, 4)
    assert d.mode == "ground" and abs(d.z - 15) < 1  # on top of it
    x0 = d.x
    run(d, clk, 3)
    assert 25 < d.x - x0 < 35  # 3 s at 10 cm/s, on the pad
    assert d.crash is None


def test_fog_hides_far_things_more():
    w = load_map({"objects": [{"type": "panel", "position": [-60, 150, 80], "width_cm": 60, "height_cm": 60,
                               "color": "black"},
                              {"type": "panel", "position": [60, 600, 80], "width_cm": 60, "height_cm": 60,
                                "color": "black"}],
                  "fog": {"visibility_cm": 500, "color": [200, 200, 200]}})
    t = Tello(world=w, realtime=False, noise_std=0, blur_sigma=0)
    cam = t.sim.camera
    img = t.sim.renderer.render(0, 0, 80, 0)

    def px_of(p):
        r, d, f = cam.basis(0)
        pc = (np.array(p) - np.array([0, 0, 80])) @ np.stack([r, d, f]).T
        u, v = cam.project(pc[None])[0]
        return img[int(v), int(u)].astype(int)
    near, far = px_of((-60, 150, 80)), px_of((60, 600, 80))
    assert near.mean() < far.mean() < 200  # both fade towards the fog, the far one more
    assert abs(near.mean() - 200 * (1 - math.exp(-3 * 162 / 500))) < 15  # exp(-3 d / visibility)
    clear = Tello(world=w, realtime=False, fog=False, noise_std=0, blur_sigma=0)
    assert clear.sim.world.fog is None


def test_round_pad_has_no_square():
    from tellosim import LandingPad
    p = LandingPad(position=(0, 0), size_cm=60, shape="circle", height_cm=20)
    img = p._image()
    assert img.shape[2] == 4 and img[2, 2, 3] == 0 and img[img.shape[0] // 2, img.shape[1] // 2, 3] == 255
    corner = np.array([25.0, 25.0, 10.0])  # inside the square, outside the circle
    assert p.collides(corner, 3) is None and p.top_height(25, 25) == 0
    assert p.collides(np.array([0.0, 0.0, 10.0]), 3) is not None


def test_builtin_maps_pass_the_checks():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("check_maps", Path(__file__).parents[1] / "tools" / "check_maps.py")
    cm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cm)
    for name in tellosim.builtin_maps():
        rep = cm.check_map(name, None)
        assert rep.fails == 0, "\n".join([name] + rep.lines)


def test_motion_without_pause_is_the_same_triangle_wave():
    """pause_s=0 (the default) gives exactly the motion maps had before pause_s existed."""
    from tellosim.objects import Motion
    m = Motion(direction_deg=90, distance_cm=200, speed_cm_s=10, start_s=10)
    L = 200.0
    for t in np.arange(0, 120, 0.37):
        s = max(0.0, t - 10) * 10
        phase = (s + L / 2) % (2 * L)
        old = (phase - L / 2) if phase < L else (1.5 * L - phase)
        assert abs(m.offset(t)[0] - (old if t > 10 else 0.0)) < 1e-6


def test_motion_pauses_at_each_end():
    from tellosim.objects import Motion
    # 45 cm at 15 cm/s = 3 s per run, 1 s stopped at each end, starting at the center going east
    m = Motion(direction_deg=90, distance_cm=45, speed_cm_s=15, start_s=0, pause_s=1)
    assert abs(m.offset(1.5)[0] - 22.5) < 1e-6          # reached the east end after half a run
    assert abs(m.offset(2.0)[0] - 22.5) < 1e-6 and m.velocity(2.0)[0] == 0   # waiting there
    assert m.velocity(2.6)[0] < 0                          # going back west after the pause
    assert abs(m.offset(5.5)[0] + 22.5) < 1e-6           # west end: 1.5 + 1 + 3 s
    assert m.velocity(6.0)[0] == 0 and m.velocity(6.6)[0] > 0
    xs = [m.offset(t)[0] for t in np.arange(0, 40, 0.05)]
    assert abs(max(xs) - 22.5) < 1e-6 and abs(min(xs) + 22.5) < 1e-6


# ---------------------------------------------------------------------------------- glare
def _glare_world(lamp_xy, gloss=0.7):
    from tellosim.objects import Marker
    from tellosim.world import Light, World
    w = World(objects=[Marker(id=0, dictionary="DICT_6X6_100", size_cm=20, gloss=gloss)])
    w.lights = [Light(position=(lamp_xy[0], lamp_xy[1], 270), size_cm=30)]
    return w


def test_no_glare_without_gloss():
    """Lamps alone change nothing: only glossy surfaces reflect them (old maps look the same)."""
    from tellosim.camera import Camera
    from tellosim.render import Renderer
    cam = Camera.load("lab_tello", mode="mirror", noise_std=0.0)
    plain = Renderer(_glare_world((0, 0), gloss=None), cam).render(0, 15, 60, 0)
    lit = _glare_world((0, 0), gloss=None)
    assert np.array_equal(plain, Renderer(lit, cam).render(0, 15, 60, 0))


def test_glare_appears_where_the_lamp_is_mirrored():
    from tellosim.camera import Camera
    from tellosim.render import Renderer
    cam = Camera.load("lab_tello", mode="mirror", noise_std=0.0)
    matte = Renderer(_glare_world((0, 0), gloss=None), cam).render(0, 0, 60, 0).astype(int)
    above = Renderer(_glare_world((0, 0)), cam).render(0, 0, 60, 0).astype(int)
    far = Renderer(_glare_world((600, 0)), cam).render(0, 0, 60, 0).astype(int)
    brighter = (above - matte).max(axis=2) > 40
    assert brighter.sum() > 200                              # a lamp above: a bright patch on the marker
    assert (above >= matte - 2).all()                        # it only adds light
    assert ((far - matte).max(axis=2) > 40).sum() == 0       # a lamp far away is not mirrored into the camera


def test_glare_in_maps_and_apply_glare():
    from tellosim.world import apply_glare, load_map
    w = load_map({"objects": [{"type": "landing_pad", "marker": {"dictionary": "DICT_6X6_100", "id": 0,
                                                                  "size_cm": 10, "gloss": 0.8}}],
                  "lights": [{"position": [0, 0, 250], "size_cm": 20}]})
    assert len(w.lights) == 1 and w.objects[0].glossy_surfaces()[0][1] == 0.8
    with pytest.raises(ValueError):
        load_map({"lights": [{"position": [0, 0, 250], "colour": "white"}]})
    w = load_map("tmr_landing")
    assert not w.lights and not w.objects[0].glossy_surfaces()
    apply_glare(w, {"strength": 0.6, "lamp_height_cm": 300, "lamp_spacing_cm": 200})
    assert w.lights and all(L.position[2] == 300 for L in w.lights)
    assert w.objects[0].glossy_surfaces()[0][1] == 0.6
