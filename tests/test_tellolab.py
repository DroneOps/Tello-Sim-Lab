"""Tests of the controllers against the simulated lab Tellos (simulated time: fast)."""
import math

import cv2
import numpy as np
import pytest

from tellolab import (AltitudeFilter, AltitudeHold, FloorFlow, HeadingHold, Pilot, PositionHold, SafeFlip,
                      SafetyMonitor, TargetCentering, altitude_cm, point_below_drone)
from tellosim import Faults, Tello, load_map, load_model
from tellosim.drone import SimClock, SimDrone

HZ = 20


def airborne(model="lab_healthy", seed=1, **kw):
    clk = SimClock()
    d = SimDrone(load_map(kw.pop("world", "lab_grid")), load_model(model), seed=seed, clock=clk, **kw)
    d.command("takeoff")
    step(d, clk, 9)
    assert d.mode == "air"
    return d, clk


def step(d, clk, seconds, rc=(0, 0, 0, 0)):
    d.set_rc(rc, clk.t)
    for _ in range(int(round(seconds / 0.02))):
        clk.t += 0.02
        d.step(0.02, clk.t)


def loop(d, clk, seconds, control):
    """Run a controller at 20 Hz: control(state, now) -> rc tuple."""
    for _ in range(int(seconds * HZ)):
        step(d, clk, 1 / HZ, control(d.state, clk.t))


# ----------------------------------------------------------------------------- altitude
def test_altitude_filter_ignores_spikes():
    f = AltitudeFilter()
    t = 0.0
    for alt in [80] * 10 + [14] * 3 + [80] * 5:  # lab: 81 -> 14 -> 83 in 0.4 s
        t += 0.1
        out = f.update(alt, t)
    assert out == pytest.approx(80, abs=1)


def test_altitude_hold_against_sinking():
    def fly(with_hold):
        d, clk = airborne()
        hold = AltitudeHold(target_cm=altitude_cm(d.state))
        start = altitude_cm(d.state)
        alts = []

        def control(s, now):
            alts.append(altitude_cm(s))
            ud = hold.update(altitude_cm(s), now) if with_hold else 0
            return (15, 15, ud, 0)  # forward AND sideways: the Tello sinks ~9 cm/s (lab: 8)
        loop(d, clk, 8, control)
        return start - min(alts[20:]), start - alts[-1]
    worst_free, end_free = fly(with_hold=False)
    worst_hold, end_hold = fly(with_hold=True)
    assert worst_free > 40
    # the Tello climbs slowly (rc 30 ≈ 11 cm/s): the hold wins, but slowly
    assert worst_hold < worst_free / 2 and end_hold < 14


# ----------------------------------------------------------------------------- heading
def test_heading_turn_and_hold():
    d, clk = airborne()
    h = HeadingHold()
    h.turn_by(90, d.state["yaw"])
    target = h.target_deg
    loop(d, clk, 8, lambda s, now: (0, 0, 0, h.update(s["yaw"])))
    assert abs(((d.yaw - target) + 180) % 360 - 180) < 6


# ----------------------------------------------------------------------------- position
def test_position_hold_against_tello7_drift():
    def fly(with_hold):
        d, clk = airborne("lab_tello7", seed=4)
        x0, y0 = d.x, d.y
        hold = PositionHold()
        worst = 0.0

        def control(s, now):
            nonlocal worst
            ps = math.radians(d.yaw)  # true pose -> drone frame (right, forward), relative to start
            dx, dy = d.x - x0, d.y - y0
            pos = (dx * math.cos(ps) - dy * math.sin(ps), dx * math.sin(ps) + dy * math.cos(ps))
            vw = d.v_world
            vel = (vw[0] * math.cos(ps) - vw[1] * math.sin(ps), vw[0] * math.sin(ps) + vw[1] * math.cos(ps))
            worst = max(worst, math.hypot(dx, dy))
            if not with_hold:
                return (0, 0, 0, 0)
            lr, fb, _ = hold.update(pos, vel, now)
            return (lr, fb, 0, 0)
        loop(d, clk, 20, control)
        return worst
    assert fly(with_hold=False) > 100  # Tello 7 alone: 1–3 m in 20 s
    assert fly(with_hold=True) < 45


def test_position_hold_cuts_off_when_measurement_is_reversed():
    d, clk = airborne("lab_tello7", seed=5)
    x0 = d.x
    hold = PositionHold()
    why = None

    def control(s, now):
        nonlocal why
        right = -(d.x - x0)  # reversed sign: the correction pushes it away
        lr, fb, why = hold.update((right, 0.0), (-d.v_world[0], 0.0), now)
        return (lr, fb, 0, 0)
    loop(d, clk, 15, control)
    assert why is not None


# ----------------------------------------------------------------------------- floor flow
def test_floor_flow_measures_motion_through_mirror():
    t = Tello(realtime=False, camera_mode="mirror", model="ideal", seed=2)
    t.connect()
    t.streamon()
    t.takeoff()
    cam = t.sim.camera
    K, _ = cam.detection_K
    flow = FloorFlow(K[0, 0], center=point_below_drone(K, cam.mirror_tilt_deg))
    y0 = t.sim.pose()[1]
    t.send_rc_control(0, 25, 0, 0)
    for _ in range(4 * 30):  # 4 s at 30 frames/s
        t.sim.advance(1 / 30)
        img = cv2.flip(t.sim.capture()[..., ::-1], 0)
        flow.update(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), t.sim.clock.now(), altitude_cm(t.get_current_state()))
    moved = t.sim.pose()[1] - y0
    measured = flow.state()[0][1]
    assert moved > 40
    assert measured == pytest.approx(moved, rel=0.25)


# ----------------------------------------------------------------------------- target centering
def test_target_centering_gets_over_a_point():
    d, clk = airborne("lab_tello1", seed=6)
    tx, ty = d.x + 30, d.y + 25  # a target 30 cm right and 25 ahead (heading north)
    c = TargetCentering()
    settled_at = None

    def control(s, now):
        nonlocal settled_at
        lr, fb, settled = c.update(ty - d.y, tx - d.x, now, s["yaw"])
        if settled and settled_at is None:
            settled_at = now
        return (lr, fb, 0, 0)
    loop(d, clk, 10, control)
    assert settled_at is not None
    assert math.hypot(tx - d.x, ty - d.y) < 15


# ----------------------------------------------------------------------------- flip
def test_safe_flip_does_not_hit_the_floor():
    d, clk = airborne(battery=95)
    f = SafeFlip()
    assert f.start("b", d.state, clk.t) is None
    lowest = math.inf
    for _ in range(30 * HZ):
        ud = f.update(d.state, clk.t, d.command)
        step(d, clk, 1 / HZ, (0, 0, ud, 0))
        if f.phase not in ("climb", "flip"):
            lowest = min(lowest, d.z)
        if f.finished:
            break
    assert f.phase == "done" and "done" in f.result
    assert lowest > 45
    assert abs(altitude_cm(d.state) - f.return_alt) < 12


def test_safe_flip_refused_with_low_battery():
    d, clk = airborne(battery=45)
    f = SafeFlip()
    assert "battery" in f.start("b", d.state, clk.t)


# ----------------------------------------------------------------------------- safety
def test_safety_catches_collapsing_takeoff():
    clk = SimClock()
    d = SimDrone(load_map("lab_grid"), load_model("lab_healthy"), clock=clk, faults=Faults(takeoff="collapse"))
    mon = SafetyMonitor()
    d.command("takeoff")
    why = None
    for _ in range(8 * HZ):
        step(d, clk, 1 / HZ)
        why = why or mon.check_takeoff(d.state)
    assert why and "collapsed" in why


def test_safety_catches_a_drone_that_does_not_obey():
    d, clk = airborne(faults=Faults(imu_bad=(12.0, 40.0, -35.0)))
    mon = SafetyMonitor()
    why = None
    for _ in range(6 * HZ):
        step(d, clk, 1 / HZ, (0, 0, 40, 0))  # asking to climb at full power...
        why = why or mon.check(d.state, 0.0, clk.t, ud_sent=40)
    assert why and "does not obey" in why  # ...while it goes down


def test_safety_preflight():
    mon = SafetyMonitor()
    assert mon.preflight({"bat": 90, "temph": 65}, 0.1) == []
    problems = mon.preflight({"bat": 20, "temph": 85}, 0.1)
    assert any("battery" in p for p in problems) and any("temperature" in p for p in problems)


# ----------------------------------------------------------------------------- pilot
def test_pilot_holds_altitude_and_lands():
    t = Tello(realtime=False, seed=7)
    t.connect()
    t.streamon()
    pilot = Pilot(t, threaded=False, clock=t.sim.clock.now)
    pilot.takeoff(altitude_cm=100)
    pilot._wait(8)  # climb to 100 and hold
    assert altitude_cm(t.get_current_state()) == pytest.approx(100, abs=10)
    pilot.set_rc(15, 15, 0, 0)  # the combination that sinks the Tello
    pilot._wait(6)
    pilot.hover()
    pilot._wait(2)
    assert altitude_cm(t.get_current_state()) == pytest.approx(100, abs=12)
    assert pilot.reason is None
    pilot.land()
    assert t.sim.drone.mode == "ground"
