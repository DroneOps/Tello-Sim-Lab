# Controllers (`tellolab`)

They take readings (the state dict, positions) and return rc commands, so the same code works
with `djitellopy.Tello` and `tellosim.Tello`. The defaults are what we tuned on the lab Tellos.
Every config is a dataclass: change only what you need, e.g.
`AltitudeHold(AltitudeHoldConfig(max_rc=40))`.

Tune them in the simulator first, then check on the real drone, starting with
`examples/hover_hold.py`.

## Pilot: everything together

```python
from tellolab import Pilot, PilotConfig

pilot = Pilot(tello, frame_read=tello.get_frame_read(), log_path="logs/flight.csv")
pilot.takeoff(altitude_cm=100)
pilot.set_rc(lr, fb, ud, yaw)    # 0 on an axis = hold it
pilot.turn_by(90); pilot.set_altitude(120); pilot.flip("b")
pilot.land()
```

**What it does every 1/20 s:**
- **vertical:** your ud, or `AltitudeHold` if you send 0;
- **yaw:** your yaw, or `HeadingHold`;
- **lr / fb:** yours, plus `PositionHold` if you pass `position=` and send 0 on both;
- **`SafetyMonitor`:** lands with the reason in `pilot.reason`.

`threaded=False` and `tick()` are for simulated time.

## AltitudeHold: height with the ToF

**Problem:**
- the Tello sinks by itself when flying forward (lab: 60 → 33 cm, 10–13 cm/s);
- the ToF gives false jumps (81 → 14 → 83 cm in 0.4 s);
- the drone's own `h` is wrong.

**How:**
- a PI on the ToF height (`altitude_cm(state)` = tof − 10);
- filtered: jumps shorter than 0.6 s are ignored, and the height follows real changes at 30 cm/s
  at most;
- with a minimum rc of 20, because smaller rc barely moves the Tello vertically.

```python
hold = AltitudeHold(target_cm=100)
ud = hold.update(altitude_cm(state), time.monotonic(), ud_request=my_ud)   # my_ud != 0: move, then hold there
```

| Parameter | Default | Change it if... |
|---|---|---|
| `kp`, `ki` | 1.2, 0.5 | slow to come back: raise kp; it oscillates up and down: lower kp |
| `i_max` | 15 | the integral is the steady push against sinking; with 40 it got stuck climbing |
| `min_rc`, `max_rc` | 20, 30 | it cannot win against sinking: raise `max_rc` carefully (rc 40 overshot 45 → 89 cm on one drone) |
| `tol_cm` | 4 | |
| `filter` | spike 20 cm / 0.6 s, 30 cm/s | it follows a box step too slowly: raise `max_rate_cm_s` |

**Limits:**
- **Forward and sideways at once,** the simulated lab Tello sinks ~9 cm/s and climbs only ~11
  cm/s with rc 30. The hold wins, but slowly (−16 cm worst case in the tests).
- **Avoid combining both axes for long.** The AprilTag mission centers first, then flies
  straight.

`GoToAltitude`: `ud, done = goto.update(target, alt)`, to reach a new height.

## HeadingHold: heading

**Problem:**
- rc below ~14 barely turns the Tello;
- after a turn it keeps rotating (88° → 101°).

**How:** a P controller with a minimum of rc 20. `turn_to(deg)` and `turn_by(deg, yaw)` use the
faster turning gains (rc up to 60 ≈ 44°/s) until it gets there.

```python
heading = HeadingHold()
heading.turn_by(90, state["yaw"])
yaw_rc = heading.update(state["yaw"])
```

| Parameter | Default | |
|---|---|---|
| `kp`, `min_rc`, `max_rc`, `tol_deg` | 1.0, 20, 30, 3 | holding |
| `turn_kp`, `turn_max_rc` | 2.0, 60 | turning |

## FloorFlow: how far it moved, from a downward mirror

**Problem:** the Tello does not tell you where it is, and its `vgx` / `vgy` are coarse (whole
dm/s).

**How:**
- phase correlation between consecutive mirror frames at 240x180 (~2 ms), turned into cm with
  the ToF height;
- it compensates the floor growing or shrinking when climbing, around the pixel right below
  the drone (`point_below_drone`, which depends on the mirror tilt);
- it pauses while turning.

```python
K, dist = mirrored_intrinsics(K, dist)            # or tello.sim.camera.detection_K
flow = FloorFlow(K[0, 0], center=point_below_drone(K, view_forward_deg=-10.3))
flow.update(gray_flipped_back, t, altitude_cm(state), turning=yaw_changed)
(pos_right, pos_fwd), (vel_right, vel_fwd) = flow.state()
```

It needs texture on the floor: the lab tape grid works, a plain shiny floor does not. In the
simulator it measured 40–70 cm moves within 25 % (tests).

## PositionHold: stay at a point (or on a line)

**Problem:** lab Tello #7 drifts ~12 cm/s to its right and never holds still.

**How:**
- a PID per axis on (right, forward) relative to where it should be, from any source
  (`FloorFlow`, OptiTrack, a marker...);
- the integral learns the constant push against the drift, and `i_start` gives it from takeoff
  (Tello #7: 4.5);
- **safety cutoff:** if correcting makes it move away 35 cm in 1.5 s (a reversed measurement), or
  it ends up 150 cm away, it switches off for the rest of the flight and tells you why.

```python
hold = PositionHold()
lr, fb, why_off = hold.update(pos, vel, now)                  # hover at the point
lr, fb, why_off = hold.update(pos, vel, now, along_track=True)  # flying a leg: only sideways
```

**Results** (simulated Tello #7, 20–30 s hovering): without it, 1.3–4.2 m away; with it, 41–45 cm
at most.

**Symptoms and what to change:**

| Symptom | Change |
|---|---|
| stays off to one side | raise `ki`, or set `i_start` |
| goes back and forth (±10 cm or more) | raise `kd`, lower `kp` |
| slow to come back | raise `kp` (lab candidate: 1.2) |
| switches itself off: "moved away..." | the measurement is reversed (mirror axis? sign?). Never raise `runaway_cm` to make it "pass" |

## TargetCentering: get over or in front of a target and stay

**Problem:** with ~170 ms of video delay and the drone's inertia, a plain P overshoots (lab:
±15–20 cm around a tag).

**How:**
- the command brakes with the target's velocity;
- that velocity is computed without the part caused only by the drone turning (a target 10 cm
  away seemed to move 6 cm/s while turning).

```python
c = TargetCentering()
lr, fb, settled = c.update(target_forward_cm, target_right_cm, t_frame, yaw)
```

`tol_cm` 12 and `vel_tol_cm_s` 8 say when it counts as "over it". `kp` 0.6, `kd` 1.0, rc between
10 and 15. For a precise landing use a stricter config:
`TargetCenteringConfig(tol_cm=5, vel_tol_cm_s=5)`.

To keep a distance instead of being over the target, pass the error:
`c.update(distance - wanted, right, t, yaw)`. `examples/aruco_follow.py` does this.

## SafeFlip

**Problem:**
- the Tello refuses flips below 50 % battery;
- after the flip it falls ~50 cm/s while its 'h' does not notice (117 → 39 cm, once to the
  floor);
- its "ok" reply is not proof that it flipped.

**How:**
1. climb to 130 cm;
2. flip;
3. **brake:** full climb until it stops falling;
4. **hold** there 2 s (descending right away makes it fall again);
5. **settle:** down gently to where it was.

If it falls fast again at any point, it brakes again. The flip is confirmed by pitch/roll going
past 60° and back.

```python
f = SafeFlip()
f.start("b", state)                       # None, or why it won't flip
while not f.finished:
    ud = f.update(state, time.monotonic(), tello.send_command_without_return)
    tello.send_rc_control(0, 0, ud, 0)
print(f.result)
```

In the simulator it never goes below ~45 cm after the flip (test). Adjust `min_alt_cm` (130) if
your ceiling is low, never below ~110: it falls ~45 cm. You need ~1.7 m free above the floor.

## SafetyMonitor

```python
mon = SafetyMonitor()
problems = mon.preflight(state, state_age)          # [] = ok to take off
why = mon.check_takeoff(state)                      # during takeoff: did it collapse?
why = mon.check(state, state_age, now, video_age, ud_sent)   # in the air, every loop
```

**Reasons to refuse a takeoff:**
- battery < 30 %;
- temperature ≥ 80 °C (cool it down: the Tello heats 60 → 85 °C in 5 min on the ground with video
  on).

**Reasons to land:**
- telemetry lost > 1 s, or video lost > 3 s;
- battery < 15 %;
- temperature ≥ 90 °C;
- on the floor without asking for 1 s;
- the drone does not obey: asking ±20 rc for 0.7 s and moving 15 cm the other way.

**Never switch a watchdog off to make a test pass.** If one fires on the real drone, the problem
is the drone:
- recalibrate the IMU and center of gravity in the Tello app;
- check the propellers;
- let it cool down.

## Vision helpers (`tellolab.vision`)

- **`unmirror`, `mirrored_intrinsics`:** for the mirror mounting.
- **`detect_markers(gray, dictionary, ids)`:** pass `ids` to reject false positives (a stray
  ID 29 appeared once on the lab floor).
- **`marker_position(corners, size_cm, K, dist, mode, ...)`:** (forward, right, up) of a marker
  in cm, with solvePnP.
- **`floor_point_from_pixel(u, v, K, dist, height_cm, mode, ...)`:** where a pixel is on the
  floor, in cm. Use it for lines, pads, anything on the floor.

In the simulator they match the true positions within ~1 cm (forward and mirror).
