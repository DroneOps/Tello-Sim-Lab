# Your existing code, in the simulator

There are two ways. Neither needs the real drone or its WiFi.

## 1. Change one import

```python
# from djitellopy import Tello
from tellosim import Tello
```

Then pass simulator options where you create it, if you want:
`Tello(world="tmr_gates", model="lab_tello7", seed=2)`.

## 2. Don't change anything: `python -m tellosim.run`

```bash
python -m tellosim.run --world tmr_gates my_program.py
```

- **How it works:** while your program runs, `import djitellopy` gets the simulator. Your files
  stay untouched, so the same program then flies the real drone.
- **Your program's own arguments** go after its name:
  `python -m tellosim.run my_program.py --speed 20`.
- **Imports of your other files** work as usual: the program's folder is on the path. If it
  imports from its parent folder (as `tests/` scripts often do), run it from the project's root
  folder.

| Option | What |
|---|---|
| `--world`, `--model`, `--camera`, `--camera-mode`, `--seed` | the map, the drone, the camera ([maps.md](maps.md), [drone-model.md](drone-model.md), [camera.md](camera.md)) |
| `--camera-set KEY=VALUE` | any camera field, e.g. `--camera-set mirror_axis=h --camera-set mirror_tilt_deg=-5`; repeat it |
| `--faults JSON` | e.g. `--faults '{"takeoff": "collapse"}'` ([faults.md](faults.md)) |
| `--fog N` or `--fog JSON` | fog with N cm of visibility, or smoke: `--fog '{"visibility_cm": 300, "patches": 0.5}'` ([maps.md](maps.md#fog-and-smoke)) |
| `--viewer` | also show the map from above |
| `--record run.mp4` | record the map and the camera, side by side |
| `--headless` | no windows: `cv2.imshow` does nothing, `cv2.waitKey` returns -1 |
| `--keys t@4,l@40` | press keys for your program: `cv2.waitKey` returns `t` at 4 s and `l` at 40 s |
| `--seconds N` | stop the program after N s, as if you pressed Ctrl+C |

**Keyboard programs:**
- **Without `--headless`,** your OpenCV windows appear and your keys work as always (click the
  window first).
- **With `--keys`,** a program that waits for `t` to take off runs alone, and the same command
  gives the same flight every time. Only for keys read with `cv2.waitKey` / `cv2.pollKey`;
  programs that read the keyboard another way (pygame, `keyboard`, Qt) need you at the keys.

**At the end it says what really happened,** which the real drone never tells you:

```text
[tellosim.run] flight time 42 s, battery 86%, crash: none
[tellosim.run] flew 257 cm; ended at x=24 y=4 z=0 cm, yaw 0; highest 93 cm
[tellosim.run] gate gate1: flown through
[tellosim.run] distance to the line: median 6 cm, max 26 cm
[tellosim.run] landing pad pad: touched down 64 cm from its center (off it), at 61 s
```

The gate lines appear on maps with gates, the line distance on maps with floor lines, and the
landing on maps with pads (where the pad was when the drone touched down, if it moves).
Positions are in cm from the start: x to the east (right at the start), y to the north (ahead
at the start). See [maps.md](maps.md#coordinates).

## What we saw with the group's projects (Nov 2026)

Each one was run unchanged, with `python -m tellosim.run`. These are first runs, not a
verdict: use them to know where to start.

**[Gate_tmr_color](https://github.com/DroneOps/Gate_tmr_color)** has two versions:
`MissionControl.py` (distance from the gate's width in pixels) and `MissionControl_PnP.py`
(PnP + Kalman + PID, their latest commit). We flew both on `tmr_gates`: three 110 cm gates,
5 m apart, the first 5 m from the start, centered at 90–95 cm and up to 10 cm off the line.

```bash
python -m tellosim.run --world tmr_gates --record gates.mp4 --seconds 110 MissionControl_PnP.py
```

| Version | `tmr_gates` |
|---|---|
| `MissionControl_PnP.py` (3 runs) | two runs went through **all three**. The other went through the first, drifted ~45 cm to the side flying blind to the second, and hit its post |
| `MissionControl.py` (1 run) | went through the first, but flew low (~50 cm) and slowly: it did not reach the second in 110 s |

Neither zigzagged: the sideways rc stayed at ±15 (once 26).

**What we learned:**
- **The camera looks 11° down and sees only ~10° above the horizon.**
  - From the takeoff height (~70 cm), a gate centered at ~90 cm fits whole only from ~5 m
    away.
  - Closer than that, the top bar is out of view and there is no closed rectangle, so both
    versions finish each approach blind ("sprint": lr = 0, ud = 0).
  - In those last ~4 m nothing corrects the drift or the sinking. That is what made the
    second PnP run hit the post.
- **Several gates in view can hide the nearest one.**
  - The next gates are seen inside the nearest one.
  - If their frames touch in the image, the nearest gate's hole stops being a rectangle and
    neither version finds it.
  - With the next gate 30 cm off the line, or a gate 15 cm higher than the one in front, the
    PnP version did not see the second gate from 5 m (it did from 4 m). That is why this map
    keeps them within 10 cm.
  - The first version of this map (first gate at 2.5 m, 30 cm offsets) was worse. The
    frames crossed and made small false rectangles, and `MissionControl.py`, which takes any
    rectangular hole (the last one it finds), followed them. It believed the gate was 16–70
    m away and up to 9 m to the side, sent rc ±100 and zigzagged.
  - On a real course with gates in a row this can happen too: keep the biggest hole, and
    give the gates a stand that is not green (`leg_color` in the map).
- **`MissionControl.py` flies too low.**
  - It asks ud ~20–25 while going forward and sideways. The lab Tello climbs only ~5–10 cm/s
    with rc 20 (calibration of 2026-10-24) and sinks when moving forward and sideways, so it
    barely climbs.
  - The PnP version asks up to ud 44 and reaches the gate's height.
- **Two details in `Control_Gate_PnP.py`:**
  - `derivada_z` subtracts `self.errorPrev[0]`, which is the previous *sideways* error
    (`errorPrev` is (y, z, distance)): the vertical D term mixes in the sideways error. It
    should be `errorPrev[1]`.
  - The Kalman filters take `get_speed_*` as cm/s, but the Tello gives dm/s (and `vgz` is
    positive going down), so the prediction moves 10x too little. It matters little: the
    filter mostly follows the camera.
- **`scale_signal` turns any command into at least ±15.** Near the center it wobbles at ±15
  instead of stopping: a small side-to-side wobble, on the real drone too.

**[TMR_Landing](https://github.com/DroneOps/TMR_Landing)** (`tests/test_tracking.py`, map
`tmr_landing`): it takes off with `t` and lands with `l`. On this map the pad starts moving 10 s
after the start: left and right at 10 cm/s, 1 m to each side.

```bash
python -m tellosim.run --world tmr_landing --keys t@4,l@60 --seconds 70 tests/test_tracking.py
python -m tellosim.run --world tmr_landing --keys t@4,l@60 --seconds 70 --fog '{"visibility_cm": 300, "patches": 0.5}' tests/test_tracking.py
```

- It runs as is: it detects the ArUco and follows it.
- **The signs are right with the lab mirror** (`mirror_axis` `v`, the default). With
  `--camera-set mirror_axis=h` the commands point away from the pad and it flies off. If your
  mirror flips the image the other way, check the signs before flying.
- **With the pad still** (first version of the map): after 38 s of tracking it was 19–34 cm
  from the pad, and it had flown 2.6–3 m back and forth.
  - a 15 cm error already gives the maximum command (±50);
  - it centers the marker in the image, but with the mirror tilt the image center is ~20 cm
    behind the point below the drone.
- **With the pad moving** (2 runs, with and without smoke of 3 m visibility: the same flight):
  - it follows the pad, 15–20 cm from it, including when the pad turns around;
  - but it sends ud = 0 while chasing, and the lab Tello sinks when it moves sideways and
    forward (5–7 cm/s measured). In 18 s it went down from 92 to 27 cm;
  - at 27 cm the mirror sees only a small patch of floor: it lost the pad at ~30 s and
    hovered there; when `l` came, it touched down 64 cm from the pad.
- **Fog and smoke** (still frames, its own detector): 
  - at 90 cm high it still sees the marker with 1.5 m of visibility, and loses it below that
    (with drifting smoke, below 2 m);
  - at 130 cm it needs 2 m of visibility.
- **What to try:**
  - hold the height while tracking (`tellolab.AltitudeHold`, or a P on the ToF);
  - lower gains or a lower maximum;
  - aim at the point below the drone (`tellolab.point_below_drone`, or
    `tellolab.vision.floor_point_from_pixel`).
- `main.py` never takes off: its keyboard part is left out ("omitted for brevity").

**[Line_Follower](https://github.com/DroneOps/Line_Follower)** was not run here: its window uses
PySide6 and pyvisual. Every Tello call it makes is in `tellosim.Tello`, so it should run after
installing them. Try it on the `line_follower` map.

## When something is missing

`tellosim.Tello` has the djitellopy calls the group's projects use, and more:
- connection and video;
- takeoff and land, rc;
- `move_*`, `rotate_*`, flips, `go_xyz_speed`;
- every `get_*` reading;
- `send_command_with_return` / `send_command_without_return`.

If your program uses a call it does not have, you get an `AttributeError` with its name. Open an
issue or add it in `tellosim/tello.py`: most are one line over `SimDrone.command()` or
`get_state_field()`.
