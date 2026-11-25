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
```

The gate lines appear on maps with gates, and the line distance on maps with floor lines.
Positions are in cm from the start: x to the east (right at the start), y to the north (ahead
at the start). See [maps.md](maps.md#coordinates).

## What we saw with the group's projects (Nov 2026)

Each one was run unchanged, with `python -m tellosim.run`. These are first runs, not a
verdict: use them to know where to start.

**[Gate_tmr_color](https://github.com/DroneOps/Gate_tmr_color)** has two versions:
`MissionControl.py` (distance from the gate's width in pixels) and `MissionControl_PnP.py`
(PnP + Kalman + PID, their latest commit). We flew both against one gate and against `tmr_gates`.
The one-gate map:

```yaml
# one_gate.yaml: a 110 cm opening (what their code assumes), 5 m ahead, centered at 100 cm
floor: {type: lab_grid}
objects:
  - {type: gate, position: [0, 500, 100], width_cm: 110, height_cm: 110, bar_cm: 10, color: green}
```

```bash
python -m tellosim.run --world one_gate.yaml --record gate.mp4 --seconds 65 MissionControl_PnP.py
```

| Version | One gate, 5 m ahead (2 runs) | `tmr_gates`: three gates in a row (1 run) |
|---|---|---|
| `MissionControl_PnP.py` | **flew through both times**, at most 27 cm off center, sideways rc never above 15 | hit the first gate's frame |
| `MissionControl.py` | hit the bottom bar both times: it flew too low | zigzag with rc up to ±100, hit the first gate |

**What we learned:**
- **The big zigzag on `tmr_gates` comes from the map, not from the drone or its controller.**
  - The three gates are in a row, so from the start they look overlapped and their bars
    cross in the image, making small false rectangles.
  - The detector takes any rectangular hole in the green mask (the last one it finds), so it
    sometimes followed one of those.
  - A false rectangle a few pixels wide made it believe the gate was 16–70 m away and up to
    9 m to the side, so it sent rc ±100.
  - With a single gate it never sent more than rc 19 sideways.
  - If the real course has gates in a row, this can happen for real: keep the biggest hole,
    not the last one.
- **The camera looks 11° down and sees only ~10° above the horizon.**
  - From the takeoff height (~70 cm), a gate centered at 100 cm fits whole only from ~5 m
    away (a 110 cm opening) or ~4.6 m (90 cm).
  - Closer than that, the top bar is out of view, there is no closed rectangle, and both
    versions finish the approach blind ("sprint").
  - On `tmr_gates` the first gate is 2.5 m away, so the real gate was never detected, only
    the false rectangles.
- **`MissionControl.py` flew too low.**
  - It asked ud ~20 while going forward and sideways. The lab Tello climbs only ~5–10 cm/s
    with rc 20 (calibration of 2026-10-24) and sinks when moving forward and sideways, so it
    barely climbed.
  - It lost the gate at ~4.4 m, sank to ~36 cm flying blind and hit the bottom bar.
  - The PnP version asks up to ud 44, reaches the gate's height and keeps seeing it until
    ~2.8 m.
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
`tmr_landing`): it takes off with `t` and lands with `l`.

```bash
python -m tellosim.run --world tmr_landing --keys t@4,l@45 --seconds 55 tests/test_tracking.py
```

- It runs as is: it detects the ArUco and follows it.
- **The signs are right with the lab mirror** (`mirror_axis` `v`, the default). With
  `--camera-set mirror_axis=h` the commands point away from the pad and it flies off. If your
  mirror flips the image the other way, check the signs before flying.
- **Result:** after 38 s of tracking it was 19–34 cm from the pad, and it had flown 2.6–3 m back
  and forth.
- **Why:**
  - a 15 cm error already gives the maximum command (±50);
  - it centers the marker in the image, but with the mirror tilt the image center is ~20 cm
    behind the point below the drone.
- **What to try:**
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
