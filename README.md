# Tello-Sim-Lab

A **DJI Tello simulator tuned to real flights of the DroneOps lab Tellos**, and the
**controllers that keep a Tello stable**, both extracted from
[DroneOps/AprilTag-Detection](https://github.com/DroneOps/AprilTag-Detection) and generalized
for any Tello project.

- **`tellosim`**: a drop-in replacement for `djitellopy.Tello`. Change one import line and your
  code flies in a 3D simulated world. The camera image, the sensors and the drone's quirks are
  the ones we measured on the lab Tellos.
- **`tellolab`**: altitude hold, heading hold, position hold, visual target centering, safe
  flips, safety watchdogs and a `Pilot` that puts them together. They work the same with the real
  drone and with the simulator.

![viewer](docs/img/viewer.jpg)

## Install

```bash
git clone https://github.com/DroneOps/Tello-Sim-Lab.git
cd Tello-Sim-Lab
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,real]"
```

`real` installs djitellopy (for the real drone); `dev` installs pytest.

## Your code, in the simulator

```python
# from djitellopy import Tello
from tellosim import Tello

tello = Tello()                 # or Tello(world="line_follower", model="lab_tello7", camera_mode="mirror")
tello.connect()
tello.streamon()
frame_read = tello.get_frame_read()
tello.takeoff()                 # ~5 s like the real one, ends at 65–74 cm
tello.send_rc_control(0, 20, 0, 0)
frame = frame_read.frame        # RGB 960x720, ~170 ms late, like the real stream
tello.land()
tello.end()
```

Everything else (your threads, `time.sleep`, OpenCV processing) stays the same. To see what is
going on from above:

```python
viewer = tello.sim.viewer(record="run.mp4")   # map from above + camera, live and recorded
print(tello.sim.pose())                        # the TRUE position, which the real drone never gives you
```

### What it reproduces (measured on the lab Tellos)

| Behavior | Real lab Tello | Where it is |
|---|---|---|
| rc deadband and lag | below rc ~10 it barely moves; ~0.75 s to reach speed | `models/*.json` |
| sinks when flying forward + sideways | ~8 cm/s | `combo_sink_cm_s` |
| drifts by itself | random ~5 cm/s; Tello #7 always ~12 cm/s to its right | `random_drift_cm_s`, `drift_right_cm_s` |
| slides when turning, keeps turning after stopping | 26–42 cm in a 90° turn; 88° → 101° | `turn_drift_cm_s`, `yaw_carry` |
| takeoff | 65–74 cm, ok after 4.5–7.8 s, rc ignored until then | `takeoff_*` |
| falls after a flip | ~50 cm/s for 1–3 s | `flip_fall_*` |
| ToF reads height + 10 cm, garbage mid-flip; `h` unreliable | yes | sensors in `drone.py` |
| video delay | ~170 ms | `video_delay_s` |
| standard Tello answers `unknown command` to `sdk?` and `sn?` | yes | `tello.py` |

And what it does **not** reproduce: motion blur, reflections, a dirty mirror, WiFi drops (unless
you turn on `video_dropouts`), and the forward speed with a mirror (the model is ~1.6x faster
than what we measured with the mirror clip). See [docs/how-it-works.md](docs/how-it-works.md).

## Stable flight on the real drone too

```python
from tellolab import Pilot

pilot = Pilot(tello, frame_read=tello.get_frame_read())
pilot.takeoff(altitude_cm=100)     # preflight checks, takeoff, climb to 100 cm
pilot.set_rc(0, 20, 0, 0)          # your commands; 0 on an axis = hold it (altitude, heading)
pilot.flip("b")                    # climbs to 130 cm, flips, brakes the post-flip fall
pilot.land()
print(pilot.reason)                # why the safety monitor landed it, if it did
```

See [docs/controllers.md](docs/controllers.md) for each controller and how to tune it.

## Examples

Each one flies in the simulator, and on the real Tello with `--real`:

```bash
python examples/hover_hold.py              # Tello #7 drifting, held in place with the floor seen by a mirror
python examples/line_follower.py           # follow a black line with a downward mirror
python examples/color_gate.py              # fly through orange, green and blue gates
python examples/landing_pad.py             # find a pad and land on it
python examples/aruco_follow.py            # stay 1.2 m in front of an ArUco marker
python examples/apriltag_mission.py        # tags on the floor: turn, flip, land...
```

Add `--no-window` to run without the video window, `--model lab_tello7` for a drifting drone.
Results in the simulator and how to move them to the real drone:
[docs/examples.md](docs/examples.md).

## Make it yours

| I want to... | Read |
|---|---|
| build my own track (lines, markers, gates, boxes, walls, my floor) | [docs/maps.md](docs/maps.md) |
| use another Tello's calibration, the forward camera or a mirror | [docs/camera.md](docs/camera.md) |
| make the simulated drone behave like MY drone | [docs/drone-model.md](docs/drone-model.md) |
| test what happens when things go wrong | [docs/faults.md](docs/faults.md) |
| add new object types, sensors, physics; run faster than real time | [docs/extending.md](docs/extending.md) |

## Tests

```bash
pytest -q
```

They run in simulated time (seconds), also on every push (GitHub Actions).

## Where this comes from

Built by the DroneOps drone team (Tec de Monterrey) for the Autonomous Mechatronics Grand Prix
2026 AprilTag challenge. The numbers come from lab sessions in Oct–Nov 2026: the flight logs and
notes are in AprilTag-Detection (`sesiones/`, `docs/tello.md`). MIT license.
