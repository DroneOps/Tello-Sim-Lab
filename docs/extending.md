# Extending: what you can modify, add or remove

## Quick reference

| I want to... | Do this |
|---|---|
| a different track | a map YAML ([maps.md](maps.md)) |
| my drone's behavior | a model JSON with `base` ([drone-model.md](drone-model.md)) |
| my camera | a calibration JSON, `camera_mode`, `mirror_tilt_deg` ([camera.md](camera.md)) |
| things going wrong | `Faults` ([faults.md](faults.md)) |
| more or less noise in the image | `noise_std`, `blur_sigma` |
| remove the drift entirely | `model="ideal"`, or `random_drift_cm_s=0, turn_drift_cm_s=0` |
| remove crashes | `Tello(crash_stops=False)`: the crash is still recorded in `tello.sim.crash` |
| run 100 flights quickly | simulated time (below) |

## Simulated time: fast, repeatable runs

```python
t = Tello(realtime=False, seed=3)
t.connect(); t.streamon(); t.takeoff()       # takeoff advances the simulated time itself
for _ in range(200):                         # 20 s of flight, as fast as the CPU allows
    img = t.get_frame_read().frame           # drawn at the current simulated time
    lr, fb, ud, yaw = my_controller(img, t.get_current_state())
    t.send_rc_control(lr, fb, ud, yaw)
    t.sim.advance(0.1)
print(t.sim.pose(), t.sim.crash)
```

- **Same seed, same flight.** Use it to compare two versions of your code, or to train or tune
  something.
- **Without images, it is much faster:** don't read `.frame`.
- **To run many in parallel,** use processes, not threads (the renderer uses the CPU).

## A new kind of object

Subclass `SimObject` in `tellosim/objects.py` (or in your own file) and register it:

```python
from dataclasses import dataclass
import numpy as np
from tellosim.objects import SimObject, SolidPoly, TexturedQuad, OBJECT_TYPES, bgr

@dataclass
class Pole(SimObject):
    """A thin vertical pole (drawn as a flat strip facing north; crash if touched)."""
    position: tuple = (0.0, 200.0)
    height_cm: float = 180.0
    width_cm: float = 5.0
    color: object = "yellow"

    def primitives(self):
        x, y = self.position
        w = self.width_cm / 2
        v = np.array([[x - w, y, self.height_cm], [x + w, y, self.height_cm], [x + w, y, 0], [x - w, y, 0]])
        return [SolidPoly(v, bgr(self.color))]          # normal=None: visible from both sides

    def collides(self, p, radius):
        if np.hypot(p[0] - self.position[0], p[1] - self.position[1]) < radius + self.width_cm / 2 \
                and p[2] < self.height_cm:
            return f"hit pole {self.name}"

OBJECT_TYPES["pole"] = Pole        # now "type: pole" works in map files
```

**What an object can implement:**
- **`primitives()`:** what to draw.
  - `SolidPoly(vertices, color, normal)` is a flat convex polygon.
  - `TexturedQuad(corners, image, normal)` is a picture: corners in the order top-left,
    top-right, bottom-right, bottom-left of the image.
  - Several primitives make a 3D shape (see `Box`).
- **`paint_floor(canvas, to_px)`:** paint on the floor texture (cheap, for flat things on the
  floor).
- **`top_height(x, y)`:** height the ToF reads under (x, y).
- **`collides(p, radius)`:** a message if the drone touches it.

Curved things are polygons with many sides (see the circular `Gate`).

## Another sensor

The world gives you everything a sensor needs. For example, a forward range sensor:

```python
def front_distance(sim, max_cm=400, step=2):
    x, y, z, yaw = sim.pose()
    h = np.array([np.sin(np.radians(yaw)), np.cos(np.radians(yaw)), 0])
    for d in np.arange(0, max_cm, step):
        p = np.array([x, y, z]) + d * h
        if sim.world.collision(p, 1.0) or sim.world.surface_height(p[0], p[1]) > z:
            return d
    return None
```

Other ideas:
- **A second camera:** a `Camera` + `Renderer` ([camera.md](camera.md)).
- **A downward ToF:** it is already in the state (`tof`).
- **A motion-capture system:** `sim.pose()` plus noise.

## Other physics

- **The numbers:** a model JSON ([drone-model.md](drone-model.md)).
- **New behavior** (e.g. ground effect, wind gusts near a wall): subclass `SimDrone` and
  override `_step`. Keep the structure:
  1. read the delayed rc with `self._rc_now(now)`;
  2. move `self.x / y / z / yaw`;
  3. let the base class update the sensors.

  Then pass your drone in: build a `Sim` and replace `sim.drone`, or add a `drone_class` option
  to `Sim`.
- **Another drone entirely** (not a Tello): the camera, world and renderer don't care. Write a
  model for it, and a class like `tellosim.Tello` with its own API.

## Removing things

- **The video:** don't call `streamon` / `get_frame_read`. The renderer only runs when frames
  are read.
- **An object type you don't need:** just don't use it.
- **The lab quirks you don't want:** set them to 0 in your model (`combo_sink_cm_s`,
  `turn_drift_cm_s`, `yaw_carry`, `height_bias_max_cm`...).

## Tests

Add a test in `tests/` for anything you add. They run in simulated time; see
`tests/test_tellosim.py` for the patterns:
- **teleport the drone** with `place()`;
- **detect markers** in `t.sim.capture()`;
- **step the physics** with `SimDrone` directly.

```bash
pytest -q
```
