# Maps: building or adapting the virtual world

A map is a floor plus objects. Built-in maps are in `tellosim/maps/`:

| Map | What | Camera |
|---|---|---|
| `lab_grid` | the empty lab floor: dark with light tape every 60 cm | forward |
| `apriltag_course` | 8 cm tag25h9 tags 1 → 2 → 7 on the floor | mirror |
| `line_follower` | a black 5 cm line loop on a light floor | forward (try mirror) |
| `color_gates` | orange square gate, green ring, blue square gate | forward |
| `landing_pad` | an "H" pad and a raised platform with an ArUco marker | mirror |
| `aruco_board` | four ArUco markers on a wall, and a box | forward |
| `tmr_gates` | three green 90 cm gates in a row (like DroneOps/Gate_tmr_color) | forward |
| `tmr_landing` | a pad with a 10 cm 6x6 ArUco, 20 cm behind the start (like DroneOps/TMR_Landing) | mirror |

```python
Tello(world="line_follower")          # built-in
Tello(world="my_track.yaml")          # your file
Tello(world={"objects": [...]})       # a dict with the same structure
Tello(world=my_world)                 # a World built in Python
```

## Coordinates

- **Units:** centimeters.
- **x** to the east, **y** to the north (the default takeoff heading), **z** up.
- **The drone takes off at `start`** (default (0, 0), facing north).
- **Angles are compass degrees:** 0 = north (+y), 90 = east (+x).
- **Colors are RGB** (`[255, 0, 0]` is red) or a name: black, white, gray, red, green, blue,
  yellow, orange, purple, cyan, pink, lab_tape, dark_floor.

## A map file

```yaml
name: my_track
description: Two tags and a gate.
camera: {mode: forward}            # optional: the camera this map is meant for (Tello(camera_mode=...) wins)
floor: {type: lab_grid}            # see "Floor" below
start: {position: [0, 0], yaw_deg: 0}
background: [150, 150, 145]        # what the camera sees above the horizon
objects:
  - {type: apriltag, id: 1, position: [0, 100], size_cm: 8}
  - {type: gate, position: [0, 250, 100], width_cm: 80, height_cm: 80, color: orange}
  - type: line
    points: [[0, 0], [0, 300], [100, 400]]
    width_cm: 5
    color: black
```

Unknown fields are an error (with the list of valid ones), so typos do not pass silently.

## Floor

| Field | Default | Meaning |
|---|---|---|
| `type` | `lab_grid` | `lab_grid`, `plain`, `checker` or `image` |
| `color` | `dark_floor` | base color |
| `tape_color`, `tape_cm`, `spacing_cm` | lab tape, 6, 60 | grid tape (`lab_grid`) or square size (`checker`) |
| `second_color` | white | the other squares of `checker` |
| `noise` | 1.0 | soft patches and grain. 0 = perfectly smooth (a real Tello drifts more there, and floor flow cannot measure) |
| `image` | | a picture of your floor (path relative to the map file), stretched to `size_cm` |
| `size_cm` | [1400, 1400] | area covered by the texture; outside it, plain `color` |
| `center` | [0, 0] | center of that area |
| `px_per_cm` | 2 | texture resolution. Raise it for thin lines; a 14 m texture at 2 px/cm is 23 MB |

**Your own floor:** take a photo from above (or draw it), and set `type: image`, `image:
floor.png`, `size_cm: [width, depth]` and `center`.

## Objects

**Markers:** `apriltag`, `aruco`, `marker`

| Field | Default | Meaning |
|---|---|---|
| `id` | 0 | marker id |
| `dictionary` | apriltag: `DICT_APRILTAG_25h9`; aruco: `DICT_4X4_50` | any OpenCV dictionary (`DICT_APRILTAG_36h11`, `DICT_6X6_250`...) |
| `size_cm` | 8 | side of the BLACK square (what detectors call the marker size). A white margin of one module is added around it |
| `mount` | `floor` | `floor` or `wall` |
| `position` | | floor: `[x, y]`; wall: `[x, y, z]` of the center |
| `rotation_deg` | 0 | floor: where the marker's top edge points |
| `facing_deg` | 180 | wall: where the printed face points (180 = it faces a drone coming from the south) |
| `wear` | none | `lab` (faded like the lab tags seen through the mirror), `very_faded`, `stains`, `missing`, `all`, or a dict `{black, white, stains, missing}` |

**`line`:** a painted or taped line on the floor.
`points: [[x, y], ...]`, `width_cm` (5), `color` (black), `closed` (false).

**`landing_pad`**

| Field | Default | Meaning |
|---|---|---|
| `position` | | `[x, y]` |
| `size_cm` | 60 | side or diameter |
| `shape` | square | `square` or `circle` |
| `color`, `border_color`, `border_cm` | white, red, 4 | |
| `letter`, `letter_color` | "H", black | drawn when there is no marker |
| `marker` | none | `{dictionary, id, size_cm}` in the middle |
| `height_cm` | 0 | > 0: a raised platform (the ToF reads its top; its sides can be hit) |
| `rotation_deg`, `side_color` | 0, gray | |

**`gate`:** fly through it.
- `position: [x, y, z]`: the center of the OPENING.
- `width_cm`, `height_cm` (80): the opening; for a circle, `width_cm` is the inner diameter.
- `bar_cm` (8), `facing_deg` (the direction you fly through it), `color` (orange).
- `shape`: `rect` or `circle`.
- `legs` (true): posts to the floor, rect only.

Hitting the frame or the legs is a crash.

**`panel`:** a flat colored rectangle standing up.
`position: [x, y, z]` (center), `width_cm`, `height_cm`, `facing_deg`, `color`.

**`box`:** an obstacle or a platform on the floor.
- `position: [x, y]` (center), `size: [width (x), depth (y), height]`, `rotation_deg`, `color`.
- Its sides are shaded so it looks 3D.
- The ToF reads its top when the drone is above it.

**`wall`:** `start: [x, y]`, `end: [x, y]`, `height_cm` (200), `color`.

Any object can have a `name`, used by `world.find(name)`, `world.remove(name)` and the crash
messages.

## In Python

```python
from tellosim import Tello, World, Floor, AprilTag, Gate, FloorLine, Box, load_map

world = load_map("lab_grid")                       # start from a built-in map...
world.add(AprilTag(id=3, position=(50, 120), size_cm=15),
          Gate(name="g1", position=(0, 300, 110), color="green", shape="circle"))
world.remove("g1")                                 # by name, instance, or type (world.remove(Gate))
tello = Tello(world=world)

# ...or build it from scratch
world = World(floor=Floor(kind="plain", color=(220, 220, 220)),
              objects=[FloorLine(points=[(0, 0), (0, 400)], width_cm=3, color="blue")],
              start=(0, -50, 0))
```

To change the world **while flying**, edit `tello.sim.world` and then call
`tello.sim.world_changed()`. For example, move a gate between runs, or remove a box to test a
fallback.

## Making a map like a real track

1. **Measure where each thing is** from a reference point (where the drone takes off), with a
   tape measure.
2. **Print markers at a known size;** that size goes in `size_cm`.
3. **Floor:** if the real one is plain and shiny, use `plain` with low `noise`. A lab-like
   floor with tape: `lab_grid` with your `spacing_cm`.
4. **Fly the real drone over it once,** recording video (`--record`). Compare its frames with
   the simulator's (`tello.sim.capture()` at the same pose) to check sizes and colors.

Random tracks: AprilTag-Detection's `sim/pista.py` generated competition-like tracks. It is a
good starting point for a generator that builds a `World` in Python.
