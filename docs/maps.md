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
| `tmr_gates` | three green 110 cm gates 5 m apart, the first 5 m ahead (like DroneOps/Gate_tmr_color) | forward |
| `tmr_landing` | a pad with a 10 cm 6x6 ArUco, 20 cm behind the start, moving left and right at 10 cm/s (like DroneOps/TMR_Landing) | mirror |

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
fog: {visibility_cm: 400}          # optional: fog or smoke (see "Fog and smoke")
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

**`landing_pad`** (it can move: see "Moving objects")

| Field | Default | Meaning |
|---|---|---|
| `position` | | `[x, y]` |
| `size_cm` | 60 | side or diameter |
| `shape` | square | `square` or `circle` |
| `color`, `border_color`, `border_cm` | white, red, 4 | |
| `letter`, `letter_color` | "H", black | drawn when there is no marker |
| `marker` | none | `{dictionary, id, size_cm}` in the middle |
| `height_cm` | 0 | > 0: a raised platform (the ToF reads its top; its sides can be hit). Round ones are cylinders |
| `rotation_deg`, `side_color` | 0, gray | |

**`gate`:** fly through it.
- `position: [x, y, z]`: the center of the OPENING.
- `width_cm`, `height_cm` (80): the opening; for a circle, `width_cm` is the inner diameter.
- `bar_cm` (8), `facing_deg` (the direction you fly through it), `color` (orange).
- `shape`: `rect` or `circle`.
- `legs` (true): posts to the floor, rect only; `leg_color` (the frame's color): e.g. `gray` stands, so a color mask sees only the frame.

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

## Moving objects

Any object except a line can move at constant speed along a straight line, with `motion`:

```yaml
  - type: landing_pad
    position: [0, -20]
    size_cm: 50
    motion: {direction_deg: 90, distance_cm: 200, speed_cm_s: 10, start_s: 10}
```

| Field | Default | Meaning |
|---|---|---|
| `direction_deg` | 90 | where it moves (compass: 90 = east = to the right at the start) |
| `distance_cm` | 200 | length of the run, centered on `position` (here from x = −100 to +100) |
| `speed_cm_s` | 10 | constant speed |
| `back_and_forth` | true | true: it turns around at each end, forever. false: it goes once from `position`, `distance_cm` along `direction_deg`, and stops |
| `start_s` | 0 | seconds after the start before it moves (time to take off and find it) |
| `pause_s` | 0 | back and forth only: seconds it stays still at each end before turning around, like a car driven "forward 3 s, stop 1 s, back 3 s, stop 1 s" |

- **The camera sees it where it is** (also through the ~170 ms video delay), and the viewer
  draws it moving.
- **It moves its collisions and the ToF too:** a moving box or raised pad can hit the drone.
- **A drone that lands on a moving pad or box moves with it.**
- **The report of `tellosim.run`** says how far from the pad's center the drone touched down,
  measured where the pad was at that moment.
- A line is painted on the floor texture and cannot move: use a pad or a panel.
- **A car instead of a flat pad:** give the pad `height_cm` (the ToF then reads the car top
  when the drone is over it, the floor beside it, and its side can be hit) and `pause_s`. The
  built-in `tmr_landing_car` map is an example; measure your car's height, size, speed and
  times, and change them.
- In Python: `LandingPad(..., motion={"direction_deg": 90, "speed_cm_s": 10})`, and
  `world.offset(obj, t)` says where it is at `t` seconds.

## Glossy markers and lamps

A marker in a plastic sleeve or laminated reflects the room's lamps. On the real drone this was a
real problem (TMR_Landing, Oct 2026): a white patch over the marker, and where it falls on the
black border the detector does not find the marker at all, although the rest looks fine.

```yaml
lights:                              # the room's lamps
  - {position: [0, 0, 270], size_cm: 30}
  - {position: [240, 0, 270], size_cm: 30}
objects:
  - type: landing_pad
    position: [0, -20]
    marker: {dictionary: DICT_6X6_100, id: 0, size_cm: 10, gloss: 0.7}   # only the marker's card
  - type: marker
    id: 3
    position: [100, 100]
    gloss: {strength: 0.7, roughness: 0.04}
```

| Field | Default | Meaning |
|---|---|---|
| `lights[].position` | `[0, 0, 270]` | where the lamp is, in cm (z = its height) |
| `lights[].size_cm` | 30 | its diameter: a ceiling panel ~60, the width of a tube ~30, a bulb ~10. Bigger = a bigger reflection |
| `lights[].strength` | 1 | 0..1 |
| `gloss` or `gloss.strength` | 0.7 | how bright the reflection gets: 0.7 turns black into light gray (what the lab photo showed); 1 = white |
| `gloss.roughness` | 0.04 | how blurred, in radians: glossy laminate ~0.03–0.05, a matte-ish sleeve ~0.15 |

`gloss` goes on a `marker` (its printed card, margin included), on a `landing_pad` (its whole
top) or only on a pad's marker (`marker: {..., gloss: ...}`).

- **How:** for every pixel of a glossy surface, the camera's ray is mirrored on it and compared
  with the direction to each lamp. Where they line up, the surface shows the lamp, blurred by
  the roughness. So the reflection is where the real one would be, **moves when the drone
  moves**, and is bigger when the drone is higher.
- **Where to expect it:** looking straight down from height h, with a lamp at height H, the
  reflection falls on the floor at h/(h+H) of the way from below the drone to below the lamp.
  So it is on the marker only when the drone is almost between the marker and a lamp: at 30 cm
  that is up to ~60 cm from below the lamp, at 1 m only ~20 cm.
- **Without editing the map:** `--glare 0.7` (or `Tello(glare=0.7)`) makes every marker glossy
  and, if the map has no `lights`, puts a grid of ceiling lamps every 240 cm at 270 cm, one
  right above the start. `--glare '{"strength": 0.7, "lamp_height_cm": 300,
  "lamp_spacing_cm": 200, "lamp_size_cm": 60}'` describes your room.
- It costs ~5–15 ms per frame while a reflection is on screen, nothing otherwise.
- Not simulated: shadows (the drone's own shadow over the reflection), windows.

## Fog and smoke

The real TMR will have fog or smoke. Add it to the map, or when you run:

```yaml
fog: {visibility_cm: 300, patches: 0.5}
```

```bash
python -m tellosim.run --world tmr_landing --fog 300 my_program.py
python -m tellosim.run --world tmr_gates --fog '{"visibility_cm": 800, "patches": 0.6, "color": [120, 120, 120]}' my_program.py
```

```python
Tello(world="tmr_gates", fog=800)                                  # even fog, 8 m of visibility
Tello(world="tmr_gates", fog={"visibility_cm": 400, "patches": 0.6})  # smoke
Tello(world="tmr_gates", fog=False)                                # clear air, even if the map has fog
```

| Field | Default | Meaning |
|---|---|---|
| `visibility_cm` | 500 | at this distance things are 95 % hidden. Each pixel fades towards `color` as exp(−3 d / visibility), d = its distance to the camera |
| `color` | [205, 205, 205] | light gray fog; darker for smoke |
| `patches` | 0 | 0 = even fog. 0.3–1 = smoke: thicker and thinner patches that drift across the image |
| `patch_speed` | 0.03 | how fast the patches drift (image widths per second) |
| `seed` | 0 | another pattern of patches |

**How much is a lot:**
- with 4 m of visibility, a gate 5 m away is gone;
- with 1.5 m, a marker 1 m below a mirror is barely detected (TMR_Landing's detector, see
  [your-code.md](your-code.md));
- start with 3–8 m and go down.

**Only the image changes:** the ToF, the drone's own sensors and the physics do not see the
fog. The patches move across the image, not through the room, so turning does not move them.
It adds ~6 ms per frame.

## Check your map

```bash
python tools/check_maps.py my_track.yaml
python tools/check_maps.py my_track.yaml --images views/   # also save what each check saw
```

It checks:
- **Textures:**
  - the floor texture covers every object (and the whole run of moving ones) and the start;
  - lines are at least 3 px wide in the texture;
  - round pads have no square around them.
- **Dimensions:**
  - every marker is detected by OpenCV and measures its `size_cm` within 3 %;
  - every rectangular gate's opening measures its `width_cm` in the image;
  - markers fit on their pad;
  - gates do not go into the floor.
- **Collisions:**
  - flying through a gate's opening is free, and its frame and legs are not;
  - boxes, walls, panels and raised pads stop the drone and leave room around them;
  - flat things never do;
  - nothing crosses the takeoff spot.
- **Motion and fog:** moving objects keep their speed and run, and the fog has a sensible
  visibility.

Every built-in map passes it (it is also a test).

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
