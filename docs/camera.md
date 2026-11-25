# The simulated camera

```python
Tello()                                         # lab calibration, forward camera
Tello(camera_mode="mirror", mirror_tilt_deg=-8) # a mirror clip looking down, tilted 8° back
Tello(camera="my_tello.json")                   # your own Tello's calibration
Tello(noise_std=0, blur_sigma=0)                # a perfectly clean image
```

Any `Camera` field can be passed to `Tello(...)` as a keyword. The map can suggest a mode with
`camera: {mode: mirror}`; `camera_mode=` wins.

## Fields

| Field | Default | Meaning |
|---|---|---|
| `K`, `dist` | lab calibration | intrinsics and OpenCV distortion (k1, k2, p1, p2, k3) of the image as the camera delivers it |
| `width`, `height` | 960, 720 | the Tello's video size |
| `mode` | `forward` | `forward` or `mirror` |
| `pitch_deg` | 11 | forward: how far below the horizon the optical axis points (measured on the lab Tellos) |
| `mirror_axis` | `v` | mirror: how it flips the image (`v` top-bottom, `h` left-right) |
| `mirror_tilt_deg` | -10.3 | mirror: view tilt from straight down, + = towards the nose (lab clip: -10.3, i.e. a bit backwards) |
| `fps` | 30 | frames per second of the stream |
| `noise_std` | 3 | sensor noise (gray levels) |
| `blur_sigma` | 0.6 | slight blur |

The video delay (~170 ms) is in the drone model (`video_delay_s`), because it is the link, not
the lens.

## The two mountings

**Forward camera.**
- The Tello's own camera looks forward, 11° down.
- It sees from ~32° below the horizon to only ~10° above it.
- So an object at your height but close is out of the top of the image: a wall marker 30 cm
  above the drone leaves the image when closer than ~1.6 m. `examples/aruco_follow.py` climbs to
  the marker's height before approaching for this reason.

**Downward mirror.**
- A mirror clipped in front of the lens makes the camera look down.
- **The image comes out flipped.** Flip it back before detecting markers (a mirrored marker
  does not decode): `tellolab.vision.unmirror(img)`.
- **Use the intrinsics of the flipped image:** `tello.sim.camera.detection_K` in the simulator,
  or `tellolab.vision.mirrored_intrinsics(K, dist)` on the real drone.
- **How much it sees.** With the lab clip (-10.3°) at 100 cm it sees ~20 cm ahead, ~60 cm behind
  and ±52 cm to the sides. Things ahead appear when they are almost below the drone.

## Using another Tello's calibration

Every Tello's lens is a little different, and distances measured with markers (solvePnP)
depend on it.

1. **Calibrate it** with a chessboard, for example with DroneOps/camera-calibration or OpenCV's
   `calibrateCamera`. Use the 960x720 stream.
2. **Save a JSON** like `tellosim/cameras/lab_tello.json`:
   ```json
   {"description": "Tello #3, 2026-12-01", "resolution": [960, 720],
    "K": [[920.0, 0, 471.5], [0, 911.5, 386.3], [0, 0, 1]],
    "dist": [-0.086, 0.27, -0.0086, -0.0063, -0.37], "pitch_deg": 11.0}
   ```
3. `Tello(camera="tello3.json")`, or put it in `tellosim/cameras/` and use its name.
4. **With the real drone,** use the same K and dist in your code (`examples/common.py` reads the
   lab file).

**Measuring the camera pitch:** look at a marker on the floor at a known distance and compare the
pose solvePnP gives with the tape measure. The lab Tellos were 11° in three videos.

**Measuring the mirror tilt.** It changes with every clip, and even when the clip gets bumped.
- Fly over a floor marker and compare where solvePnP says it is with the ToF height, or use
  AprilTag-Detection's `scripts/documentar/inclinacion_vuelo.py`.
- **Measure it in flight, not in your hand:** the lab clip read +4° in the hand and -10.3° in
  flight.
- A tilt that is wrong in your code makes the drone stop ~20 cm short of a tag. Try
  `Tello(camera_mode="mirror", mirror_tilt_deg=-2)` with code that assumes -10.3.

## Changing what the image looks like

- **Cleaner or noisier:** `noise_std`, `blur_sigma`.
- **Colors and textures:** they come from the map (object colors, floor texture or your own
  floor image).
- **Lighting:** there is none. To test color thresholds under other light, change the colors in
  the map, or post-process the frames in your test (brightness, gamma).
- **Another resolution:** set `width`, `height` and a `K` that matches. The Tello always sends
  960x720, so this is only for other cameras.
- **Rendering speed:** `Renderer(..., floor_scale=0.5)` draws the floor at half resolution
  (default). 1.0 is sharper and slower.

## A second camera (e.g. forward AND mirror)

The Tello has one camera, but a simulator can have more:

```python
from tellosim import Camera, Renderer
down = Camera.load("lab_tello", mode="mirror")
r2 = Renderer(tello.sim.world, down)
x, y, z, yaw = tello.sim.pose()
img_down = r2.render(x, y, z, yaw)     # BGR, flipped like a mirror image
```
