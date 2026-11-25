# Examples

Each example is written like a normal djitellopy program: a loop that reads frames, decides and
sends rc. The only difference between the simulator and the real drone is the `--real` flag,
which picks the Tello class in `examples/common.py`.

**Common options:**

| Option | What |
|---|---|
| `--real` | fly the real Tello (djitellopy). Needs the STOP window: someone with a hand on the keyboard |
| `--world`, `--model`, `--camera-mode`, `--tilt`, `--seed` | simulator settings |
| `--no-window` | simulator only, no video window |
| `--timeout` | land after this many seconds |

**Results in the simulator** (lab_healthy model unless stated, one run each, Nov 2026):

| Example | What it does | Simulator result |
|---|---|---|
| `hover_hold.py` | takeoff to 100 cm and hover. With a mirror, `FloorFlow` + `PositionHold` hold the position | Tello #7 model, 30 s: at most **41 cm** from the start (without position hold: **4.2 m**) |
| `line_follower.py` | follow a dark line with a downward mirror: the dark pixels become floor points, a line is fitted, then yaw from its angle and sideways from its offset | whole loop, **6 cm** from the line (median), 26 max |
| `color_gate.py` | orange, green, blue gates with the forward camera: find the opening, center it on the horizon row, go through | **3 of 3** gates, no crash |
| `landing_pad.py` | find the red-border pad with the mirror, get over it at 150 cm, land | **5 cm** from the center |
| `aruco_follow.py` | keep 120 cm in front of an ArUco marker on a wall | **105–133 cm** in 4 runs (it varies from run to run) |
| `apriltag_mission.py` | tags on the floor with a mirror: 1 turn right, 2 left, 7 land... | executed **1, 2, 7**, landed **10 cm** from tag 7 |

**Things the examples had to learn** (they will happen with your own code too):
- **`color_gate`:** the forward camera looks 11° down and sees only ~10° above the horizon.
  Closer than ~2 m at the gate's height, its top bar is out of view, so the opening is not a
  closed shape. It climbs to see the whole frame, aligns, and goes through steering by the
  posts once the top bar leaves the view.
- **`landing_pad`:** the mirror sees only ~20–30 cm ahead, so the front of a 60 cm pad is cut
  off and the center of the visible red is shifted back ~10 cm. It measures from the back edge,
  which is always in view.
- **`aruco_follow`:** a marker above the drone leaves the top of the image when approaching. It
  gets to the marker's height first.
- **`apriltag_mission`:** after drifting at takeoff, a tag to the side can be nearer than the one
  ahead. It picks the tag most in line with its heading.

## From the simulator to the real drone

1. **Run it in the simulator** with a drifting model (`--model lab_tello7`) and a couple of
   seeds. If it fails there, it will fail in the lab.
2. **Use your drone's camera calibration and your mirror tilt** ([camera.md](camera.md)).
   `examples/common.py` loads `tellosim/cameras/lab_tello.json` for `--real`: replace it with
   yours.
3. **Print the markers at the size the example expects** (`--size`), or pass your size.
4. **Check the colors under your light.** HSV thresholds are in the example (`HSV` in
   `color_gate.py`, `color_ranges` in `landing_pad.py`). Take a frame with the real drone and
   test them.
5. **First real flight:**
   - someone at the keyboard with the window focused: SPACE lands, `k` kills the motors (only
     if it goes towards a person);
   - battery > 70 % for flips;
   - temperature < 70 °C;
   - nobody in the flight area.
6. **Start with `hover_hold.py --real`.** If it does not hover well, nothing else will.
