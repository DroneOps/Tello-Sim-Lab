# The drone model: making the simulated Tello behave like yours

```python
Tello(model="lab_healthy")      # default: a healthy lab Tello
Tello(model="lab_tello7")       # lab Tello #7: always drifts ~12 cm/s to its right
Tello(model="ideal")            # no drift, no lag: to debug your logic first
Tello(model="my_tello.json")    # yours
```

## Built-in models

| Model | What | From |
|---|---|---|
| `lab_healthy` | healthy lab Tello | 14 real flights fitted (2026-11-09); drift, sinking and slide measured with the floor seen through a mirror (2026-11-11) |
| `lab_tello1` | Tello #1: drifts ~3.7 cm/s right | hover videos, 2026-11-11 |
| `lab_tello7` | Tello #7: drifts ~11.7 cm/s right, wanders ~10 cm/s | hover videos, 2026-11-11 |
| `ideal` | optimistic: no drift, lag, sinking or slide | — |

## Every field

All speeds in cm/s (or deg/s), times in s.

| Field | lab_healthy | Meaning, and what we saw in the lab |
|---|---|---|
| `forward_gain` | 1.45 | horizontal speed per rc unit above the deadband: rc 15 → 14.5 cm/s. ⚠️ With the mirror clip we measured ~9 cm/s with rc 15: refit for a heavier drone |
| `deadband` | 5 | rc below this does nothing horizontally |
| `response_s` | 0.75 | time to reach the asked speed |
| `combo_factor` | 1.0 | forward and sideways at once: both are multiplied by this. Lab, forward camera: 0.55 |
| `combo_sink_cm_s` | 8 | ...and it sinks this much |
| `forward_sink_cm_s` | 1 | sinks when flying forward |
| `up_gain`, `down_gain` | 0.4, 0.15 | vertical speed per rc unit: climbing is much faster than descending |
| `vertical_deadband`, `vertical_response_s` | 3, 0.6 | |
| `yaw_gain`, `yaw_deadband` | 0.75, 14 | deg/s per rc unit; rc below ~14 barely turns |
| `yaw_response_s` | 0.1 | |
| `yaw_carry`, `yaw_carry_s` | 0.4, 2.0 | part of the turn that continues after releasing the stick (asked to stop at 88°, it reached 101°) |
| `command_delay_s` | 0.04 | rc takes this long to act |
| `video_delay_s` | 0.17 | the image is this late (measured filming a clock) |
| `random_drift_cm_s` | 5.4 | random wander (the std of a slow random walk) |
| `drift_right_cm_s`, `drift_forward_cm_s` | 0, 0 | constant drift in the drone frame (Tello #7: 11.7 right) |
| `turn_drift_cm_s` | 12 | slide to a random side while turning hard (26–42 cm per 90° turn) |
| `takeoff_height_cm` | 65–74 | where takeoff settles |
| `takeoff_peak_cm` | 9 | overshoot before settling |
| `takeoff_ok_s` | 4.5–7.8 | when the "ok" arrives; rc is ignored until then |
| `takeoff_climb_cm_s`, `land_speed_cm_s` | 30, 40 | |
| `flip_min_battery` | 50 | flips refused below this battery |
| `flip_fall_cm_s`, `flip_fall_s` | 50, 1–3 | the fall after a flip (from 117 to 39 cm; once to the floor) |
| `tof_noise_cm` | 2 | |
| `height_bias_max_cm` | 40 | how far the Tello's own 'h' wanders (it read 10 while flying at 76) |
| `battery_pct_per_min` | 6 | |
| `temp_start_c`, `temp_c_per_min` | 65, 6.5 | it heats up (60 → 85 °C in ~5 min with video on) |
| `move_speed_cm_s`, `rotate_speed_deg_s` | 30, 60 | `move_*` / `rotate_*` speeds. ⚠️ not measured |

## Your own model

**Start from a built-in model and change only what differs:**

```json
{
  "base": "lab_healthy",
  "description": "Our Tello with the prop guards and a mirror clip",
  "forward_gain": 0.9,
  "drift_right_cm_s": 4.0
}
```

**Or in Python:**

```python
from tellosim import load_model
heavy = load_model("lab_healthy").with_changes(forward_gain=0.9, up_gain=0.3)
Tello(model=heavy)
```

## Fitting it with your flights

1. **Log real flights.**
   - The easiest way is with the Pilot: `Pilot(tello, log_path="logs/f1.csv")`.
   - Or use `tellolab.FlightLog(tello, path)` and call `log.write(rc)` in your loop.
   - Fly varied commands: turns with different rc, climbs and descents, forward and sideways
     moves. Several flights of 1–2 min.
2. **See how well a model reproduces them:**
   ```bash
   python tools/fit_model.py logs/*.csv --base lab_healthy
   ```
3. **Fit and save:**
   ```bash
   python tools/fit_model.py logs/*.csv --fit --out my_tello.json
   ```

**What it fits:**
- **Yaw and vertical:** from the logged yaw and ToF.
- **Horizontal:** only if the logs have a position (x, y). Pass `FlightLog(..., position=...)`
  from a motion-capture system like OptiTrack.
- **Takeoff height:** with 3 or more flights.

It also reads AprilTag-Detection logs. Against the 13 flights of 2026-11-09, `lab_healthy`
predicts the change over 1 s within 0.3° of yaw and 3.9 cm of height (the `ideal` model: 0.5°,
4.5 cm).

**The drift is not fitted** because it needs a position. Measure it hovering with rc 0:
- with a downward mirror and `tellolab.FloorFlow`: the median drift is
  `drift_right/forward_cm_s`, and the spread is `random_drift_cm_s`;
- or with OptiTrack;
- or with a tape mark on the floor and a stopwatch.

**Check it after every lab session:** if the simulator stops looking like the drone (a new clip,
new props, another drone), refit.
