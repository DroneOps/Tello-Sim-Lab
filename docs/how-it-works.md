# How the simulator works

```
your code ──> tellosim.Tello            same methods as djitellopy.Tello
                 │
                 └─ tello.sim  (Sim)    world + drone + camera + renderer, and their threads
                      ├─ World           floor + objects (the map)                 world.py, objects.py
                      ├─ SimDrone        physics at 50 Hz, sensors, faults           drone.py, model.py
                      ├─ Camera          intrinsics, forward or mirror mounting     camera.py
                      └─ Renderer        draws the image at 30 fps                   render.py
```

## 1. A fake Tello: `tellosim.Tello`

It has the methods of `djitellopy.Tello` that Tello projects use:
- **connecting and video:** `connect`, `streamon` / `streamoff`, `get_frame_read().frame`,
  `end`;
- **flying:** `takeoff`, `land`, `emergency`, `send_rc_control`, `move_*`, `rotate_*`, `flip_*`,
  `go_xyz_speed`, `set_speed`;
- **state:** `get_battery`, `get_height`, `get_distance_tof`, `get_yaw` / `pitch` / `roll`,
  `get_speed_*`, `get_acceleration_*`, `get_temperature`, `get_flight_time`,
  `get_current_state`, `get_state_field`;
- **low level:** `send_command_with_return` / `_without_return`, `send_control_command`,
  `query_*`, `get_own_udp_object`, `is_flying`.

Blocking calls block like the real ones (`takeoff` waits for the "ok", ~5 s). Failures raise
`tellosim.TelloException`, with the same name as djitellopy's exception.

`Tello(...)` takes the simulation options on top of djitellopy's arguments:

| Option | Default | What |
|---|---|---|
| `world` | `"lab_grid"` | map: built-in name, YAML path, dict, or `World` ([maps.md](maps.md)) |
| `model` | `"lab_healthy"` | drone model: built-in name, JSON path, dict, or `DroneModel` ([drone-model.md](drone-model.md)) |
| `camera` | `"lab_tello"` | camera calibration ([camera.md](camera.md)) |
| `camera_mode` | the map's, else `"forward"` | `"forward"` or `"mirror"` |
| `seed` | 0 | random seed: drift, noise, takeoff height... Same seed, same flight (in simulated time) |
| `faults` | none | `Faults(...)` or a dict ([faults.md](faults.md)) |
| `battery`, `temperature_c` | 90, model's | starting values |
| `realtime` | True | False: simulated time (see below) |
| `crash_stops` | True | after hitting something the drone falls to the floor |
| any camera field | | e.g. `mirror_tilt_deg=-8`, `noise_std=0` |

## 2. The physics: `SimDrone`

Every 20 ms it turns the rc that arrived `command_delay_s` ago into motion:
- **horizontal:** speed = `forward_gain` · (|rc| − `deadband`), reached with a first-order lag
  of `response_s`;
- **vertical:** the same law, with different gains up and down;
- **yaw:** the same law, plus a "carry" that keeps turning after releasing the stick.

On top of that, the drone moves by itself:
- a constant drift in its own frame;
- a slow random wander;
- a slide to a random side while turning hard;
- sinking when flying forward, and more with forward and sideways at once.

Takeoff, landing, flips (and the fall after them), the floor, boxes and collisions with objects
have their own rules. Every number is in the model JSON and explained in
[drone-model.md](drone-model.md).

**The state dict, at ~10 Hz like the real one:**
- `tof` = height above whatever is below + 10 cm (+ noise, garbage mid-flip);
- `h` = its own height estimate, which wanders: don't use it;
- `yaw`, `pitch`, `roll`;
- `vgx` / `vgy` / `vgz` in dm/s;
- `bat`, `templ` / `temph`, `baro`, `time`, `agx` / `agy` / `agz`.

## 3. The camera: `Camera` + `Renderer`

Each frame is drawn from the pose the drone had `video_delay_s` ago (~170 ms):

1. **The floor, by ray casting.**
   - Every pixel's ray, with the real lens distortion of the calibration, hits the floor plane
     z = 0, and the floor texture (seen from above, with lines and pads painted on it) is
     sampled there with `cv2.remap`.
   - This works the same for the forward camera (tilted 11° down) and for a mirror looking
     down.
   - It is computed at half resolution: the floor is smooth.
2. **Objects, far to near.**
   - Markers and pads are flat pictures warped with a homography.
   - Gates, boxes, walls and panels are colored polygons clipped against the camera's near
     plane.
   - Markers stuck on a wall are drawn right after their wall.
3. **Blur and sensor noise.** Then, with a mirror, the image is flipped like the real one.

You get RGB frames (like djitellopy). Rendering takes ~20 ms per frame on a laptop.

## Real time and simulated time

- **`realtime=True` (default).** Threads run the physics and the video as the real drone would.
  Your code with its own loops and `time.sleep` behaves as it does with the real drone.
- **`realtime=False`.** Nothing moves until you call `tello.sim.advance(seconds)`, which runs the
  physics as fast as the CPU allows. `get_frame_read().frame` draws the frame at the current
  simulated time. Blocking calls (`takeoff`, `move_forward`...) advance the simulated time
  themselves. Use it for tests, batches of runs, or training. The tests in `tests/` fly this
  way, so they take seconds.

## Coordinates

All in cm:
- **x** to the east, **y** to the north (the default takeoff heading), **z** up.
- **yaw** in compass degrees, clockwise from +y.
- **The drone frame** is (forward, right, up).

`tello.sim.pose()` returns the true (x, y, z, yaw).

## What it does NOT simulate

- **Motion blur, reflections, glare, a dirty mirror,** and real lighting (it has flat colors
  with noise).
- **WiFi drops,** unless you turn on `video_dropouts` in [faults.md](faults.md). Lost commands
  are not simulated.
- **The real speed with a mirror clip.** The model gives ~14.5 cm/s with rc 15; with the mirror
  clip we measured ~9 cm/s. Long legs are easier in the simulator than in the lab.
- **Aerodynamics** (ground effect, wind gusts near walls, prop wash). The drift is a statistical
  model of what we saw, not physics.
- **The Tello's internal vision positioning (VPS)** is not modeled: its effect is inside the
  measured drift. Over a floor without texture a real Tello drifts more; use
  `random_drift_cm_s` to mimic that.
- **`move_*` and `rotate_*` speeds were never measured** in our lab (defaults: 30 cm/s, 60°/s).

**The lab lesson:** use the simulator to find bugs in your logic and to compare ideas; trust the
real drone for the final numbers.
