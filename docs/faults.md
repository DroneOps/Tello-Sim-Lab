# Faults: testing what happens when things go wrong

Everything here happened to us in the lab. Test your code against it before it happens to you.

```python
from tellosim import Tello, Faults
Tello(faults=Faults(takeoff="collapse"))
Tello(faults={"imu_bad": (10, 40, -35), "video_dropouts": [(15, 18)]})
```

| Fault | What it does | What happened in the lab | What your code should do |
|---|---|---|---|
| `takeoff="high"` | climbs by itself to ~141 cm, ok at 10 s | 2026-11-09, with a mirror clip | bring it down to your height before starting |
| `takeoff="collapse"` | climbs ~40 cm and comes back to the floor while its 'h' says it climbs; never ok | 2026-11-09: the clip covered the downward sensors | detect it (ToF goes back below ~15 cm after passing 30) and land: `SafetyMonitor.check_takeoff` |
| `takeoff="no_ok"` | flies normally but never answers "ok": `takeoff()` raises after 20 s | 2 of 14 takeoffs on 2026-11-09 | don't trust the reply alone: confirm with the ToF |
| `imu_bad=(from_s, fwd_cm_s, up_cm_s)` | from that second it moves by itself, whatever you ask | 2026-10-27: asked to climb at full power, it went down 45 cm and forward 120 cm | land: `SafetyMonitor` ("the drone does not obey") |
| `tof_glitches=[(from_s, to_s, value)]` | the ToF reads that value | 81 → 14 → 83 cm in 0.4 s | ignore isolated jumps: `AltitudeFilter` |
| `wind_cm_s=(x, y)` | constant wind in world coordinates | fans and the AC in the lab | hold the position (`PositionHold`) |
| `speed_factor` | the same rc moves more or less | rc 15 gave 12–20 cm/s between flights | measure the speed instead of assuming it |
| `video_dropouts=[(from_s, to_s)]` | the video freezes | WiFi trouble with many drones around | detect it (the frame stops changing) and hover or land: `Pilot` lands after 3 s |

**Faults that are always on, because the real drone always does them:**
- **Low battery:** flips are refused below 50 %, and it lands by itself below 5 %.
- **It heats up,** at 6.5 °C per minute.
- **Its own 'h' is unreliable.**
- **It falls after flips.**

To make them worse or remove them, change the model ([drone-model.md](drone-model.md)) or the
starting values: `Tello(battery=40)` to test a refused flip, `Tello(temperature_c=82)` to test a
preflight check.

**Times** (`from_s`, `to_s`) count from when the `Tello` is created.
