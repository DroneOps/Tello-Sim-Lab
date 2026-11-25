"""Flight logs in a simple CSV, the same for the real drone and the simulator.

    log = FlightLog(tello, "logs/flight1.csv")     # or Pilot(..., log_path="logs/flight1.csv")
    log.write(rc_sent)                             # every control loop
    log.close()

Columns: t (s), flying, rc_lr, rc_fb, rc_ud, rc_yaw (what you SENT), then the Tello state (tof,
h, yaw, pitch, roll, vgx, vgy, vgz, bat, temph), and x, y, z when the simulator (or your
motion-capture) gives a true position. tools/fit_model.py reads these to fit a drone model.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

FIELDS = ["t", "flying", "rc_lr", "rc_fb", "rc_ud", "rc_yaw", "tof", "h", "yaw", "pitch", "roll",
          "vgx", "vgy", "vgz", "bat", "temph", "x", "y", "z"]


class FlightLog:
    def __init__(self, tello, path: str | Path, position=None, clock=time.monotonic) -> None:
        """position: optional callable -> (x, y, z) in cm (motion capture). With tellosim the
        true pose is logged automatically."""
        self.tello, self.clock = tello, clock
        self.position = position
        if position is None and hasattr(tello, "sim"):
            self.position = lambda: tello.sim.pose()[:3]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._f = open(path, "w", newline="")
        self._w = csv.writer(self._f)
        self._w.writerow(FIELDS)
        self._t0 = clock()

    def write(self, rc=(0, 0, 0, 0), flying: bool = True) -> None:
        s = self.tello.get_own_udp_object()["state"] or {}
        pos = self.position() if self.position else None
        self._w.writerow([f"{self.clock() - self._t0:.3f}", int(flying), *[int(v) for v in rc],
                          *[s.get(k, "") for k in FIELDS[6:16]],
                          *([f"{v:.1f}" for v in pos] if pos else ["", "", ""])])

    def close(self) -> None:
        self._f.close()
