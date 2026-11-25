#!/usr/bin/env python3
"""Fit a drone model to real flights.

It replays the rc you sent in each flight through the drone model and compares, in 1 s windows,
what the model did with what the drone did: how much it turned, climbed and (if the log has a
position: motion capture) moved. With --fit it searches the values that reproduce your flights
best and writes them as a model JSON for tellosim.

    python tools/fit_model.py logs/*.csv                         # how well lab_healthy fits them
    python tools/fit_model.py logs/*.csv --base lab_tello7       # another starting model
    python tools/fit_model.py logs/*.csv --fit --out my_tello.json
    python -c "from tellosim import Tello; Tello(model='my_tello.json')"   # then use it

Logs: tellolab.FlightLog CSVs (Pilot(..., log_path=...)), or AprilTag-Detection flight logs
(columns t, state, rc_lr, rc_fb, rc_ud, rc_yaw, tof, yaw). Fly varied commands (turns of
different rc, climbs, descents, forward/sideways moves) for a useful fit. Horizontal values are
only fitted when the log has x, y (cm) from a motion-capture system.

The drift (how the drone moves by itself) is not fitted here: measure it hovering with rc 0
(e.g. with a downward mirror, tellolab.FloorFlow) and set drift_right_cm_s / random_drift_cm_s.
See docs/drone-model.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as st
import sys
from collections import deque
from dataclasses import fields, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tellosim.drone import stick  # noqa: E402
from tellosim.model import DroneModel, load_model  # noqa: E402

WINDOW_S = 1.0
SUB_DT = 0.02
GROUND_STATES = {"IDLE", "TAKEOFF", "LANDING", "EMERGENCY_LAND", "DONE", "KILLED"}

GRID = {  # channel: {field: (low, high, step)}
    "yaw": {"yaw_gain": (0.4, 1.2, 0.05), "yaw_deadband": (4, 20, 1), "yaw_response_s": (0.05, 0.5, 0.05),
            "yaw_carry": (0.0, 0.8, 0.1), "command_delay_s": (0.0, 0.2, 0.02)},
    "vertical": {"up_gain": (0.2, 1.2, 0.05), "down_gain": (0.05, 0.8, 0.05), "vertical_deadband": (0, 10, 1),
                 "vertical_response_s": (0.2, 1.2, 0.1)},
    "horizontal": {"forward_gain": (0.8, 2.2, 0.05), "deadband": (0, 10, 1), "response_s": (0.2, 1.2, 0.05)},
}


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


class Flight:
    def __init__(self, path: Path) -> None:
        self.name = path.name
        rows = list(csv.DictReader(path.open()))
        self.t = [_f(r.get("t")) for r in rows]
        self.rc = [tuple(_f(r.get(k)) or 0.0 for k in ("rc_lr", "rc_fb", "rc_ud", "rc_yaw")) for r in rows]
        tof = [_f(r.get("tof")) for r in rows]
        self.alt = [None if x is None else x - 10 for x in tof]
        if "flying" in rows[0]:
            self.flying = [r.get("flying") in ("1", "True", "true") for r in rows]
        else:  # AprilTag-Detection logs
            self.flying = [r.get("state", "") not in GROUND_STATES for r in rows]
        self.x = [_f(r.get("x")) for r in rows]
        self.y = [_f(r.get("y")) for r in rows]
        self.has_pos = sum(v is not None for v in self.x) > 10
        self.yaw, prev, off = [], None, 0.0  # continuous heading (no jump from 180 to -180)
        for r in rows:
            y = _f(r.get("yaw"))
            if y is None:
                self.yaw.append(None)
                continue
            if prev is not None:
                off -= 360 * round((y - prev) / 360)
            prev = y
            self.yaw.append(y + off)
        flips = [self.t[i] for i, r in enumerate(rows) if abs(_f(r.get("pitch")) or 0) > 30 or abs(_f(r.get("roll")) or 0) > 30]
        self.ok = [self.flying[i] and self.t[i] is not None and self.yaw[i] is not None and tof[i] is not None
                   and 20 < tof[i] < 400 and not any(-0.5 <= self.t[i] - ft <= 3.0 for ft in flips)
                   for i in range(len(rows))]
        self.takeoff_cm = next((self.alt[i] for i in range(len(rows)) if self.ok[i]), None)


def replay(fl: Flight, m: DroneModel):
    """Integrate the model with the flight's rc (no drift). Per row: accumulated turn (deg),
    climb (cm), forward and right (cm, drone frame)."""
    hist: deque = deque([(-1e9, (0, 0, 0, 0))])
    rate = carry = 0.0
    v = [0.0, 0.0, 0.0]
    Y, Z, A, R = [0.0], [0.0], [0.0], [0.0]
    for i in range(len(fl.t) - 1):
        t0, t1 = fl.t[i], fl.t[i + 1]
        y, z, a, r = Y[-1], Z[-1], A[-1], R[-1]
        if t0 is None or t1 is None or t1 <= t0 or not fl.flying[i]:
            rate = carry = 0.0
            v = [0.0, 0.0, 0.0]
        else:
            hist.append((t0, fl.rc[i]))
            n = max(1, round((t1 - t0) / SUB_DT))
            h = (t1 - t0) / n
            for k in range(n):
                lim = t0 + (k + 1) * h - m.command_delay_s
                while len(hist) > 1 and hist[1][0] <= lim:
                    hist.popleft()
                lr, fb, ud, yr = hist[0][1] if hist[0][0] <= lim else (0, 0, 0, 0)
                want = [stick(fb, m.deadband, m.forward_gain), stick(lr, m.deadband, m.forward_gain),
                        stick(ud, m.vertical_deadband, m.up_gain if ud > 0 else m.down_gain)]
                if want[0] and want[1]:
                    want[0], want[1] = want[0] * m.combo_factor, want[1] * m.combo_factor
                    want[2] -= m.combo_sink_cm_s
                if want[0]:
                    want[2] -= m.forward_sink_cm_s
                kh, kz = min(1, h / max(m.response_s, 1e-3)), min(1, h / max(m.vertical_response_s, 1e-3))
                v = [v[0] + (want[0] - v[0]) * kh, v[1] + (want[1] - v[1]) * kh, v[2] + (want[2] - v[2]) * kz]
                w = stick(yr, m.yaw_deadband, m.yaw_gain)
                rate += (w - rate) * min(1, h / max(m.yaw_response_s, 1e-3))
                carry += (m.yaw_carry * w - carry) * min(1, h / max(m.yaw_carry_s, 1e-3))
                y += (rate + carry) * h
                z += v[2] * h
                a += (v[0] + m.drift_forward_cm_s) * h
                r += (v[1] + m.drift_right_cm_s) * h
        Y.append(y), Z.append(z), A.append(a), R.append(r)
    return Y, Z, A, R


def windows(fl: Flight):
    i, n = 0, len(fl.t)
    while i < n:
        if not fl.ok[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and fl.ok[j + 1] and fl.t[j + 1] - fl.t[i] <= WINDOW_S + 1e-6:
            j += 1
        if fl.t[j] - fl.t[i] >= WINDOW_S * 0.9:
            yield i, j
        i = j if j > i else i + 1


def errors(flights, m: DroneModel) -> dict:
    out = {"yaw": [], "vertical": [], "horizontal": []}
    for fl in flights:
        Y, Z, A, R = replay(fl, m)
        for i, j in windows(fl):
            out["yaw"].append((Y[j] - Y[i]) - (fl.yaw[j] - fl.yaw[i]))
            if abs(fl.alt[j] - fl.alt[i]) < 60:  # no false ToF jumps
                out["vertical"].append((Z[j] - Z[i]) - (fl.alt[j] - fl.alt[i]))
            if fl.has_pos and None not in (fl.x[i], fl.x[j]) and abs(fl.yaw[j] - fl.yaw[i]) < 5:
                ps = math.radians(fl.yaw[i])
                dx, dy = fl.x[j] - fl.x[i], fl.y[j] - fl.y[i]
                fwd, rgt = dx * math.sin(ps) + dy * math.cos(ps), dx * math.cos(ps) - dy * math.sin(ps)
                out["horizontal"].append(math.hypot((A[j] - A[i]) - fwd, (R[j] - R[i]) - rgt))
    return out


def mean_abs(xs):
    return sum(abs(x) for x in xs) / len(xs) if xs else float("nan")


def fit(flights, m: DroneModel, channels) -> DroneModel:
    for ch in channels:
        best = mean_abs(errors(flights, m)[ch])
        if math.isnan(best):
            print(f"  {ch}: no data, unchanged")
            continue
        for _ in range(3):  # coordinate descent
            for k, (lo, hi, step) in GRID[ch].items():
                for val in [round(lo + i * step, 4) for i in range(int(round((hi - lo) / step)) + 1)]:
                    q = replace(m, **{k: val})
                    e = mean_abs(errors(flights, q)[ch])
                    if e < best * 0.995:
                        best, m = e, q
        print(f"  {ch}: mean error {best:.2f}")
    return m


def report(flights, m: DroneModel, title: str) -> None:
    e = errors(flights, m)
    print(f"{title}:")
    units = {"yaw": "deg", "vertical": "cm", "horizontal": "cm"}
    for ch, xs in e.items():
        if xs:
            print(f"  {ch:10s} error per {WINDOW_S:.0f} s window: mean {mean_abs(xs):5.2f} {units[ch]} "
                  f"(median {st.median(abs(x) for x in xs):.2f}, {len(xs)} windows)")
        else:
            print(f"  {ch:10s} no data" + (" (needs x, y from motion capture)" if ch == "horizontal" else ""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", help="flight CSVs")
    ap.add_argument("--base", default="lab_healthy", help="model to start from (built-in name or JSON)")
    ap.add_argument("--fit", action="store_true", help="search the values that reproduce the flights best")
    ap.add_argument("--out", help="with --fit: write the fitted model JSON here")
    args = ap.parse_args()

    flights = [Flight(Path(p)) for p in args.logs]
    flights = [f for f in flights if sum(f.ok) > 20]
    if not flights:
        raise SystemExit("no usable flights (need rows in the air with a valid ToF)")
    print(f"{len(flights)} flights, {sum(sum(f.ok) for f in flights)} rows in the air")
    base = load_model(args.base)
    report(flights, base, f"model {args.base}")
    if not args.fit:
        return
    channels = ["yaw", "vertical"] + (["horizontal"] if any(f.has_pos for f in flights) else [])
    print("fitting " + ", ".join(channels) + " (takes a minute)...")
    fitted = fit(flights, base, channels)
    takeoffs = sorted(f.takeoff_cm for f in flights if f.takeoff_cm and f.takeoff_cm < 150)
    if len(takeoffs) >= 3:
        fitted = replace(fitted, takeoff_height_cm=(takeoffs[len(takeoffs) // 10], takeoffs[(9 * len(takeoffs)) // 10]))
    report(flights, fitted, "fitted model")
    changed = {f.name: getattr(fitted, f.name) for f in fields(DroneModel)
               if getattr(fitted, f.name) != getattr(base, f.name)}
    print("changed: " + (", ".join(f"{k}={v}" for k, v in changed.items()) or "nothing"))
    if args.out:
        data = {"base": args.base if not args.base.endswith(".json") else str(args.base),
                "description": f"Fitted with tools/fit_model.py on {len(flights)} flights, starting from {args.base}.",
                **{k: list(v) if isinstance(v, tuple) else v for k, v in changed.items()}}
        Path(args.out).write_text(json.dumps(data, indent=2) + "\n")
        print(f"written: {args.out}")


if __name__ == "__main__":
    main()
