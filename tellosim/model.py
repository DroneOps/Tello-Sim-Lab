"""How the simulated drone responds: the drone model.

Every number here was measured on the DroneOps lab Tellos (standard DJI Tello) in real flights
and fitted with ``tools/fit_model.py``. See ``docs/drone-model.md`` for where each one comes
from and how to refit it for your own drone.

Built-in models (``tellosim/models/*.json``):

- ``lab_healthy``: a healthy lab Tello (fitted on 14 flights, Nov 2026).
- ``lab_tello1``: lab Tello #1, which drifts ~3.7 cm/s to its right.
- ``lab_tello7``: lab Tello #7, which never holds still: ~11.7 cm/s to its right.
- ``ideal``: an optimistic model with no drift or lag, for debugging your logic.

A model JSON can start from another one with ``"base": "lab_healthy"`` and change only a few
fields.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent / "models"


@dataclass(frozen=True)
class DroneModel:
    """Response of the drone to rc commands and its own misbehavior. Units: cm, s, degrees.

    The rc -> speed law is ``speed = gain * (|rc| - deadband)`` (with the sign of rc, and 0
    inside the deadband); the drone reaches that speed with a first-order lag of
    ``response_s``."""

    description: str = ""

    # --- horizontal (forward/back and left/right), cm/s per rc unit above the deadband
    forward_gain: float = 1.45
    deadband: float = 5.0
    response_s: float = 0.75
    # asking forward AND sideways at the same time slows both (lab: fb 15 + lr 15 -> ~10 and ~3 cm/s)
    combo_factor: float = 1.0
    # ...and makes it sink this much (cm/s)
    combo_sink_cm_s: float = 8.0
    # it sinks this much when flying forward, even asking to climb a little (cm/s)
    forward_sink_cm_s: float = 1.0

    # --- vertical: climbing is much faster than descending (lab: rc 40 up ~40 cm/s, rc -40 down ~7)
    up_gain: float = 0.4
    down_gain: float = 0.15
    vertical_deadband: float = 3.0
    vertical_response_s: float = 0.6

    # --- yaw, deg/s per rc unit above the deadband
    yaw_gain: float = 0.75
    yaw_deadband: float = 14.0
    yaw_response_s: float = 0.1
    # part of the turn that keeps going after releasing the stick, fading in yaw_carry_s
    # (lab: asked to stop at 88°, it kept rotating to 101°)
    yaw_carry: float = 0.4
    yaw_carry_s: float = 2.0

    # --- delays
    command_delay_s: float = 0.04  # rc takes this long to act
    video_delay_s: float = 0.17  # the image arrives this late (measured filming a clock)

    # --- the drone moves by itself
    random_drift_cm_s: float = 5.4  # random wind-like wandering (std of the slow random walk)
    drift_right_cm_s: float = 0.0  # constant drift in the drone frame (+ = to its right)
    drift_forward_cm_s: float = 0.0
    # when turning hard it slides to a random side (lab: 26–42 cm during a 90° turn)
    turn_drift_cm_s: float = 12.0

    # --- takeoff: climbs to a peak, settles between the two heights, answers "ok" in that window
    takeoff_height_cm: tuple = (65.0, 74.0)
    takeoff_peak_cm: float = 9.0
    takeoff_ok_s: tuple = (4.5, 7.8)
    takeoff_climb_cm_s: float = 30.0
    land_speed_cm_s: float = 40.0

    # --- flips: the Tello refuses them under this battery; after the rotation it FALLS (~50 cm/s)
    # for 1–3 s and its own height estimate does not notice (lab: from 130 cm to the floor)
    flip_min_battery: float = 50.0
    flip_fall_cm_s: float = 50.0
    flip_fall_s: tuple = (1.0, 3.0)

    # --- sensors
    tof_noise_cm: float = 2.0  # the ToF reads height + 10 cm (+ noise); real altitude = tof - 10
    height_bias_max_cm: float = 40.0  # 'h' (the Tello's own height) wanders up to this much: don't trust it

    # --- battery and temperature
    battery_pct_per_min: float = 6.0
    temp_start_c: float = 65.0
    temp_c_per_min: float = 6.5

    # --- move_* / rotate_* commands (the Tello's own position control). NOT measured in the lab.
    move_speed_cm_s: float = 30.0
    rotate_speed_deg_s: float = 60.0

    def with_changes(self, **changes) -> DroneModel:
        """A copy with some fields changed, e.g. ``model.with_changes(drift_right_cm_s=8)``."""
        return replace(self, **changes)

    def to_dict(self) -> dict:
        return asdict(self)


def _read(path: Path) -> dict:
    data = json.loads(path.read_text())
    base = data.pop("base", None)
    merged = _read(_resolve(base)) if base else {}
    merged.update(data)
    return merged


def _resolve(name_or_path: str | Path) -> Path:
    p = Path(name_or_path)
    if p.suffix == ".json" and p.exists():
        return p
    builtin = MODELS_DIR / f"{name_or_path}.json"
    if builtin.exists():
        return builtin
    names = sorted(x.stem for x in MODELS_DIR.glob("*.json"))
    raise FileNotFoundError(f"drone model {name_or_path!r} not found (built-in: {', '.join(names)})")


def load_model(model: str | Path | dict | DroneModel = "lab_healthy") -> DroneModel:
    """A built-in model name, a JSON file, a dict of fields, or a DroneModel."""
    if isinstance(model, DroneModel):
        return model
    data = dict(model) if isinstance(model, dict) else _read(_resolve(model))
    if "base" in data:
        base = _read(_resolve(data.pop("base")))
        base.update(data)
        data = base
    known = {f.name for f in fields(DroneModel)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown drone model fields: {sorted(unknown)}")
    for k in ("takeoff_height_cm", "takeoff_ok_s", "flip_fall_s"):
        if isinstance(data.get(k), list):
            data[k] = tuple(data[k])
    return DroneModel(**data)
