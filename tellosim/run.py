"""Run an UNMODIFIED djitellopy program in the simulator.

    python -m tellosim.run my_program.py
    python -m tellosim.run --world color_gates --model lab_tello7 my_program.py --its-own-args
    python -m tellosim.run --headless --seconds 90 --record run.mp4 my_program.py
    python -m tellosim.run --keys t@4,l@40 my_keyboard_program.py

``import djitellopy`` inside the program gets tellosim instead, with the options given here. At
the end it prints what really happened: crashes, where it ended, gates flown through, distance to
a line.

Options:
  --world, --model, --camera, --camera-mode, --seed   simulation settings (see docs/how-it-works.md)
  --camera-set mirror_axis=h                          any camera field (docs/camera.md); repeat it
  --faults '{"takeoff": "no_ok"}'                     faults as JSON, or a .json file (docs/faults.md)
  --fog 300  or  --fog '{"visibility_cm": 300, "patches": 0.6}'   fog or smoke, or a .json file (docs/maps.md)
  --headless       no windows: cv2.imshow does nothing and cv2.waitKey returns -1 (no keys)
  --viewer         also show the map from above
  --record FILE    record the map + camera view to an mp4
  --seconds N      stop the program after N seconds (as if you pressed Ctrl+C)
  --keys t@4,q@40  press keys for the program: cv2.waitKey returns 't' at 4 s, 'q' at 40 s
                   (for programs that take off or land from the keyboard)
"""
from __future__ import annotations

import _thread
import argparse
import json
import os
import runpy
import sys
import threading
import time
import types

import tellosim
from .report import summary

INSTANCES: list = []


def _install_fake_djitellopy(options: dict, viewer: bool, record: str | None) -> None:
    class Tello(tellosim.Tello):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **{**options, **kwargs})
            self._path = []
            INSTANCES.append(self)
            if viewer or record:
                self.sim.viewer(show=viewer, record=record)
            threading.Thread(target=self._sample, daemon=True).start()

        def _sample(self):
            while not self.sim._stop.is_set():
                self._path.append(self.sim.pose()[:3])
                time.sleep(0.1)

    mod, sub = types.ModuleType("djitellopy"), types.ModuleType("djitellopy.tello")
    for m in (mod, sub):
        m.Tello, m.TelloException = Tello, tellosim.TelloException
        m.BackgroundFrameRead = tellosim.BackgroundFrameRead
    mod.tello = sub
    sys.modules["djitellopy"], sys.modules["djitellopy.tello"] = mod, sub


def _headless() -> None:
    import cv2

    def nothing(*a, **k):
        return None

    def wait_key(delay=0):
        time.sleep(max(delay, 1) / 1000)
        return -1
    for name in ("imshow", "namedWindow", "destroyAllWindows", "destroyWindow", "setMouseCallback",
                 "moveWindow", "resizeWindow", "setWindowProperty", "setWindowTitle"):
        setattr(cv2, name, nothing)
    cv2.waitKey = wait_key
    cv2.pollKey = lambda: -1


def _press_keys(keys: list[tuple[str, float]]) -> None:
    """cv2.waitKey / pollKey return each key once its time (s from the start) has come."""
    import cv2
    t0, pending = time.monotonic(), sorted(keys, key=lambda k: k[1])
    real_wait, real_poll = cv2.waitKey, getattr(cv2, "pollKey", None)

    def due():
        if pending and time.monotonic() - t0 >= pending[0][1]:
            key = pending.pop(0)[0]
            print(f"[tellosim.run] key '{key}'", flush=True)
            return ord(key)
        return None

    def wait_key(delay=0):
        k = due()
        return k if k is not None else real_wait(delay)

    def poll_key():
        k = due()
        return k if k is not None else (real_poll() if real_poll else -1)
    cv2.waitKey, cv2.pollKey = wait_key, poll_key


def _json_arg(text: str):
    """--fog / --faults: JSON text, a number, or the path of a .json file (handy on Windows, where
    quotes inside a command are hard to get right)."""
    if os.path.isfile(text):
        with open(text, encoding="utf-8") as f:
            return json.load(f)
    return json.loads(text)


def _value(text: str):
    try:
        return json.loads(text)
    except ValueError:
        return text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", default="lab_grid")
    ap.add_argument("--model", default="lab_healthy")
    ap.add_argument("--camera", default="lab_tello")
    ap.add_argument("--camera-mode", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--camera-set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--faults", default=None, help="JSON, e.g. '{\"takeoff\": \"collapse\"}'")
    ap.add_argument("--keys", default="", help="key@seconds,... e.g. t@4,q@40")
    ap.add_argument("--fog", default=None, help="visibility in cm, or JSON with the Fog fields")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--record", default=None)
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("script")
    ap.add_argument("script_args", nargs=argparse.REMAINDER)
    a = ap.parse_args()

    options = {"world": a.world, "model": a.model, "camera": a.camera, "seed": a.seed}
    if a.camera_mode:
        options["camera_mode"] = a.camera_mode
    for kv in a.camera_set:
        key, _, value = kv.partition("=")
        options[key.strip()] = _value(value.strip())
    if a.faults:
        options["faults"] = _json_arg(a.faults)
    if a.fog:
        options["fog"] = _json_arg(a.fog)
    _install_fake_djitellopy(options, a.viewer, a.record)
    if a.headless:
        _headless()
    if a.keys:
        keys = []
        for item in a.keys.split(","):
            key, _, t = item.strip().rpartition("@")
            if len(key) != 1:
                ap.error(f"--keys: '{item}' should be one character, @, seconds (e.g. t@4)")
            keys.append((key, float(t)))
        _press_keys(keys)
    if a.seconds:
        threading.Timer(a.seconds, _thread.interrupt_main).start()
    script = os.path.abspath(a.script)
    sys.argv = [script, *a.script_args]
    sys.path.insert(0, os.path.dirname(script))
    print(f"[tellosim.run] {os.path.basename(script)} in the simulator: world {a.world}, model {a.model}"
          + (f", camera {a.camera_mode}" if a.camera_mode else "") + (f", fog {a.fog}" if a.fog else ""), flush=True)
    try:
        runpy.run_path(script, run_name="__main__")
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        for t in INSTANCES:
            print("[tellosim.run] " + "\n[tellosim.run] ".join(summary(t.sim, t._path)), flush=True)
            try:
                t.sim.close()
            except Exception:
                pass
        os._exit(0)


if __name__ == "__main__":
    main()
