"""What every example shares: simulator or real drone, camera parameters, and the window.

The only difference between flying in the simulator and flying the real Tello is which Tello
class you create (see make_tello). Everything else is the same code.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # run the examples without installing the package

from tellolab.vision import mirrored_intrinsics  # noqa: E402


def parser(description: str, world: str, camera_mode: str = "forward", model: str = "lab_healthy"):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--real", action="store_true", help="fly the real Tello (djitellopy) instead of the simulator")
    p.add_argument("--world", default=world, help="simulator map (built-in name or YAML file)")
    p.add_argument("--model", default=model, help="simulated drone: lab_healthy, lab_tello1, lab_tello7, ideal")
    p.add_argument("--camera-mode", default=camera_mode, choices=["forward", "mirror"],
                   help="camera mounting (mirror = a mirror clipped on the camera, looking down)")
    p.add_argument("--tilt", type=float, default=-10.3, help="mirror view tilt in degrees (measure it on your clip)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-window", action="store_true", help="no video window (simulator only: no STOP button)")
    p.add_argument("--timeout", type=float, default=120.0, help="give up and land after this many seconds")
    return p


def make_tello(args):
    """The one line that changes between simulator and real drone."""
    if args.real:
        from djitellopy import Tello
        return Tello()
    from tellosim import Tello
    return Tello(world=args.world, model=args.model, camera_mode=args.camera_mode, seed=args.seed,
                 mirror_tilt_deg=args.tilt)


class CameraInfo:
    """K and distortion of the image you process (already flipped back with a mirror), and how
    the camera is mounted."""

    def __init__(self, tello, args) -> None:
        self.mode = args.camera_mode
        self.tilt = args.tilt
        if hasattr(tello, "sim"):
            cam = tello.sim.camera
            self.K, self.dist = cam.detection_K
            self.pitch = cam.pitch_deg
            self.tilt = cam.mirror_tilt_deg
            self.mode = cam.mode
        else:  # the real drone: the lab calibration (recalibrate yours: docs/camera.md)
            data = json.loads((ROOT / "tellosim" / "cameras" / "lab_tello.json").read_text())
            K, dist = np.array(data["K"]), np.array(data["dist"])
            self.pitch = data.get("pitch_deg", 11.0)
            self.K, self.dist = mirrored_intrinsics(K, dist) if self.mode == "mirror" else (K, dist)


class Window:
    """The STOP window, or nothing with --no-window. show() returns "land", "kill", "quit" or None."""

    def __init__(self, args, title: str) -> None:
        self.win = None
        if not args.no_window:
            from tellolab.hmi import StopWindow
            self.win = StopWindow(title)
        elif args.real:
            raise SystemExit("--no-window is only for the simulator: with the real drone you need the STOP button")

    def show(self, bgr, lines, flying=True):
        return self.win.show(bgr, lines, flying) if self.win else None

    def close(self):
        if self.win:
            self.win.close()


def true_pose(tello):
    """The simulator's true pose (x, y, z, yaw), or None with the real drone."""
    return tello.sim.pose() if hasattr(tello, "sim") else None
