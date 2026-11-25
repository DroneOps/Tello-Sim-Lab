"""Find a landing pad, get over it and land, with a downward mirror.

    python examples/landing_pad.py                      # simulator: the red-border "H" pad 1.5 m ahead
    python examples/landing_pad.py --pad aruco          # the raised platform with an ArUco marker
    python examples/landing_pad.py --real

It flies forward until it sees the pad (the red border, or the marker), turns the pad's pixel
into a floor point in cm (vision.floor_point_from_pixel / marker_position), gets over it with
TargetCentering, descends to 40 cm and lands.
"""
from __future__ import annotations

import math
import time

import cv2
import numpy as np

from common import CameraInfo, Window, make_tello, parser, true_pose
from tellolab import Pilot, TargetCentering, TargetCenteringConfig, altitude_cm
from tellolab.vision import detect_markers, floor_point_from_pixel, marker_position, unmirror


def find_pad(bgr, cam, height_cm, kind, size_cm=60.0,
             color_ranges=(((0, 140, 80), (8, 255, 255)), ((170, 140, 80), (179, 255, 255)))):
    """(forward, right) of the pad center in cm, or None.

    The mirror sees only ~20–30 cm ahead (the view tilts back), so the front of the pad is often
    cut by the edge of the image and the middle of the visible red is shifted backwards. The
    back edge is always in view: the center is half a pad ahead of it, and sideways it is
    halfway between the left and right edges."""
    if kind == "aruco":
        found = detect_markers(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), "DICT_4X4_50", ids={0})
        if not found:
            return None
        f, r, _ = marker_position(found[0][1], 30, cam.K, cam.dist, "mirror", view_forward_deg=cam.tilt)
        return f, r
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in color_ranges:
        mask |= cv2.inRange(hsv, lo, hi)
    vs, us = np.nonzero(mask[::6, ::6])
    if len(us) < 60:
        return None
    pts = [floor_point_from_pixel(float(u * 6), float(v * 6), cam.K, cam.dist, height_cm, "mirror",
                                  view_forward_deg=cam.tilt) for u, v in zip(us, vs)]
    pts = np.array([q for q in pts if q is not None])
    back = np.percentile(pts[:, 0], 1)
    left, right = np.percentile(pts[:, 1], 1), np.percentile(pts[:, 1], 99)
    return back + size_cm / 2, (left + right) / 2


def main() -> None:
    p = parser(__doc__.splitlines()[0], world="landing_pad", camera_mode="mirror")
    p.add_argument("--pad", choices=["h", "aruco"], default="h")
    p.add_argument("--altitude", type=float, default=100.0)
    args = p.parse_args()

    tello = make_tello(args)
    tello.connect()
    tello.streamon()
    frames = tello.get_frame_read()
    cam = CameraInfo(tello, args)
    pilot = Pilot(tello, frame_read=frames)
    center = TargetCentering()
    win = Window(args, "landing_pad")
    pilot.takeoff(altitude_cm=args.altitude)
    if args.pad == "aruco" and not args.real:
        pilot.turn_by(90)  # the raised pad is to the right of the H pad: go there first
        pilot._wait(4)
        pilot.turn_by(-90)
        pilot._wait(4)
    # search (forward) -> high (climb to 150 cm: the whole pad width fits in the view) -> low
    # (center again, stricter) -> land straight down (the real Tello drifts a few cm landing).
    fine = TargetCentering(TargetCenteringConfig(tol_cm=5.0, vel_tol_cm_s=5.0))
    t0, phase, t_settled, t_low = time.monotonic(), "search", None, None
    try:
        while pilot.flying and time.monotonic() - t0 < args.timeout:
            view = unmirror(cv2.cvtColor(frames.frame, cv2.COLOR_RGB2BGR))
            state = tello.get_current_state()
            now, alt, yaw = time.monotonic(), altitude_cm(state), state.get("yaw", 0)
            pad = find_pad(view, cam, alt, args.pad)
            if pad is None:
                pilot.set_rc(0, 15 if phase == "search" else 0, 0, 0)
                status = "looking for the pad"
                center.reset()
                fine.reset()
            else:
                ctl = fine if phase == "low" else center
                lr, fb, settled = ctl.update(pad[0], pad[1], now, yaw)
                pilot.set_rc(lr, fb, 0, 0)
                t_settled = (t_settled or now) if settled else None
                status = f"{phase}: pad at {pad[0]:+.0f} forward, {pad[1]:+.0f} right"
                if phase == "search":
                    pilot.set_altitude(150)
                    phase, t_low = "high", now
                elif phase == "high" and alt >= 140:
                    phase, t_low, t_settled = "low", now, None
                elif phase == "low" and ((t_settled and now - t_settled > 1.0) or now - t_low > 10):
                    print(f"over the pad at {alt} cm: landing")
                    break
            action = win.show(view, [status, f"alt {alt} cm  rc {pilot.sent}"])
            if action in ("land", "quit"):
                break
            if action == "kill":
                pilot.kill()
                break
            time.sleep(1 / 30)
    finally:
        pilot.land()
        pilot.stop()
        win.close()
        pose = true_pose(tello)
        if pose:
            name = "pad_aruco" if args.pad == "aruco" else "pad_h"
            pad = tello.sim.world.find(name)
            print(f"result: landed {math.hypot(pose[0] - pad.position[0], pose[1] - pad.position[1]):.0f} cm "
                  f"from the center of {name}")
        tello.end()


if __name__ == "__main__":
    main()
