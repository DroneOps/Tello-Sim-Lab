"""Fly through colored gates in order, with the forward camera.

    python examples/color_gate.py                       # simulator, map color_gates
    python examples/color_gate.py --colors orange blue  # other order or other gates
    python examples/color_gate.py --real

For each gate: find its color (HSV threshold), find the opening (a hole inside the colored
contour), turn and climb/descend until the opening is centered on the horizon row (things at the
camera's height), fly forward; when it is so close that the top bar leaves the view, go straight
through steering by the posts. The Pilot holds altitude and heading in between.
"""
from __future__ import annotations

import math
import time

import cv2
import numpy as np

from common import CameraInfo, Window, make_tello, parser, true_pose
from tellolab import Pilot, altitude_cm, with_min

# OpenCV hue (0-179) ranges; high saturation so the blue floor tape (S ~150) is not a "blue gate"
HSV = {"orange": [((4, 170, 90), (22, 255, 255))], "green": [((40, 170, 60), (85, 255, 255))],
       "blue": [((100, 175, 60), (130, 255, 255))],
       "red": [((0, 170, 80), (4, 255, 255)), ((170, 170, 80), (179, 255, 255))],
       "yellow": [((22, 160, 100), (35, 255, 255))]}


def find_opening(bgr, color):
    """(cx, cy, width_px, found_hole) of the gate opening, or None if the color is not seen."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in HSV[color]:
        mask |= cv2.inRange(hsv, lo, hi)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    if mask.sum() / 255 < 150:
        return None
    contours, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    holes = [c for c, h in zip(contours, hier[0]) if h[3] >= 0]
    if holes:
        hole = max(holes, key=cv2.contourArea)
        if cv2.contourArea(hole) > 300:
            x, y, w, h = cv2.boundingRect(hole)
            return x + w / 2, y + h / 2, w, True
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return x + w / 2, y + h / 2, w, False


def main() -> None:
    p = parser(__doc__.splitlines()[0], world="color_gates", camera_mode="forward")
    p.add_argument("--colors", nargs="+", default=["orange", "green", "blue"])
    p.add_argument("--altitude", type=float, default=100.0)
    args = p.parse_args()

    tello = make_tello(args)
    tello.connect()
    tello.streamon()
    frames = tello.get_frame_read()
    cam = CameraInfo(tello, args)
    horizon = cam.K[1, 2] - cam.K[1, 1] * math.tan(math.radians(cam.pitch))  # row of things at camera height
    pilot = Pilot(tello, frame_read=frames)
    win = Window(args, "color_gate")
    pilot.takeoff(altitude_cm=args.altitude)
    pilot._wait(3)
    # per gate: "find" (turn until the color is seen) -> "frame" (whole frame in view: align and
    # approach) -> "through" (the top bar left the view: close and aligned, go straight) -> next.
    # With the camera tilted 11° down, the top bar of a gate is out of view when closer than ~2 m
    # at the gate's height: if it sees the color but not the opening, it climbs to see the frame.
    t0, k, phase, t_phase, t_gone, path = time.monotonic(), 0, "find", time.monotonic(), None, []
    was_aligned = False
    try:
        while pilot.flying and k < len(args.colors) and time.monotonic() - t0 < args.timeout:
            img = cv2.cvtColor(frames.frame, cv2.COLOR_RGB2BGR)
            now, color = time.monotonic(), args.colors[k]
            W = img.shape[1]
            gate = find_opening(img, color)
            if phase == "find":
                if gate:
                    phase, t_phase = "frame", now
                pilot.set_rc(0, 0, 0, 25)  # search by turning
                status = f"looking for the {color} gate"
            elif phase == "frame":
                if gate is None:
                    phase = "find"
                    status = "lost it"
                else:
                    cx, cy, w, hole = gate
                    ex = cx - W / 2
                    yaw = with_min(0.08 * ex, 20, 35) if abs(ex) > 25 else 0
                    if hole:
                        ey = cy - horizon
                        ud = -with_min(0.2 * ey, 20, 30) if abs(ey) > 25 else 0
                        aligned = abs(ex) < 60 and abs(ey) < 40
                        fb = 20 if aligned else 0
                        status = f"gate {k + 1} ({color}): opening {w:.0f} px, off {ex:+.0f}, {ey:+.0f} px"
                        t_phase, was_aligned = now, aligned
                    elif now - t_phase < 1.5 and was_aligned and w > 0.4 * W:
                        # it had the opening centered and is now too close to see the top bar: go
                        ud, fb = 0, 0
                        phase, t_gone = "through", None
                        status = "going through"
                    elif w > 0.4 * W:  # close but not aligned: back up to see the whole frame again
                        ud, fb = 0, -15
                        status = f"gate {k + 1} ({color}): backing up to see the frame"
                    else:  # no opening yet: climb to see the whole frame
                        ud, fb = (25 if altitude_cm(tello.get_current_state()) < 170 else 0), 0
                        status = f"gate {k + 1} ({color}): climbing to see the frame"
                    pilot.set_rc(0, fb, ud, yaw)
            else:  # through: straight on, steering by the posts, until the gate leaves the view
                if gate is None or gate[2] > 0.95 * W:
                    t_gone = t_gone or now
                else:
                    t_gone = None
                yaw = with_min(0.05 * (gate[0] - W / 2), 20, 30) if gate and abs(gate[0] - W / 2) > 40 else 0
                pilot.set_rc(0, 25, 0, yaw if t_gone is None else 0)
                status = f"going through gate {k + 1} ({color})"
                if t_gone and now - t_gone > 1.5:
                    print(f"gate {k + 1} ({color}) passed at {now - t0:.0f} s")
                    k, phase, t_phase = k + 1, "find", now
            pose = true_pose(tello)
            if pose:
                path.append(pose[:3])
            action = win.show(img, [status, f"alt {altitude_cm(tello.get_current_state())} cm  rc {pilot.sent}"])
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
        if path and hasattr(tello, "sim"):
            report(tello, path)
        tello.end()


def report(tello, path) -> None:
    """Simulator only: which gates it really flew through, and crashes."""
    from tellosim.objects import Gate, compass
    gates = [o for o in tello.sim.world.objects if isinstance(o, Gate)]
    for g in gates:
        c, n = np.array(g.position, float), compass(g.facing_deg)
        right = np.cross(np.array([0, 0, 1.0]), n)
        through = False
        for a, b in zip(path, path[1:]):
            da, db = (np.array(a) - c) @ n, (np.array(b) - c) @ n
            if da < 0 <= db:
                q = np.array(a) + (np.array(b) - np.array(a)) * (-da / (db - da)) - c
                r = math.hypot(q @ right, q[2]) if g.shape == "circle" else max(abs(q @ right) - g.width_cm / 2,
                                                                                 abs(q[2]) - g.height_cm / 2)
                through = through or (r < g.width_cm / 2 if g.shape == "circle" else r < 0)
        print(f"result: {g.name or g.color} {'flown through' if through else 'NOT flown through'}")
    print(f"result: crash: {tello.sim.crash or 'none'}")


if __name__ == "__main__":
    main()
