"""Follow a dark line on the floor with a downward mirror.

    python examples/line_follower.py                    # simulator, map line_follower
    python examples/line_follower.py --model lab_tello7 # a drone that drifts
    python examples/line_follower.py --real             # the real Tello with the mirror clip

Every frame: flip the image back, take the dark pixels, turn them into floor points in cm
(vision.floor_point_from_pixel, using the ToF height), fit a straight line to them, and steer:
yaw with the line's angle, sideways with its distance from the drone, slow forward. The Pilot
holds the altitude.

With the lab mirror the camera sees only ~20 cm ahead and ~60 behind (the view tilts towards
the back), so it follows slowly and looks mostly at the line it is already on.
"""
from __future__ import annotations

import math
import time

import cv2
import numpy as np

from common import CameraInfo, Window, make_tello, parser, true_pose
from tellolab import Pilot, altitude_cm, with_min
from tellolab.vision import floor_point_from_pixel, unmirror


def line_in_floor(gray, cam, height_cm, dark=70):
    """(angle_deg, offset_cm) of the line relative to the drone, or None. angle > 0: the line
    turns right ahead; offset > 0: the line is to the right of the drone."""
    ys, xs = np.nonzero(gray[::8, ::8] < dark)
    if len(xs) < 15:
        return None
    pts = []
    for u, v in zip(xs * 8, ys * 8):
        p = floor_point_from_pixel(float(u), float(v), cam.K, cam.dist, height_cm, "mirror", view_forward_deg=cam.tilt)
        if p and -45 < p[0] < 25:
            pts.append(p)
    if len(pts) < 15:
        return None
    pts = np.array(pts)  # (forward, right)
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c)
    d = vt[0] if vt[0][0] > 0 else -vt[0]  # direction of the line, pointing forward
    angle = math.degrees(math.atan2(d[1], d[0]))
    offset = c[1] - c[0] * d[1] / d[0]  # lateral distance of the line at forward = 0
    return angle, offset


def main() -> None:
    p = parser(__doc__.splitlines()[0], world="line_follower", camera_mode="mirror")
    p.add_argument("--speed", type=int, default=15, help="forward rc")
    p.add_argument("--altitude", type=float, default=70.0)
    p.add_argument("--seconds", type=float, default=90.0)
    args = p.parse_args()

    tello = make_tello(args)
    tello.connect()
    tello.streamon()
    frames = tello.get_frame_read()
    cam = CameraInfo(tello, args)
    pilot = Pilot(tello, frame_read=frames)
    win = Window(args, "line_follower")
    pilot.takeoff(altitude_cm=args.altitude)
    t0, lost_since, path = time.monotonic(), None, []
    try:
        while pilot.flying and time.monotonic() - t0 < args.seconds:
            view = unmirror(cv2.cvtColor(frames.frame, cv2.COLOR_RGB2BGR))
            gray = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
            state = tello.get_current_state()
            line = line_in_floor(gray, cam, altitude_cm(state))
            if line is None:
                lost_since = lost_since or time.monotonic()
                pilot.set_rc(0, 0, 0, 0)  # stop and hold until it sees the line again
                status = "line lost"
                if time.monotonic() - lost_since > 5:
                    print("line lost for 5 s: landing")
                    break
            else:
                lost_since = None
                angle, offset = line
                yaw = with_min(1.2 * angle, 20, 40) if abs(angle) > 6 else 0
                lr = with_min(0.8 * offset, 10, 20) if abs(offset) > 4 else 0
                fb = args.speed if abs(angle) < 25 else 0
                pilot.set_rc(lr, fb, 0, yaw)
                status = f"line: angle {angle:+.0f} deg, offset {offset:+.0f} cm"
            pose = true_pose(tello)
            if pose:
                path.append(pose[:2])
            action = win.show(view, [status, f"alt {altitude_cm(state)} cm  rc {pilot.sent}"])
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
    """Simulator only: how far from the line it flew, and how much of it it covered."""
    line = next(o for o in tello.sim.world.objects if o.__class__.__name__ == "FloorLine")
    pts = np.array(line.points, float)
    segs = list(zip(pts[:-1], pts[1:]))
    lengths = np.array([np.linalg.norm(b - a) for a, b in segs])

    def nearest(q):
        best = (1e9, 0.0)
        for k, (a, b) in enumerate(segs):
            t = float(np.clip((q - a) @ (b - a) / max((b - a) @ (b - a), 1e-9), 0, 1))
            d = float(np.linalg.norm(q - (a + t * (b - a))))
            if d < best[0]:
                best = (d, lengths[:k].sum() + t * lengths[k])
        return best
    near = [nearest(np.array(q)) for q in path[30:]]
    dist = [d for d, _ in near]
    covered = max(s for _, s in near) / lengths.sum() if near else 0
    print(f"result: {100 * covered:.0f}% of the line covered; distance to the line: median {np.median(dist):.0f} cm, "
          f"max {max(dist):.0f} cm")


if __name__ == "__main__":
    main()
