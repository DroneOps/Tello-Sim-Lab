"""Stay in front of an ArUco marker on a wall, with the forward camera.

    python examples/aruco_follow.py                     # simulator: map aruco_board, marker 0
    python examples/aruco_follow.py --id 3 --distance 100
    python examples/aruco_follow.py --real --size 15    # your printed marker size in cm

Every frame: detect the marker, solvePnP gives where it is (forward, right, up) in cm, and a
proportional command keeps it ``--distance`` ahead, centered. It only moves forward once it is
at the marker's height: the forward camera looks 11° down and sees only ~10° above the horizon.
The Pilot holds the heading.
"""
from __future__ import annotations

import math
import time

import cv2

from common import CameraInfo, Window, make_tello, parser, true_pose
from tellolab import Pilot, TargetCentering, TargetCenteringConfig, altitude_cm, with_min
from tellolab.vision import detect_markers, marker_position


def main() -> None:
    p = parser(__doc__.splitlines()[0], world="aruco_board", camera_mode="forward")
    p.add_argument("--id", type=int, default=0)
    p.add_argument("--size", type=float, default=15.0, help="marker size in cm (black square)")
    p.add_argument("--dictionary", default="DICT_4X4_50")
    p.add_argument("--distance", type=float, default=120.0)
    p.add_argument("--seconds", type=float, default=35.0)
    p.add_argument("--altitude", type=float, default=130.0, help="takeoff height: near the marker's")
    args = p.parse_args()

    tello = make_tello(args)
    tello.connect()
    tello.streamon()
    frames = tello.get_frame_read()
    cam = CameraInfo(tello, args)
    pilot = Pilot(tello, frame_read=frames)
    # keep a point (the marker, minus the distance) like TargetCentering keeps a drone over a tag:
    # proportional, braking with the target's velocity (the video is ~170 ms late)
    keep = TargetCentering(TargetCenteringConfig(kp=0.4, kd=1.0, min_rc=13, max_rc=22, tol_cm=8))
    win = Window(args, "aruco_follow")
    pilot.takeoff(altitude_cm=args.altitude)
    t0, last = time.monotonic(), None
    try:
        while pilot.flying and time.monotonic() - t0 < args.seconds:
            img = cv2.cvtColor(frames.frame, cv2.COLOR_RGB2BGR)
            found = [c for i, c in detect_markers(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), args.dictionary) if i == args.id]
            if found:
                f, r, u = marker_position(found[0], args.size, cam.K, cam.dist, "forward", cam.pitch)
                last = (f, r, u)
                # the camera looks 11° down and sees only ~10° above the horizon: get to the marker's
                # height first, or it leaves the top of the image while approaching
                ud = with_min(0.8 * u, 20, 30) if abs(u) > 10 else 0
                lr, fb, _ = keep.update(f - args.distance, r, time.monotonic(), tello.get_yaw())
                pilot.set_rc(lr, fb if abs(u) < 15 else 0, ud, 0)
                status = f"marker {args.id}: {f:.0f} ahead, {r:+.0f} right, {u:+.0f} up"
            else:
                keep.reset()
                pilot.set_rc(0, 0, 0, 20 if last is None else 0)  # never seen: turn slowly to find it
                status = f"marker {args.id} not seen"
            action = win.show(img, [status, f"alt {altitude_cm(tello.get_current_state())} cm  rc {pilot.sent}"])
            if action in ("land", "quit"):
                break
            if action == "kill":
                pilot.kill()
                break
            time.sleep(1 / 30)
    finally:
        pose = true_pose(tello)
        pilot.land()
        pilot.stop()
        win.close()
        if pose:
            m = next(o for o in tello.sim.world.objects if getattr(o, "id", None) == args.id and o.mount == "wall")
            dist = m.position[1] - pose[1]
            print(f"result: {dist:.0f} cm from the wall marker (wanted {args.distance:.0f}), "
                  f"{pose[0] - m.position[0]:+.0f} cm to its side; last seen {None if last is None else [round(v) for v in last]}")
        tello.end()


if __name__ == "__main__":
    main()
