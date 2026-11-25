"""Hover in place: altitude, heading and (with a downward mirror) position hold.

    python examples/hover_hold.py                       # simulated lab Tello #7 (it drifts ~12 cm/s)
    python examples/hover_hold.py --no-flow             # same without position hold: watch it go
    python examples/hover_hold.py --real                # the real drone (mirror clipped on the camera)

With the mirror, FloorFlow measures how far it moved over the floor and PositionHold brings it
back. Without a mirror (--camera-mode forward) only altitude and heading are held: the Tello's
own downward sensor does the rest.
"""
from __future__ import annotations

import math
import time

import cv2

from common import CameraInfo, Window, make_tello, parser, true_pose
from tellolab import FloorFlow, Pilot, altitude_cm, point_below_drone
from tellolab.vision import unmirror


def main() -> None:
    p = parser(__doc__.splitlines()[0], world="lab_grid", camera_mode="mirror", model="lab_tello7")
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--altitude", type=float, default=100.0)
    p.add_argument("--no-flow", action="store_true", help="no position hold")
    args = p.parse_args()

    tello = make_tello(args)
    tello.connect()
    tello.streamon()
    frames = tello.get_frame_read()
    cam = CameraInfo(tello, args)
    use_flow = cam.mode == "mirror" and not args.no_flow
    flow = FloorFlow(cam.K[0, 0], center=point_below_drone(cam.K, cam.tilt)) if use_flow else None
    pilot = Pilot(tello, frame_read=frames, position=(lambda: flow.state() if flow.valid(time.monotonic()) else None)
                  if use_flow else None)
    win = Window(args, "hover_hold")
    print(f"battery {tello.get_battery()}%, taking off...")
    pilot.takeoff(altitude_cm=args.altitude)
    start = true_pose(tello)
    t0, yaw_prev, worst = time.monotonic(), None, 0.0
    try:
        while pilot.flying and time.monotonic() - t0 < args.seconds:
            img = cv2.cvtColor(frames.frame, cv2.COLOR_RGB2BGR)
            state = tello.get_current_state()
            yaw = state.get("yaw", 0)
            if flow is not None:
                view = unmirror(img)
                turning = yaw_prev is not None and abs(((yaw - yaw_prev) + 180) % 360 - 180) > 1
                flow.update(cv2.cvtColor(view, cv2.COLOR_BGR2GRAY), time.monotonic(), altitude_cm(state), turning)
                img = view
            yaw_prev = yaw
            pose = true_pose(tello)
            if pose and start:
                worst = max(worst, math.hypot(pose[0] - start[0], pose[1] - start[1]))
            lines = [f"t {time.monotonic() - t0:4.1f} s  alt {altitude_cm(state)} cm  bat {state.get('bat')}%",
                     f"rc sent {pilot.sent}"]
            if flow is not None:
                (pr, pf), _ = flow.state()
                lines.append(f"floor: {pr:+.0f} right, {pf:+.0f} forward (q {flow.quality:.2f})")
            if pose and start:
                lines.append(f"simulator: {math.hypot(pose[0] - start[0], pose[1] - start[1]):.0f} cm from start")
            action = win.show(img, lines)
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
        tello.end()
    if pilot.reason:
        print(f"landed by the safety monitor: {pilot.reason}")
    if start:
        print(f"result: farthest {worst:.0f} cm from the starting point in {args.seconds:.0f} s")


if __name__ == "__main__":
    main()
