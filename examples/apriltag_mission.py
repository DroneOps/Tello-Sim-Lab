"""A short version of the AprilTag-Detection mission: tags on the floor, seen with a mirror.

    python examples/apriltag_mission.py                 # simulator: circuit 1 -> 2 -> 7
    python examples/apriltag_mission.py --model lab_tello7
    python examples/apriltag_mission.py --real

It flies forward looking down; when a tag (25h9, ids 1-10) appears it gets over it
(TargetCentering), runs its action and continues straight ahead:
1 turn right 90°, 2 turn left 90°, 3 turn 180°, 4 flip, 5 up 30 cm, 6 down 30 cm, 7 land,
8 turn 180° and flip, 9 wait 5 s, 10 turn 360°.

The full mission (tag tracking for repeated ids, leg anchoring, floor-flow position hold...) is
in DroneOps/AprilTag-Detection, scripts/principales/mision_espejo.py.
"""
from __future__ import annotations

import math
import time

import cv2

from common import CameraInfo, Window, make_tello, parser, true_pose
from tellolab import Pilot, TargetCentering, altitude_cm
from tellolab.vision import detect_markers, marker_position, unmirror

ACTIONS = {1: [("turn", 90)], 2: [("turn", -90)], 3: [("turn", 180)], 4: [("flip",)], 5: [("climb", 30)],
           6: [("climb", -30)], 7: [("land",)], 8: [("turn", 180), ("flip",)], 9: [("wait", 5)],
           10: [("turn", 90)] * 4}


def main() -> None:
    p = parser(__doc__.splitlines()[0], world="apriltag_course", camera_mode="mirror")
    p.add_argument("--size", type=float, default=8.0, help="tag size in cm (black square)")
    p.add_argument("--speed", type=int, default=15)
    args = p.parse_args()

    tello = make_tello(args)
    tello.connect()
    tello.streamon()
    frames = tello.get_frame_read()
    cam = CameraInfo(tello, args)
    pilot = Pilot(tello, frame_read=frames)
    center = TargetCentering()
    win = Window(args, "apriltag_mission")
    pilot.takeoff(altitude_cm=100)
    pilot._wait(2)
    # phases: search (forward, looking down) -> center (get over the tag) -> execute (its actions,
    # staying over it) -> search. The tag just done is ignored while it is still in view.
    t0, executed, phase, target, plan = time.monotonic(), [], "search", None, []
    last_id, t_last_seen, t_phase, t_step = None, 0.0, time.monotonic(), None
    try:
        while pilot.flying and time.monotonic() - t0 < args.timeout:
            view = unmirror(cv2.cvtColor(frames.frame, cv2.COLOR_RGB2BGR))
            state = tello.get_current_state()
            now, yaw = time.monotonic(), state.get("yaw", 0)
            tags = {}
            for i, c in detect_markers(cv2.cvtColor(view, cv2.COLOR_BGR2GRAY), ids=set(range(1, 11))):
                f, r, _ = marker_position(c, args.size, cam.K, cam.dist, "mirror", view_forward_deg=cam.tilt)
                tags[i] = (f, r)
            if last_id in tags:
                t_last_seen = now
            lr = fb = 0
            if target in tags:
                lr, fb, settled = center.update(*tags[target], now, yaw)
            else:
                settled = False
            if phase == "search":
                new = {i: v for i, v in tags.items() if not (i == last_id and now - t_last_seen < 1.5)}
                if new:
                    # the one most in line with where it is going (not just the nearest: after a
                    # drift at takeoff another tag to a side can be nearer)
                    target = min(new, key=lambda i: abs(new[i][1]) + 0.3 * abs(new[i][0]))
                    phase, t_phase = "center", now
                    center.reset()
                    status = f"tag {target} found"
                else:
                    fb = args.speed
                    status = "flying forward, looking for tags"
            elif phase == "center":
                status = f"getting over tag {target}"
                if settled or now - t_phase > 8:
                    executed.append(target)
                    plan, phase, t_step = list(ACTIONS[target]), "execute", None
                    print(f"tag {target} at {now - t0:.0f} s: {plan}")
            else:  # execute
                step = plan[0] if plan else None
                status = f"tag {target}: {step[0] if step else 'done'}"
                if step is None:
                    last_id, t_last_seen, target, phase = target, now, None, "search"
                elif step[0] == "turn":
                    if t_step is None:
                        pilot.turn_by(step[1])
                        t_step = now
                    elif abs(pilot.heading.error(yaw)) < 4 or now - t_step > 10:
                        plan.pop(0)
                        t_step = None
                elif step[0] == "wait":
                    t_step = t_step or now
                    if now - t_step > step[1]:
                        plan.pop(0)
                        t_step = None
                elif step[0] == "climb":
                    pilot.set_altitude(altitude_cm(state) + step[1])
                    plan.pop(0)
                elif step[0] == "flip":
                    print(pilot.flip("b"))
                    plan.pop(0)
                elif step[0] == "land":
                    print(f"tag {target}: landing")
                    break
            pilot.set_rc(lr, fb, 0, 0)
            action = win.show(view, [status, f"done {executed}  alt {altitude_cm(state)} cm"])
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
        print(f"result: executed {executed}")
        if pose:
            tags = [o for o in tello.sim.world.objects if getattr(o, "id", None) == (executed[-1] if executed else -1)]
            if tags:
                t = min(tags, key=lambda o: math.hypot(o.position[0] - pose[0], o.position[1] - pose[1]))
                print(f"result: landed {math.hypot(t.position[0] - pose[0], t.position[1] - pose[1]):.0f} cm from tag {t.id}")
        tello.end()


if __name__ == "__main__":
    main()
