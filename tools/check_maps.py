"""Check maps: textures, collisions and dimensions.

    python tools/check_maps.py                      # every built-in map
    python tools/check_maps.py my_track.yaml        # yours
    python tools/check_maps.py --images views/      # also save what the camera sees of each object

What it checks:
- Textures: the floor texture covers the objects and the start, it is not huge, lines are wide
  enough in pixels, round pads do not show a square around them.
- Dimensions: every marker is detected by OpenCV and measured with its own size (solvePnP)
  within 3 %; the opening of every rectangular gate measures its width_cm in the image; markers
  fit on their pad; gates do not go into the floor.
- Collisions: flying through each gate's opening is free and its frame (and legs) are not; a box,
  a wall, a panel and a raised pad stop the drone and leave room around them; flat things
  (markers, lines, flat pads) never do; the takeoff column at the start is free.
- Moving objects: their whole run stays on the floor texture, never crosses the takeoff column,
  and a pad keeps its speed; fog: a sensible visibility.

Exit code 1 if anything FAILS (warnings do not count).
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tellolab.vision import detect_markers, marker_position  # noqa: E402
from tellosim import Camera, Renderer, builtin_maps, load_map  # noqa: E402
from tellosim.drone import RADIUS_CM  # noqa: E402
from tellosim.objects import (Box, ColorPanel, FloorLine, Gate, LandingPad, Marker, Wall, bgr,  # noqa: E402
                              compass, marker_image)

R = RADIUS_CM


class Report:
    def __init__(self, name: str) -> None:
        self.name, self.lines, self.fails, self.warns = name, [], 0, 0

    def ok(self, msg: str) -> None:
        self.lines.append(f"  ok    {msg}")

    def warn(self, msg: str) -> None:
        self.warns += 1
        self.lines.append(f"  WARN  {msg}")

    def fail(self, msg: str) -> None:
        self.fails += 1
        self.lines.append(f"  FAIL  {msg}")

    def check(self, cond: bool, msg: str) -> None:
        (self.ok if cond else self.fail)(msg)


def label(o) -> str:
    return f"{type(o).__name__} {o.name}".strip() if o.name else f"{type(o).__name__}"


def footprint(o) -> list[tuple[float, float]]:
    """Points on the floor that the object covers (to check the texture covers them)."""
    if isinstance(o, FloorLine):
        return [tuple(p) for p in o.points]
    if isinstance(o, Wall):
        return [tuple(o.start), tuple(o.end)]
    if isinstance(o, LandingPad):
        x, y = o.position[:2]
        a = o.size_cm / 2
        return [(x - a, y - a), (x + a, y + a)]
    if isinstance(o, Box):
        x, y = o.position[:2]
        a = max(o.size[0], o.size[1]) / 2
        return [(x - a, y - a), (x + a, y + a)]
    if hasattr(o, "position"):
        return [tuple(o.position[:2])]
    return []


def run_footprint(o) -> list[tuple[float, float]]:
    """footprint() at both ends of the object's motion (or where it is, if it does not move)."""
    pts = footprint(o)
    if not o.motion:
        return pts
    m = o.motion
    ends = [m.offset(m.start_s + k * m.distance_cm / max(abs(m.speed_cm_s), 1e-6) / 2) for k in (1, 3)] \
        if m.back_and_forth else [np.zeros(2), m.offset(m.start_s + m.distance_cm / max(abs(m.speed_cm_s), 1e-6))]
    return [(x + e[0], y + e[1]) for e in ends for (x, y) in pts]


def renderer(world, mode: str, **cam) -> Renderer:
    c = Camera.load("lab_tello", **{**world.camera, "noise_std": 0.0, **cam, "mode": mode})
    return Renderer(world, c, seed=0)


def check_texture(world, rep: Report) -> None:
    f = world.floor
    mb = f.size_cm[0] * f.size_cm[1] * f.px_per_cm ** 2 * 3 / 1e6
    (rep.warn if mb > 60 else rep.ok)(f"floor texture {f.size_cm[0]:.0f}x{f.size_cm[1]:.0f} cm at "
                                      f"{f.px_per_cm:g} px/cm = {mb:.0f} MB")
    x0, x1 = f.center[0] - f.size_cm[0] / 2, f.center[0] + f.size_cm[0] / 2
    y0, y1 = f.center[1] - f.size_cm[1] / 2, f.center[1] + f.size_cm[1] / 2
    outside = [label(o) for o in world.objects for (x, y) in run_footprint(o) if not (x0 <= x <= x1 and y0 <= y <= y1)]
    sx, sy = world.start[:2]
    if not (x0 + 100 <= sx <= x1 - 100 and y0 + 100 <= sy <= y1 - 100):
        outside.append("the start (needs 1 m of texture around it)")
    if outside:
        rep.fail(f"outside the floor texture (plain floor there, nothing to measure motion with): "
                 f"{', '.join(sorted(set(outside)))}")
    else:
        rep.ok("the floor texture covers every object and the start")
    for o in world.objects:
        if isinstance(o, FloorLine):
            px = o.width_cm * f.px_per_cm
            (rep.ok if px >= 3 else rep.warn)(f"{label(o)}: {o.width_cm:g} cm = {px:.0f} px in the texture")
        if isinstance(o, LandingPad) and o.shape == "circle":
            img = o._image()
            corner = tuple(int(v) for v in img[2, 2, :3])
            transparent = img.shape[2] == 4 and img[2, 2, 3] == 0
            (rep.ok if transparent else rep.fail)(
                f"{label(o)}: round pad, the corners outside the circle are "
                f"{'transparent' if transparent else f'painted {corner}, a square on the floor'}")


def check_marker(world, o: Marker, rep: Report, images: Path | None, where: str = "") -> None:
    c, right, up = o.frame()
    if o.mount == "floor":
        mode, pose = "mirror", (c[0], c[1] + 10, c[2] + 60, 0.0)
    else:
        n = compass(o.facing_deg)
        p = c + 120 * n
        yaw = math.degrees(math.atan2(-n[0], -n[1]))
        mode, pose = "forward", (p[0], p[1], c[2] + 120 * math.tan(math.radians(11)), yaw)
    rnd = renderer(world, mode)
    img = rnd.render(*pose)
    if mode == "mirror":
        img = cv2.flip(img, 0 if rnd.cam.mirror_axis == "v" else 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found = [cr for i, cr in detect_markers(gray, o.dictionary) if i == o.id]
    if images:
        cv2.imwrite(str(images / f"{rep.name}_{o.dictionary[5:]}_{o.id}{where}.jpg"), img)
    if not found:
        rep.fail(f"{label(o)} {o.id}{where}: not detected from {'60 cm above' if mode == 'mirror' else '1.2 m in front'}")
        return
    K, d = rnd.cam.detection_K
    est = marker_position(found[0], o.size_cm, K, d, mode, rnd.cam.pitch_deg, rnd.cam.mirror_tilt_deg)
    ps = math.radians(pose[3])
    fwd, rgt = np.array([math.sin(ps), math.cos(ps), 0.0]), np.array([math.cos(ps), -math.sin(ps), 0.0])
    rel = c - np.array(pose[:3])
    true = np.array([rel @ fwd, rel @ rgt, rel[2]])
    err = np.linalg.norm(est - true) / np.linalg.norm(true)
    rep.check(err < 0.03, f"{label(o)} {o.id}{where}: detected; measured with size_cm {o.size_cm:g} it is "
                          f"{np.linalg.norm(est):.1f} cm away, really {np.linalg.norm(true):.1f} ({100 * err:.1f} %)")


def check_gate(world, g: Gate, rep: Report, images: Path | None) -> None:
    c, n, right, up = g._axes()
    w, h, b = g.width_cm / 2, g.height_cm / 2, g.bar_cm
    bottom = c[2] - (w if g.shape == "circle" else h) - b
    rep.check(bottom >= 0, f"{label(g)}: bottom of the frame at {bottom:.0f} cm (not in the floor)")
    # collisions
    rep.check(g.collides(c, R) is None, f"{label(g)}: the center of the opening is free")
    if g.shape == "circle":
        free = [c + (w - R - 2) * right, c - (w - R - 2) * up]
        hits = [c + (w + b / 2) * right, c + (w + b / 2) * up]
    else:
        free = [c + (w - R - 2) * right, c + (h - R - 2) * up, c - (h - R - 2) * up]
        hits = [c + (w + b / 2) * right, c - (w + b / 2) * right, c + (h + b / 2) * up, c - (h + b / 2) * up]
    free += [c + (w + b + R + 3) * right, c + (R + 3) * n + (w + b / 2) * right]
    rep.check(all(g.collides(p, R) is None for p in free),
              f"{label(g)}: free just inside the opening, beside the frame and in front of it")
    rep.check(all(g.collides(p, R) for p in hits), f"{label(g)}: every bar of the frame stops the drone")
    if g.shape == "rect" and g.legs and bottom > 15:
        leg = c + (w + b / 2) * right - (h + b + bottom / 2) * up
        between = c - (h + b + bottom / 2) * up
        rep.check(g.collides(leg, R) is not None and g.collides(between, R) is None,
                  f"{label(g)}: the legs stop the drone, the space between them under the gate does not")
    # dimensions: the opening in the image, 3 m in front at its height
    if g.shape == "rect":
        dist = 300.0
        p = c - dist * n
        yaw = math.degrees(math.atan2(n[0], n[1]))
        rnd = renderer(world, "forward", pitch_deg=0.0, blur_sigma=0.0)
        img = rnd.render(p[0], p[1], c[2], yaw)
        if images:
            cv2.imwrite(str(images / f"{rep.name}_{g.name or 'gate'}.jpg"), img)
        col = np.array(bgr(g.color), int)
        mask = (np.abs(img.astype(int) - col).sum(axis=2) < 40).astype(np.uint8)
        cs, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        holes = [cv2.boundingRect(cs[i]) for i in range(len(cs)) if hier[0][i][3] != -1]
        if not holes:
            rep.fail(f"{label(g)}: its opening is not a closed shape in the image from 3 m")
            return
        bw = max(holes, key=lambda r: r[2] * r[3])[2]
        K = rnd.cam.render_K[0]
        expected = K[0, 0] * g.width_cm / dist
        err = abs(bw - expected) / expected
        rep.check(err < 0.05, f"{label(g)}: the opening measures {bw} px from 3 m, expected "
                              f"{expected:.0f} for {g.width_cm:g} cm ({100 * err:.1f} %)")


def check_pad(world, p: LandingPad, rep: Report, images: Path | None) -> None:
    if p.marker:
        m = p.marker
        img_ratio = marker_image(m.get("dictionary", "DICT_APRILTAG_25h9"), int(m.get("id", 0)))[1]
        printed = float(m.get("size_cm", p.size_cm * 0.5)) * img_ratio
        room = p.size_cm - 2 * p.border_cm
        rep.check(printed <= room, f"{label(p)}: its marker ({printed:.1f} cm with the white margin) fits inside "
                                   f"the border ({room:g} cm)")
        mk = Marker(id=int(m.get("id", 0)), dictionary=m.get("dictionary", "DICT_APRILTAG_25h9"),
                    size_cm=float(m.get("size_cm", p.size_cm * 0.5)),
                    position=(p.position[0], p.position[1], p.height_cm + 0.1), rotation_deg=p.rotation_deg)
        check_marker(world, mk, rep, images, where=" on the pad")
    x, y = p.position[:2]
    if p.height_cm > 0:
        rep.check(abs(world.surface_height(x, y) - p.height_cm) < 0.1,
                  f"{label(p)}: the ToF reads its top ({p.height_cm:g} cm) over it")
        side = np.array([x + p.size_cm / 2 + R / 2, y, p.height_cm / 2])
        rep.check(p.collides(side, R) is not None, f"{label(p)}: its side stops the drone")
        rep.check(p.collides(np.array([x, y, p.height_cm + 5]), R) is None, f"{label(p)}: the drone can land on top")
        if p.shape == "circle":
            a = p.size_cm / 2 * 0.8
            rep.check(p.collides(np.array([x + a, y + a, p.height_cm / 2]), R) is None and p.top_height(x + a, y + a) == 0,
                      f"{label(p)}: round: no top or side outside the circle (the corners of its square)")
    else:
        rep.check(p.collides(np.array([x, y, 2.0]), R) is None and world.surface_height(x, y) == 0,
                  f"{label(p)}: flat: nothing to hit, the ToF reads the floor")


def check_solid(world, o, rep: Report) -> None:
    if isinstance(o, Box):
        x, y = o.position[:2]
        sx, sy, sz = o.size
        ax = compass(o.rotation_deg + 90)
        rep.check(abs(o.top_height(x, y) - sz) < 0.1 and o.collides(np.array([x, y, sz / 2]), R) is not None,
                  f"{label(o)}: stops the drone, the ToF reads its top ({sz:g} cm)")
        beside = np.array([x, y, sz / 2]) + (sx / 2 + R + 3) * np.append(ax[:2], 0)
        above = np.array([x, y, sz + 5])
        rep.check(o.collides(beside, R) is None and o.collides(above, R) is None, f"{label(o)}: free beside and above")
    elif isinstance(o, Wall):
        a, b = np.array(o.start, float), np.array(o.end, float)
        m = (a + b) / 2
        nrm = np.array([-(b - a)[1], (b - a)[0]]) / np.linalg.norm(b - a)
        rep.check(o.collides(np.array([*m, o.height_cm / 2]), R) is not None
                  and o.collides(np.array([*(m + (R + 3) * nrm), o.height_cm / 2]), R) is None
                  and o.collides(np.array([*m, o.height_cm + 5]), R) is None,
                  f"{label(o)}: stops the drone on it, free in front and above it")
    elif isinstance(o, ColorPanel):
        c, n = np.array(o.position, float), compass(o.facing_deg)
        rep.check(o.collides(c, R) is not None and o.collides(c + (R + 3) * n, R) is None,
                  f"{label(o)}: stops the drone on it, free in front of it")


def check_wall_markers(world, rep: Report) -> None:
    walls = [o for o in world.objects if isinstance(o, Wall)]
    for o in world.objects:
        if isinstance(o, Marker) and o.mount == "wall":
            c, right, up = o.frame()
            ok = False
            for w in walls:
                a, b = np.array(w.start, float), np.array(w.end, float)
                q = c[:2]
                t = (q - a) @ (b - a) / ((b - a) @ (b - a))
                dist = np.linalg.norm(q - (a + np.clip(t, 0, 1) * (b - a)))
                half = o.size_cm * 0.65
                L = np.linalg.norm(b - a)
                if dist < 3 and half / L <= t <= 1 - half / L and half <= c[2] <= w.height_cm - half:
                    ok = True
            rep.check(ok, f"{label(o)} {o.id}: on a wall, inside its edges")


def check_start(world, rep: Report) -> None:
    sx, sy = world.start[:2]
    times = np.arange(0, 120, 0.5) if world.moving else [0.0]
    hits = {world.collision((sx, sy, z), R, t) for t in times for z in range(0, 130, 5)} - {None}
    rep.check(not hits, "the takeoff column at the start is free" + (" (during 2 min)" if world.moving else "")
              + (f": {hits}" if hits else ""))
    rep.check(world.surface_height(sx, sy) == 0, "the start is on the floor")


def check_motion(world, rep: Report) -> None:
    for o in world.objects:
        if not o.motion:
            continue
        m = o.motion
        if isinstance(o, FloorLine):
            rep.fail(f"{label(o)}: a line cannot move")
            continue
        ts = np.arange(m.start_s + 0.05, m.start_s + 60, 0.1)
        pos = np.array([m.offset(t) for t in ts])
        speed = np.linalg.norm(np.diff(pos, axis=0), axis=1) / 0.1
        steady = np.abs(speed - abs(m.speed_cm_s)) < 0.05 * abs(m.speed_cm_s) + 0.5
        reach = np.ptp(pos @ compass(m.direction_deg)[:2])
        rep.check(steady.mean() > 0.9 and abs(reach - m.distance_cm) < 1.0,
                  f"{label(o)}: moves towards {m.direction_deg:g}° at {np.median(speed):.1f} cm/s "
                  f"(asked {abs(m.speed_cm_s):g}), a run of {reach:.0f} cm (asked {m.distance_cm:g})"
                  + (", back and forth" if m.back_and_forth else ", once"))


def check_fog(world, rep: Report) -> None:
    f = world.fog
    if f is None:
        return
    rep.check(30 <= f.visibility_cm <= 5000 and 0 <= f.patches <= 2,
              f"fog: visibility {f.visibility_cm:g} cm (things there are 95 % hidden), patches {f.patches:g}")


def check_overlaps(world, rep: Report) -> None:
    flats = [o for o in world.objects if (isinstance(o, Marker) and o.mount == "floor")
             or (isinstance(o, LandingPad) and o.height_cm == 0)]
    for i, a in enumerate(flats):
        for b in flats[i + 1:]:
            ra = (a.size_cm * 0.7 if isinstance(a, Marker) else a.size_cm / 2)
            rb = (b.size_cm * 0.7 if isinstance(b, Marker) else b.size_cm / 2)
            if math.dist(a.position[:2], b.position[:2]) < ra + rb:
                rep.warn(f"{label(a)} and {label(b)} overlap on the floor")


def check_map(spec, images: Path | None) -> Report:
    world = load_map(spec)
    rep = Report(world.name or Path(str(spec)).stem)
    check_texture(world, rep)
    check_start(world, rep)
    check_motion(world, rep)
    check_fog(world, rep)
    check_overlaps(world, rep)
    check_wall_markers(world, rep)
    for o in world.objects:
        if isinstance(o, Marker):
            check_marker(world, o, rep, images)
            rep.check(o.collides(np.array([*o.frame()[0][:2], 1.0]), R) is None or o.mount == "wall",
                      f"{label(o)} {o.id}: nothing to hit")
        elif isinstance(o, Gate):
            check_gate(world, o, rep, images)
        elif isinstance(o, LandingPad):
            check_pad(world, o, rep, images)
        elif isinstance(o, FloorLine):
            rep.check(world.collision((*o.points[0], 2.0), R) is None, f"{label(o)}: nothing to hit")
        else:
            check_solid(world, o, rep)
    return rep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("maps", nargs="*", help="map names or files (default: every built-in map)")
    ap.add_argument("--images", type=Path, default=None, help="save the views used for the checks here")
    a = ap.parse_args()
    if a.images:
        a.images.mkdir(parents=True, exist_ok=True)
    fails = 0
    for spec in a.maps or builtin_maps():
        rep = check_map(spec, a.images)
        fails += rep.fails
        print(f"{rep.name}: {'FAIL' if rep.fails else 'ok'}" + (f" ({rep.warns} warnings)" if rep.warns else ""))
        print("\n".join(rep.lines))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
