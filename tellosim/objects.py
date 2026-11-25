"""Things you can put in the simulated world (the "map").

Coordinates (all in cm): x to the east (right of the takeoff heading), y to the north (the
takeoff heading), z up. Angles are compass degrees: 0 = +y (north), 90 = +x (east).
Colors are RGB (``[255, 0, 0]`` is red) or a name from :data:`COLORS`.

Every object can be written in a map YAML as a dict with ``type:`` plus its fields (see
``docs/maps.md``), or created in Python and added with ``world.add(obj)``.

To add a new kind of object, subclass :class:`SimObject` and implement ``primitives()`` (what to
draw), and optionally ``paint_floor()``, ``top_height()`` and ``collides()``; then register it in
:data:`OBJECT_TYPES`. See ``docs/extending.md``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

# ---------------------------------------------------------------------------------- colors
COLORS = {
    "black": (20, 20, 20), "white": (245, 245, 245), "gray": (128, 128, 128),
    "red": (220, 30, 30), "green": (30, 170, 60), "blue": (30, 70, 210), "yellow": (240, 210, 30),
    "orange": (245, 130, 20), "purple": (140, 50, 170), "cyan": (40, 190, 210), "pink": (240, 120, 170),
    "lab_tape": (70, 120, 170), "dark_floor": (85, 85, 85),
}


def bgr(color) -> tuple[int, int, int]:
    """RGB list/tuple or color name -> OpenCV BGR tuple."""
    if isinstance(color, str):
        if color not in COLORS:
            raise ValueError(f"unknown color {color!r} (known: {', '.join(sorted(COLORS))}, or [r, g, b])")
        color = COLORS[color]
    r, g, b = (int(c) for c in color)
    return (b, g, r)


def compass(deg: float) -> np.ndarray:
    """Unit horizontal vector pointing to a compass angle (0 = +y, 90 = +x)."""
    a = math.radians(deg)
    return np.array([math.sin(a), math.cos(a), 0.0])


# ---------------------------------------------------------------------------------- primitives
@dataclass
class TexturedQuad:
    """A flat picture in 3D. corners: (4, 3) world points matching the image corners
    top-left, top-right, bottom-right, bottom-left. normal: the side it can be seen from."""
    corners: np.ndarray
    image: np.ndarray  # BGR uint8
    normal: np.ndarray


@dataclass
class SolidPoly:
    """A flat convex polygon of one color. normal=None: visible from both sides."""
    vertices: np.ndarray  # (N, 3)
    color: tuple  # BGR
    normal: np.ndarray | None = None


# ---------------------------------------------------------------------------------- base class
@dataclass
class SimObject:
    name: str = ""

    def primitives(self) -> list:
        """What to draw: TexturedQuad and SolidPoly instances in world coordinates."""
        return []

    def paint_floor(self, canvas: np.ndarray, to_px) -> None:
        """Paint onto the floor texture (lines, pads). to_px(x, y) -> (u, v) float pixel."""

    def top_height(self, x: float, y: float) -> float:
        """Height (cm) of this object's top surface under (x, y), for the ToF sensor; 0 if none."""
        return 0.0

    def collides(self, p: np.ndarray, radius: float) -> str | None:
        """A message if a drone of this radius at point p touches the object."""
        return None


# ---------------------------------------------------------------------------------- markers
WEAR_PRESETS = {
    # tags of the lab seen through the mirror (videos, Nov 2026): black reads ~150, paper ~252
    "lab": {"black": 150},
    "very_faded": {"black": 185, "white": 240},
    "stains": {"black": 150, "stains": 0.2},
    "missing": {"black": 150, "missing": 0.08},
    "all": {"black": 170, "white": 245, "stains": 0.15, "missing": 0.06},
}


def wear(img: np.ndarray, quiet_px: int, black: float = 0, white: float = 255, stains: float = 0.0,
         missing: float = 0.0, seed: int = 0) -> np.ndarray:
    """A worn marker (gray image with a white quiet zone of quiet_px).
    black/white: gray levels after fading (uneven, ±20 in big patches); stains: fraction of the
    black covered by white dots and scratches; missing: fraction of the marker torn off in 1–3
    pieces (they can fall on the black border, the worst case for detection)."""
    rng = np.random.default_rng(seed)
    n = img.shape[0]
    dark = img < 128
    ink = np.zeros(img.shape, np.float32)
    ink[dark] = 1.0
    inner = np.zeros(img.shape, bool)
    inner[quiet_px:n - quiet_px, quiet_px:n - quiet_px] = True
    lo, hi = quiet_px, n - quiet_px
    if stains > 0:
        target, total = stains * dark.sum(), 0
        while total < target:
            hole = np.zeros(img.shape, np.uint8)
            if rng.random() < 0.15:
                p0, p1 = rng.integers(lo, hi, 2), rng.integers(lo, hi, 2)
                cv2.line(hole, tuple(int(v) for v in p0), tuple(int(v) for v in p1), 1, int(rng.integers(1, 3)))
            else:
                c = tuple(int(v) for v in rng.integers(lo, hi, 2))
                cv2.circle(hole, c, int(rng.integers(2, 8)), 1, -1)
            new = (hole > 0) & (ink > 0)
            total += new.sum()
            ink[new] = 0.0
    if missing > 0:
        k = int(rng.integers(1, 4))
        area = missing * (n - 2 * quiet_px) ** 2 / k
        for _ in range(k):
            ax = math.sqrt(area / math.pi) * rng.uniform(0.6, 1.6)
            ay = area / (math.pi * ax)
            c = tuple(int(v) for v in rng.integers(lo, hi, 2))
            hole = np.zeros(img.shape, np.uint8)
            cv2.ellipse(hole, c, (max(1, int(ax)), max(1, int(ay))), float(rng.uniform(0, 180)), 0, 360, 1, -1)
            ink[(hole > 0) & inner] = 0.0
    patch = cv2.resize(rng.uniform(-1, 1, (4, 4)).astype(np.float32), img.shape[::-1], interpolation=cv2.INTER_CUBIC)
    gray_black = np.clip(black + 20 * patch * (black > 0), 0, 255)
    out = white * (1 - ink) + gray_black * ink
    out = cv2.GaussianBlur(out, (3, 3), 0.7)
    return np.clip(out, 0, 255).astype(np.uint8)


_MARKER_CACHE: dict = {}
MODULE_PX = 20  # pixels per marker module in the marker images


def marker_image(dictionary: str, marker_id: int, module_px: int = MODULE_PX) -> tuple[np.ndarray, float]:
    """Gray marker image with a one-module white quiet zone, and the ratio
    outer side / black-square side."""
    key = (dictionary, marker_id, module_px)
    if key not in _MARKER_CACHE:
        if not hasattr(cv2.aruco, dictionary):
            raise ValueError(f"unknown OpenCV dictionary {dictionary!r} (e.g. DICT_APRILTAG_25h9, DICT_4X4_50)")
        d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
        modules = d.markerSize + 2  # data bits + black border
        m = cv2.aruco.generateImageMarker(d, int(marker_id), modules * module_px)
        img = cv2.copyMakeBorder(m, module_px, module_px, module_px, module_px, cv2.BORDER_CONSTANT, value=255)
        _MARKER_CACHE[key] = (img, (modules + 2) / modules, module_px)
    img, ratio, _ = _MARKER_CACHE[key]
    return img, ratio


@dataclass
class Marker(SimObject):
    """A printed fiducial marker (AprilTag or ArUco) on the floor or on a wall.

    size_cm is the side of the black square (what detectors call the marker size).
    Floor: ``position: [x, y]`` and ``rotation_deg`` (where its top edge points).
    Wall: ``mount: wall``, ``position: [x, y, z]`` (center) and ``facing_deg`` (where the printed
    face points, e.g. 180 = it faces a drone coming from the south).
    wear: a preset from WEAR_PRESETS or a dict for :func:`wear`."""
    id: int = 0
    position: tuple = (0.0, 0.0)
    size_cm: float = 8.0
    dictionary: str = "DICT_APRILTAG_25h9"
    mount: str = "floor"  # "floor" or "wall" (not "on": YAML reads on as true)
    rotation_deg: float = 0.0
    facing_deg: float = 180.0
    wear: object = None
    _img: np.ndarray | None = field(default=None, repr=False)

    def _image(self) -> tuple[np.ndarray, float]:
        gray, ratio = marker_image(self.dictionary, self.id, MODULE_PX)
        if self._img is None:
            if self.wear:
                opts = WEAR_PRESETS[self.wear] if isinstance(self.wear, str) else dict(self.wear)
                gray = wear(gray, MODULE_PX, **{"seed": self.id, **opts})
            self._img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return self._img, ratio

    def frame(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(center, right, up) of the printed face in world coordinates; normal = cross(right, up)."""
        if self.mount == "floor":
            c = np.array([*self.position[:2], self.position[2] if len(self.position) > 2 else 0.0], float)
            up = compass(self.rotation_deg)
            n = np.array([0.0, 0.0, 1.0])
        else:
            c = np.array(self.position, float)
            n = compass(self.facing_deg)
            up = np.array([0.0, 0.0, 1.0])
        right = np.cross(up, n)
        return c, right, up

    def primitives(self) -> list:
        img, ratio = self._image()
        c, right, up = self.frame()
        a = self.size_cm * ratio / 2
        corners = np.array([c - a * right + a * up, c + a * right + a * up,
                            c + a * right - a * up, c - a * right - a * up])
        if self.mount == "floor":
            corners[:, 2] += 0.05  # just above the floor texture
        return [TexturedQuad(corners, img, np.cross(right, up))]


@dataclass
class AprilTag(Marker):
    dictionary: str = "DICT_APRILTAG_25h9"


@dataclass
class ArucoMarker(Marker):
    dictionary: str = "DICT_4X4_50"


# ---------------------------------------------------------------------------------- floor paint
@dataclass
class FloorLine(SimObject):
    """A painted or taped line on the floor (for line followers). points: [[x, y], ...]."""
    points: list = field(default_factory=list)
    width_cm: float = 5.0
    color: object = "black"
    closed: bool = False

    def paint_floor(self, canvas: np.ndarray, to_px) -> None:
        if len(self.points) < 2:
            return
        px_per_cm = abs(to_px(1, 0)[0] - to_px(0, 0)[0])
        pts = np.array([to_px(x, y) for x, y in self.points], np.float32)
        shift = 4  # sub-pixel accuracy for cv2 drawing
        cv2.polylines(canvas, [np.round(pts * (1 << shift)).astype(np.int32)], self.closed, bgr(self.color),
                      max(1, int(round(self.width_cm * px_per_cm))), cv2.LINE_AA, shift)


@dataclass
class LandingPad(SimObject):
    """A landing pad: a square or circle with a border, an optional letter (e.g. "H") and an
    optional marker in the middle (``marker: {dictionary, id, size_cm}``). height_cm > 0 makes
    it a raised platform (the ToF sees it, and the drone can crash into its side)."""
    position: tuple = (0.0, 0.0)
    size_cm: float = 60.0
    shape: str = "square"
    color: object = "white"
    border_color: object = "red"
    border_cm: float = 4.0
    letter: str = "H"
    letter_color: object = "black"
    marker: dict | None = None
    height_cm: float = 0.0
    rotation_deg: float = 0.0
    side_color: object = "gray"

    def _image(self) -> np.ndarray:
        s = 4  # px per cm
        n = int(self.size_cm * s)
        img = np.zeros((n, n, 3), np.uint8)
        if self.shape == "circle":
            img[:] = bgr("dark_floor")  # outside the circle: like the floor (drawn on top of it)
            cv2.circle(img, (n // 2, n // 2), n // 2, bgr(self.border_color), -1, cv2.LINE_AA)
            cv2.circle(img, (n // 2, n // 2), n // 2 - int(self.border_cm * s), bgr(self.color), -1, cv2.LINE_AA)
        else:
            img[:] = bgr(self.border_color)
            b = int(self.border_cm * s)
            img[b:n - b, b:n - b] = bgr(self.color)
        if self.letter and not self.marker:
            scale = n / 160
            (tw, th), _ = cv2.getTextSize(self.letter, cv2.FONT_HERSHEY_SIMPLEX, scale, max(2, n // 20))
            cv2.putText(img, self.letter, ((n - tw) // 2, (n + th) // 2), cv2.FONT_HERSHEY_SIMPLEX, scale,
                        bgr(self.letter_color), max(2, n // 20), cv2.LINE_AA)
        return img

    def _axes(self):
        c = np.array([*self.position[:2], self.height_cm], float)
        up, n = compass(self.rotation_deg), np.array([0.0, 0.0, 1.0])
        return c, np.cross(up, n), up

    def primitives(self) -> list:
        c, right, up = self._axes()
        a = self.size_cm / 2
        corners = np.array([c - a * right + a * up, c + a * right + a * up,
                            c + a * right - a * up, c - a * right - a * up])
        corners[:, 2] += 0.03
        prims: list = [TexturedQuad(corners, self._image(), np.array([0.0, 0.0, 1.0]))]
        if self.height_cm > 0:
            bottom = corners.copy()
            bottom[:, 2] = 0.0
            for i in range(4):
                j = (i + 1) % 4
                face = np.array([corners[i], corners[j], bottom[j], bottom[i]])
                normal = np.cross(face[1] - face[0], face[3] - face[0])
                normal = normal / (np.linalg.norm(normal) + 1e-9)
                if normal @ (face.mean(axis=0) - c) < 0:
                    normal = -normal
                prims.append(SolidPoly(face, bgr(self.side_color), normal))
        if self.marker:
            m = Marker(id=int(self.marker.get("id", 0)), dictionary=self.marker.get("dictionary", "DICT_APRILTAG_25h9"),
                       size_cm=float(self.marker.get("size_cm", self.size_cm * 0.5)),
                       position=(c[0], c[1], self.height_cm + 0.1), rotation_deg=self.rotation_deg)
            prims += m.primitives()
        return prims

    def top_height(self, x: float, y: float) -> float:
        if self.height_cm <= 0:
            return 0.0
        c, right, up = self._axes()
        d = np.array([x - c[0], y - c[1], 0.0])
        if self.shape == "circle":
            inside = math.hypot(d[0], d[1]) <= self.size_cm / 2
        else:
            inside = abs(d @ right) <= self.size_cm / 2 and abs(d @ up) <= self.size_cm / 2
        return self.height_cm if inside else 0.0

    def collides(self, p, radius):
        if self.height_cm > 0 and p[2] < self.height_cm - 2:
            c, right, up = self._axes()
            d = np.array([p[0] - c[0], p[1] - c[1], 0.0])
            if abs(d @ right) <= self.size_cm / 2 + radius and abs(d @ up) <= self.size_cm / 2 + radius:
                return f"hit the side of landing pad {self.name or ''}".strip()
        return None


# ---------------------------------------------------------------------------------- colored solids
@dataclass
class ColorPanel(SimObject):
    """A flat colored rectangle standing up (a sign, a target). position: [x, y, z] of its
    center; facing_deg: where its face points."""
    position: tuple = (0.0, 100.0, 100.0)
    width_cm: float = 40.0
    height_cm: float = 40.0
    facing_deg: float = 180.0
    color: object = "red"

    def primitives(self) -> list:
        c = np.array(self.position, float)
        n = compass(self.facing_deg)
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(up, n)
        w, h = self.width_cm / 2, self.height_cm / 2
        v = np.array([c - w * right + h * up, c + w * right + h * up, c + w * right - h * up, c - w * right - h * up])
        return [SolidPoly(v, bgr(self.color))]

    def collides(self, p, radius):
        c, n = np.array(self.position, float), compass(self.facing_deg)
        right = np.cross(np.array([0.0, 0.0, 1.0]), n)
        d = p - c
        if abs(d @ n) < radius and abs(d @ right) < self.width_cm / 2 and abs(d[2]) < self.height_cm / 2:
            return f"hit panel {self.name or ''}".strip()
        return None


@dataclass
class Gate(SimObject):
    """A gate to fly through: a rectangular frame (``shape: rect``) or a ring (``shape: circle``).
    position: [x, y, z] of the CENTER of the opening; width_cm/height_cm: the opening (for a
    circle, width_cm is the inner diameter); bar_cm: thickness of the frame; facing_deg: the
    direction you fly through it; legs: posts down to the floor (rect only), in leg_color (default:
    the frame's color; another color, e.g. gray stands, keeps them out of a color mask)."""
    position: tuple = (0.0, 150.0, 100.0)
    width_cm: float = 80.0
    height_cm: float = 80.0
    bar_cm: float = 8.0
    facing_deg: float = 0.0
    color: object = "orange"
    shape: str = "rect"
    legs: bool = True
    leg_color: object = None
    segments: int = 32

    def _axes(self):
        c = np.array(self.position, float)
        n = compass(self.facing_deg)
        up = np.array([0.0, 0.0, 1.0])
        return c, n, np.cross(up, n), up

    def primitives(self) -> list:
        c, n, right, up = self._axes()
        col, b = bgr(self.color), self.bar_cm
        quads = []
        if self.shape == "circle":
            r_in = self.width_cm / 2
            r_out = r_in + b
            for k in range(self.segments):
                a0, a1 = 2 * math.pi * k / self.segments, 2 * math.pi * (k + 1) / self.segments
                p = [c + r * (math.cos(a) * right + math.sin(a) * up) for r, a in
                     ((r_out, a0), (r_out, a1), (r_in, a1), (r_in, a0))]
                quads.append(SolidPoly(np.array(p), col))
            return quads
        w, h = self.width_cm / 2, self.height_cm / 2

        def rect(x0, x1, z0, z1, color=col):
            return SolidPoly(np.array([c + x0 * right + z1 * up, c + x1 * right + z1 * up,
                                       c + x1 * right + z0 * up, c + x0 * right + z0 * up]), color)
        quads += [rect(-w - b, w + b, h, h + b), rect(-w - b, w + b, -h - b, -h),
                  rect(-w - b, -w, -h, h), rect(w, w + b, -h, h)]
        if self.legs and c[2] - h - b > 0:
            z_floor = -c[2]
            leg = col if self.leg_color is None else bgr(self.leg_color)
            quads += [rect(-w - b, -w, z_floor, -h - b, leg), rect(w, w + b, z_floor, -h - b, leg)]
        return quads

    def collides(self, p, radius):
        c, n, right, up = self._axes()
        d = p - c
        if abs(d @ n) > radius + 2:
            return None
        x, z = d @ right, d @ up
        if self.shape == "circle":
            r = math.hypot(x, z)
            hit = r + radius > self.width_cm / 2 and r - radius < self.width_cm / 2 + self.bar_cm
        else:
            w, h = self.width_cm / 2, self.height_cm / 2
            inside_opening = abs(x) + radius < w and abs(z) + radius < h
            within_frame = abs(x) < w + self.bar_cm + radius and abs(z) < h + self.bar_cm + radius
            legs = self.legs and abs(abs(x) - w - self.bar_cm / 2) < self.bar_cm / 2 + radius and z < -h
            hit = (within_frame and not inside_opening) or legs
        return f"hit gate {self.name or ''}".strip() if hit else None


@dataclass
class Box(SimObject):
    """A box on the floor (an obstacle, a platform, a table). position: [x, y] of its center;
    size: [width (x), depth (y), height]; rotation_deg: turn around its center. The ToF reads
    its top when the drone is above it."""
    position: tuple = (0.0, 100.0)
    size: tuple = (40.0, 40.0, 35.0)
    rotation_deg: float = 0.0
    color: object = (160, 120, 80)

    def _corners(self) -> np.ndarray:
        c = np.array([*self.position[:2], 0.0], float)
        ax, ay = compass(self.rotation_deg + 90), compass(self.rotation_deg)
        sx, sy, sz = self.size
        base = [c + i * sx / 2 * ax + j * sy / 2 * ay for i, j in ((-1, 1), (1, 1), (1, -1), (-1, -1))]
        return np.array(base + [p + np.array([0, 0, sz]) for p in base])

    def primitives(self) -> list:
        v = self._corners()
        col = np.array(bgr(self.color), float)
        faces = [([4, 5, 6, 7], (0, 0, 1))]
        center = v.mean(axis=0)
        for i in range(4):
            j = (i + 1) % 4
            faces.append(([i, j, j + 4, i + 4], None))
        out = []
        for idx, nrm in faces:
            pts = v[idx]
            if nrm is None:
                nrm = np.cross(pts[1] - pts[0], pts[3] - pts[0])
                nrm = nrm / (np.linalg.norm(nrm) + 1e-9)
                if nrm @ (pts.mean(axis=0) - center) < 0:
                    nrm = -nrm
            nrm = np.array(nrm, float)
            shade = 0.65 + 0.35 * max(0.0, float(nrm @ np.array([0.3, -0.4, 0.86])))
            out.append(SolidPoly(pts, tuple(int(c) for c in np.clip(col * shade, 0, 255)), nrm))
        return out

    def _inside(self, x, y, margin=0.0) -> bool:
        c = np.array([*self.position[:2], 0.0])
        d = np.array([x, y, 0.0]) - c
        ax, ay = compass(self.rotation_deg + 90), compass(self.rotation_deg)
        return abs(d @ ax) <= self.size[0] / 2 + margin and abs(d @ ay) <= self.size[1] / 2 + margin

    def top_height(self, x, y):
        return float(self.size[2]) if self._inside(x, y) else 0.0

    def collides(self, p, radius):
        if p[2] < self.size[2] - 2 and self._inside(p[0], p[1], radius):
            return f"hit box {self.name or ''}".strip()
        return None


@dataclass
class Wall(SimObject):
    """A vertical wall from start [x, y] to end [x, y]."""
    start: tuple = (-200.0, 300.0)
    end: tuple = (200.0, 300.0)
    height_cm: float = 200.0
    color: object = (200, 200, 190)

    def primitives(self) -> list:
        a, b = np.array([*self.start, 0.0], float), np.array([*self.end, 0.0], float)
        h = np.array([0, 0, self.height_cm])
        return [SolidPoly(np.array([a + h, b + h, b, a]), bgr(self.color))]

    def collides(self, p, radius):
        a, b = np.array(self.start, float), np.array(self.end, float)
        q = np.array(p[:2])
        t = np.clip((q - a) @ (b - a) / max((b - a) @ (b - a), 1e-9), 0, 1)
        if np.linalg.norm(q - (a + t * (b - a))) < radius and p[2] < self.height_cm:
            return f"hit wall {self.name or ''}".strip()
        return None


OBJECT_TYPES = {
    "apriltag": AprilTag, "aruco": ArucoMarker, "marker": Marker, "line": FloorLine,
    "landing_pad": LandingPad, "panel": ColorPanel, "gate": Gate, "box": Box, "wall": Wall,
}


def make_object(spec: dict) -> SimObject:
    """An object from a map entry like ``{"type": "apriltag", "id": 1, "position": [0, 80]}``."""
    spec = dict(spec)
    kind = spec.pop("type", None)
    if kind not in OBJECT_TYPES:
        raise ValueError(f"unknown object type {kind!r} (known: {', '.join(OBJECT_TYPES)})")
    cls = OBJECT_TYPES[kind]
    known = {f for f in cls.__dataclass_fields__ if not f.startswith("_")}
    unknown = set(spec) - known
    if unknown:
        raise ValueError(f"{kind}: unknown fields {sorted(unknown)} (known: {', '.join(sorted(known))})")
    for k in ("position", "size", "start", "end"):
        if isinstance(spec.get(k), list):
            spec[k] = tuple(spec[k])
    return cls(**spec)
