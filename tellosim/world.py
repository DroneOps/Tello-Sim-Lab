"""The simulated world: a floor plus objects (the "map").

Load a built-in map by name, a YAML file, or a dict::

    world = load_map("line_follower")            # tellosim/maps/line_follower.yaml
    world = load_map("my_track.yaml")
    world = World(floor=Floor("plain"), objects=[AprilTag(id=1, position=(0, 80))])
    world.add(Gate(position=(0, 200, 100)))

See ``docs/maps.md`` for the YAML format and every object type.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

from .objects import SimObject, as_motion, bgr, make_object, moved

MAPS_DIR = Path(__file__).resolve().parent / "maps"


@dataclass
class Floor:
    """The ground (z = 0), drawn from a top-view texture.

    kind:
      - ``lab_grid``: the DroneOps lab floor, dark with a grid of light tape (a texture the
        Tello's own downward sensor and a mirror can lock onto).
      - ``plain``: one color with subtle texture (``noise`` = 0 makes it perfectly smooth: the
        real Tello drifts more over a floor without texture).
      - ``checker``: squares of ``spacing_cm`` in two colors.
      - ``image``: a picture of your floor (``image`` path), stretched to ``size_cm``.
    size_cm: width and depth covered by the texture, centered on ``center``; outside it, the
    floor is ``color`` without texture. px_per_cm: texture resolution (2 = 5 mm per pixel)."""
    kind: str = "lab_grid"
    color: object = "dark_floor"
    tape_color: object = "lab_tape"
    tape_cm: float = 6.0
    spacing_cm: float = 60.0
    second_color: object = "white"
    noise: float = 1.0
    image: str | None = None
    size_cm: tuple = (1400.0, 1400.0)
    center: tuple = (0.0, 0.0)
    px_per_cm: float = 2.0
    seed: int = 3

    def texture(self, base_dir: Path | None = None) -> np.ndarray:
        w, h = int(self.size_cm[0] * self.px_per_cm), int(self.size_cm[1] * self.px_per_cm)
        rng = np.random.default_rng(self.seed)
        base = np.array(bgr(self.color), np.float32)
        tex = np.empty((h, w, 3), np.float32)
        tex[:] = base
        if self.kind == "image":
            p = Path(self.image)
            if not p.is_absolute() and base_dir is not None:
                p = base_dir / p
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"floor image not found: {p}")
            tex = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
        elif self.kind == "checker":
            step = max(1, int(self.spacing_cm * self.px_per_cm))
            yy, xx = np.mgrid[0:h, 0:w]
            mask = ((xx // step + yy // step) % 2).astype(bool)
            tex[mask] = np.array(bgr(self.second_color), np.float32)
        if self.noise > 0:  # soft patches + fine grain: texture to measure motion against
            coarse = cv2.resize(rng.normal(0, 1, (max(2, h // 16), max(2, w // 16))).astype(np.float32), (w, h),
                                interpolation=cv2.INTER_CUBIC)
            fine = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 1.5)
            tex += (self.noise * (7 * coarse + 4 * fine))[..., None]
        if self.kind == "lab_grid":
            step, width = int(self.spacing_cm * self.px_per_cm), int(self.tape_cm * self.px_per_cm)
            tape = np.array(bgr(self.tape_color), np.float32)
            grain = rng.normal(0, 4, (h, w, 1)).astype(np.float32)
            for k in range(0, max(w, h), step):
                tex[:, k:k + width] = tape + grain[:, k:k + width]
                tex[k:k + width, :] = tape + grain[k:k + width, :]
        return np.clip(tex, 0, 255).astype(np.uint8)


@dataclass
class Fog:
    """Fog or smoke between the camera and everything it sees (``fog:`` in a map).

    visibility_cm: at this distance things are 95 % hidden (contrast falls as exp(-3 d / visibility)).
    color: of the fog (light gray) or smoke (darker). patches: 0 = even fog; 0.3–1 = smoke, thicker
    and thinner in drifting patches. patch_speed: how fast the patches drift across the image
    (image widths per second). It only changes the image: the ToF and the physics are not affected."""
    visibility_cm: float = 500.0
    color: object = (205, 205, 205)
    patches: float = 0.0
    patch_speed: float = 0.03
    seed: int = 0


def as_fog(spec) -> Fog | None:
    if spec is None or spec is False or isinstance(spec, Fog):
        return spec or None
    if isinstance(spec, (int, float)):
        return Fog(visibility_cm=float(spec))
    unknown = set(spec) - set(Fog.__dataclass_fields__)
    if unknown:
        raise ValueError(f"fog: unknown fields {sorted(unknown)} (known: {', '.join(Fog.__dataclass_fields__)})")
    return Fog(**spec)


@dataclass
class World:
    """A floor, objects, and where the drone starts (x, y in cm, yaw in compass degrees).

    Objects with ``motion`` move with time: the queries below take ``t``, the seconds since the
    simulation started."""
    objects: list = field(default_factory=list)
    floor: Floor = field(default_factory=Floor)
    name: str = ""
    description: str = ""
    start: tuple = (0.0, 0.0, 0.0)
    background: object = (150, 150, 145)  # what the camera sees above the horizon (room walls)
    # camera settings this map is meant for, e.g. {"mode": "mirror"}; Tello(camera_mode=...) wins
    camera: dict = field(default_factory=dict)
    base_dir: Path | None = None  # for relative paths in the map (floor image)
    fog: Fog | None = None  # fog or smoke (see Fog); None = clear air
    _texture: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.fog = as_fog(self.fog)
        for o in self.objects:
            o.motion = as_motion(o.motion)

    # ------------------------------------------------------------------ editing
    def add(self, *objs: SimObject) -> World:
        for o in objs:
            o.motion = as_motion(o.motion)
        self.objects.extend(objs)
        self._texture = None
        return self

    def remove(self, what) -> World:
        """Remove objects by instance, name, or type (e.g. ``world.remove("gate1")``)."""
        def match(o):
            return o is what or (isinstance(what, str) and o.name == what) or (isinstance(what, type) and isinstance(o, what))
        self.objects = [o for o in self.objects if not match(o)]
        self._texture = None
        return self

    def find(self, name: str) -> SimObject | None:
        return next((o for o in self.objects if o.name == name), None)

    # ------------------------------------------------------------------ floor texture
    def to_px(self, x: float, y: float) -> tuple[float, float]:
        """World (x, y) cm -> floor texture pixel (u, v)."""
        f = self.floor
        x0 = f.center[0] - f.size_cm[0] / 2
        y1 = f.center[1] + f.size_cm[1] / 2
        return (x - x0) * f.px_per_cm, (y1 - y) * f.px_per_cm

    @property
    def texture(self) -> np.ndarray:
        """The floor seen from above, with lines and pads painted on it (built once)."""
        if self._texture is None:
            tex = self.floor.texture(self.base_dir)
            for o in self.objects:
                o.paint_floor(tex, self.to_px)
            self._texture = tex
        return self._texture

    # ------------------------------------------------------------------ queries
    @staticmethod
    def offset(o: SimObject, t: float) -> np.ndarray:
        """How far a moving object is from its position at time t (x, y)."""
        return o.motion.offset(t) if o.motion else np.zeros(2)

    @property
    def moving(self) -> bool:
        return any(o.motion for o in self.objects)

    def primitives(self, t: float = 0.0) -> list:
        out = []
        for o in self.objects:
            prims = o.primitives()
            if o.motion:
                off = o.motion.offset(t)
                prims = [moved(p, off) for p in prims]
            out += prims
        return out

    def surface_height(self, x: float, y: float, t: float = 0.0) -> float:
        """Height of whatever is under (x, y): the floor (0) or the top of a box/platform."""
        hs = [0.0]
        for o in self.objects:
            ox, oy = self.offset(o, t)
            hs.append(o.top_height(x - ox, y - oy))
        return max(hs)

    def collision(self, p, radius: float = 10.0, t: float = 0.0) -> str | None:
        p = np.asarray(p, float)
        for o in self.objects:
            ox, oy = self.offset(o, t)
            hit = o.collides(p - np.array([ox, oy, 0.0]), radius)
            if hit:
                return hit
        return None

    def ground_velocity(self, x: float, y: float, t: float = 0.0) -> np.ndarray:
        """Velocity (x, y) of a moving pad or box under (x, y): a drone landed on it moves with it."""
        for o in self.objects:
            if o.motion and hasattr(o, "supports"):
                ox, oy = o.motion.offset(t)
                if o.supports(x - ox, y - oy):
                    return o.motion.velocity(t)
        return np.zeros(2)


def _world_from_dict(spec: dict, base_dir: Path | None = None) -> World:
    spec = dict(spec)
    fspec = dict(spec.get("floor") or {})
    if "type" in fspec:
        fspec["kind"] = fspec.pop("type")
    for k in ("size_cm", "center"):
        if isinstance(fspec.get(k), list):
            fspec[k] = tuple(fspec[k])
    unknown = set(fspec) - {f for f in Floor.__dataclass_fields__}
    if unknown:
        raise ValueError(f"floor: unknown fields {sorted(unknown)}")
    floor = Floor(**fspec)
    start = spec.get("start") or {}
    pos = start.get("position", [0, 0])
    world = World(objects=[make_object(o) for o in spec.get("objects") or []], floor=floor,
                  name=spec.get("name", ""), description=spec.get("description", ""),
                  start=(float(pos[0]), float(pos[1]), float(start.get("yaw_deg", 0.0))), base_dir=base_dir)
    if "background" in spec:
        world.background = spec["background"]
    world.camera = dict(spec.get("camera") or {})
    world.fog = as_fog(spec.get("fog"))
    unknown = set(spec) - {"name", "description", "floor", "start", "objects", "background", "camera", "fog"}
    if unknown:
        raise ValueError(f"map: unknown fields {sorted(unknown)} (known: name, description, floor, start, objects, "
                         f"background, camera, fog)")
    return world


def load_map(spec: str | Path | dict | World = "lab_grid") -> World:
    """A built-in map name, a YAML/JSON file, a dict with the same structure, or a World."""
    if isinstance(spec, World):
        return spec
    if isinstance(spec, dict):
        return _world_from_dict(spec)
    p = Path(spec)
    if not p.exists():
        p = MAPS_DIR / f"{spec}.yaml"
    if not p.exists():
        names = sorted(x.stem for x in MAPS_DIR.glob("*.yaml"))
        raise FileNotFoundError(f"map {spec!r} not found (built-in: {', '.join(names)})")
    return _world_from_dict(yaml.safe_load(p.read_text()) or {}, p.parent)


def builtin_maps() -> list[str]:
    return sorted(x.stem for x in MAPS_DIR.glob("*.yaml"))
