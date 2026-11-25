"""Draws what the camera sees.

1. The floor, by ray casting: every pixel's ray (with the real lens distortion) is intersected
   with z = 0 and the floor texture is sampled there (cv2.remap). This works for the forward
   camera and for the mirror (downward) camera alike.
2. Objects, far to near (painter's algorithm): flat pictures (markers, pads) with a homography,
   colored polygons (gates, boxes, walls, panels) clipped against the near plane.
   Things lying on the floor (markers, flat pads) are drawn first, lowest first; a picture on a
   surface (a marker on a wall or on a pad) is drawn right after that surface.
3. Fog or smoke (``world.fog``): every pixel fades towards the fog color with its distance.
4. Lens blur and sensor noise, then the mirror flip if the camera has one.

Objects with ``motion`` are drawn where they are at time ``t`` (``render(..., t=...)``).

Rendering a 960x720 frame takes ~10–25 ms on a laptop.
"""
from __future__ import annotations

import cv2
import numpy as np

from .camera import Camera
from .objects import SolidPoly, TexturedQuad, bgr, moved
from .world import World

NEAR_CM = 5.0
ROOM_CM = 1000.0  # how far the background (the room's walls) is, for the fog
FLAT_CM = 1.0  # primitives lower than this lie on the floor


def _clip_near(pc: np.ndarray, near: float = NEAR_CM) -> np.ndarray:
    """Sutherland–Hodgman clip of a camera-frame polygon against z >= near."""
    out = []
    n = len(pc)
    for i in range(n):
        a, b = pc[i], pc[(i + 1) % n]
        ina, inb = a[2] >= near, b[2] >= near
        if ina:
            out.append(a)
        if ina != inb:
            t = (near - a[2]) / (b[2] - a[2])
            out.append(a + t * (b - a))
    return np.array(out) if out else np.zeros((0, 3))


class Renderer:
    def __init__(self, world: World, camera: Camera, seed: int = 0, floor_scale: float = 0.5) -> None:
        """floor_scale: resolution of the floor relative to the image (0.5 = half; the floor is
        smooth and gets blurred anyway; markers and objects are always drawn at full size)."""
        self.world, self.cam = world, camera
        self.rng = np.random.default_rng(seed + 101)
        rays = camera.rays()
        h, w = camera.height, camera.width
        self.floor_size = (max(1, int(w * floor_scale)), max(1, int(h * floor_scale)))
        small = cv2.resize(rays, self.floor_size, interpolation=cv2.INTER_AREA) if floor_scale != 1 else rays
        self.xn, self.yn = small[..., 0].copy(), small[..., 1].copy()
        self._noise = [self.rng.normal(0, camera.noise_std, (h, w, 1)).astype(np.float32) for _ in range(6)] \
            if camera.noise_std > 0 else []
        self._k = 0
        self._prims = None

    def invalidate(self) -> None:
        """Call after changing the world's objects."""
        self._prims = None

    def _prepare(self) -> None:
        """Primitives (with the object they belong to), and for each flat picture (marker) the
        surface it is stuck on, if any: it is drawn right after that surface. Sorting by mean
        depth alone puts a big wall in front of the markers on its lower half."""
        prims, owner = [], []
        for o in self.world.objects:
            for p in o.primitives():
                prims.append(p)
                owner.append(o)
        self._moving = {i: o for i, o in enumerate(owner) if o.motion}
        self._host: dict[int, int] = {}

        def verts(q):
            return q.corners if isinstance(q, TexturedQuad) else q.vertices

        for i, p in enumerate(prims):
            if not isinstance(p, TexturedQuad):
                continue
            lo, hi = p.corners.min(axis=0), p.corners.max(axis=0)
            best = None
            for j, q in enumerate(prims):
                v = verts(q)
                if j == i or owner[j] is not owner[i] and isinstance(q, TexturedQuad) or len(v) < 3:
                    continue
                n = np.cross(v[1] - v[0], v[2] - v[0])
                n = n / (np.linalg.norm(n) + 1e-9)
                inside = (lo >= v.min(axis=0) - 3.0).all() and (hi <= v.max(axis=0) + 3.0).all()
                bigger = np.ptp(v, axis=0).sum() > np.ptp(p.corners, axis=0).sum()
                if inside and bigger and np.abs((p.corners - v[0]) @ n).max() < 3.0:
                    best = j
                    if isinstance(q, SolidPoly):
                        break
            if best is not None:
                self._host[i] = best
        self._flat = [verts(p)[:, 2].max() < FLAT_CM for p in prims]
        self._low = [float(verts(p)[:, 2].mean()) for p in prims]
        self._prims = prims
        fog = self.world.fog
        self._fog_rgb = np.array(bgr(fog.color), np.float32) if fog else None

    def _floor(self, O: np.ndarray, r, d, f, want_depth: bool = False):
        w = self.world
        wx = self.xn * r[0] + self.yn * d[0] + f[0]
        wy = self.xn * r[1] + self.yn * d[1] + f[1]
        wz = self.xn * r[2] + self.yn * d[2] + f[2]
        hit = wz < -1e-4
        z = max(O[2], 0.5)
        t = np.where(hit, -z / np.where(hit, wz, -1.0), 0.0).astype(np.float32)
        X = O[0] + t * wx
        Y = O[1] + t * wy
        fl = w.floor
        u = (X - (fl.center[0] - fl.size_cm[0] / 2)) * fl.px_per_cm
        v = ((fl.center[1] + fl.size_cm[1] / 2) - Y) * fl.px_per_cm
        far = t > 3000  # beyond 30 m: background
        u = np.where(hit & ~far, u, -10).astype(np.float32)
        v = np.where(hit & ~far, v, -10).astype(np.float32)
        tex = w.texture
        img = cv2.remap(tex, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                        borderValue=bgr(fl.color))
        img[~hit | far] = bgr(w.background)
        depth = None
        if want_depth:  # distance along each pixel's ray (the ray x*r + y*d + f is not unit length)
            depth = np.where(hit & ~far, t * np.sqrt(self.xn ** 2 + self.yn ** 2 + 1), ROOM_CM).astype(np.float32)
        if img.shape[1] != self.cam.width:
            size = (self.cam.width, self.cam.height)
            img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
            if depth is not None:
                depth = cv2.resize(depth, size, interpolation=cv2.INTER_LINEAR)
        return img, depth

    def render(self, x: float, y: float, z: float, yaw_deg: float, t: float = 0.0) -> np.ndarray:
        """BGR image as the camera delivers it (flipped in mirror mode). t: seconds since the
        simulation started (where moving objects are, how the smoke drifts)."""
        cam = self.cam
        O = np.array([x, y, z], float)
        r, d, f = cam.basis(yaw_deg)
        if self._prims is None:
            self._prepare()
        fog = self.world.fog
        img, depth = self._floor(O, r, d, f, want_depth=fog is not None)
        prims = self._prims
        if self._moving:
            prims = list(prims)
            for i, o in self._moving.items():
                prims[i] = moved(self._prims[i], o.motion.offset(t))
        R = np.stack([r, d, f])
        jobs, decals = [], {}
        for i, p in enumerate(prims):
            if isinstance(p, TexturedQuad):
                if (O - p.corners.mean(axis=0)) @ p.normal <= 0:
                    continue  # seen from behind
                pc = (p.corners - O) @ R.T
                if (pc[:, 2] < 10).any():
                    continue
                if i in self._host:
                    decals.setdefault(self._host[i], []).append((p, pc))
                    continue
            else:
                if p.normal is not None and (O - p.vertices.mean(axis=0)) @ p.normal <= 0:
                    continue
                pc = _clip_near((p.vertices - O) @ R.T)
                if len(pc) < 3:
                    continue
            jobs.append((float(pc[:, 2].mean()), p, pc, i))
        # on the floor first (lowest first: a marker on a flat pad goes on top of it), then far to near
        jobs.sort(key=lambda j: (0, self._low[j[3]], -j[0]) if self._flat[j[3]] else (1, 0.0, -j[0]))
        for _, p, pc, i in jobs:
            self._draw(img, p, pc, depth)
            stack = list(decals.get(i, []))
            while stack:  # decals, and decals on decals (a marker on a raised pad's top)
                dp, dpc = stack.pop(0)
                self._draw(img, dp, dpc, depth)
                j = next((k for k, q in enumerate(prims) if q is dp), None)
                stack += decals.get(j, []) if j is not None else []
        if fog is not None:
            img = self._apply_fog(img, depth, fog, t)
        if cam.blur_sigma > 0:
            img = cv2.GaussianBlur(img, (3, 3), cam.blur_sigma)
        if self._noise:
            self._k = (self._k + 1) % len(self._noise)
            img = np.clip(img.astype(np.float32) + self._noise[self._k], 0, 255).astype(np.uint8)
        return cam.finish(img)

    def _draw(self, img, p, pc, depth) -> None:
        if isinstance(p, TexturedQuad):
            self._draw_quad(img, p, pc, depth)
        else:
            self._draw_poly(img, p, pc, depth)

    # ------------------------------------------------------------------ fog
    def _smoke(self, fog, t: float, size: tuple[int, int]) -> np.ndarray:
        """Smoke density field in [-1, 1] over the image: a few soft waves that drift sideways."""
        if getattr(self, "_waves", None) is None:
            rng = np.random.default_rng(fog.seed + 7)
            self._waves = [(rng.uniform(0.6, 3.0), rng.uniform(0, 2 * np.pi), rng.uniform(0, 2 * np.pi),
                            rng.uniform(0.4, 1.0)) for _ in range(7)]
            gh, gw = 45, 60
            self._grid = np.meshgrid(np.linspace(0, 1, gw, dtype=np.float32),
                                     np.linspace(0, 0.75, gh, dtype=np.float32))
        u, v = self._grid
        u = u - fog.patch_speed * t
        field = np.zeros_like(u)
        for freq, ang, phase, amp in self._waves:
            field += amp * np.sin(2 * np.pi * freq * (u * np.cos(ang) + v * np.sin(ang)) + phase + 0.3 * t * freq)
        field /= np.abs(field).max() + 1e-6
        return cv2.resize(field, size, interpolation=cv2.INTER_CUBIC)

    def _apply_fog(self, img: np.ndarray, depth: np.ndarray, fog, t: float) -> np.ndarray:
        """Each pixel fades towards the fog color: transmission exp(-3 d / visibility). Computed at
        half resolution (fog is smooth) and blended with OpenCV."""
        h, w = depth.shape
        small = (max(1, w // 2), max(1, h // 2))
        d = cv2.resize(depth, small, interpolation=cv2.INTER_AREA)
        k = 3.0 / max(fog.visibility_cm, 1.0)
        if fog.patches > 0:
            k = k * np.clip(1.0 + fog.patches * self._smoke(fog, t, small), 0.0, None)
        trans = cv2.resize(np.exp(-k * d).astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        if getattr(self, "_fog_img", None) is None or self._fog_img.shape != img.shape:
            self._fog_img = np.empty_like(img)
            self._fog_img[:] = self._fog_rgb.astype(np.uint8)
        return cv2.blendLinear(img, self._fog_img, trans, 1.0 - trans)

    def _plane_depth(self, pc: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
        """Distance along each pixel's ray to the plane of a camera-frame polygon, in a box."""
        n = np.cross(pc[1] - pc[0], pc[2] - pc[0])
        rays = self.cam.rays()[y0:y1, x0:x1]
        dot = rays[..., 0] * n[0] + rays[..., 1] * n[1] + n[2]
        dot = np.where(np.abs(dot) < 1e-9, 1e-9, dot)
        s = float(n @ pc[0]) / dot
        return (np.abs(s) * np.sqrt(rays[..., 0] ** 2 + rays[..., 1] ** 2 + 1)).astype(np.float32)

    def _draw_quad(self, img: np.ndarray, p: TexturedQuad, pc: np.ndarray, depth=None) -> None:
        uv = self.cam.project(pc)
        if uv is None or (np.abs(uv) > 5000).any():
            return
        H_img, W_img = img.shape[:2]
        x0, y0 = np.floor(uv.min(axis=0)).astype(int)
        x1, y1 = np.ceil(uv.max(axis=0)).astype(int)
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1 + 1, W_img), min(y1 + 1, H_img)
        if x1 <= x0 or y1 <= y0:
            return
        h, w = p.image.shape[:2]
        dst = (uv - np.array([x0, y0])).astype(np.float32)
        Hm = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), dst)
        roi = img[y0:y1, x0:x1]
        size = (x1 - x0, y1 - y0)
        alpha = None
        if p.image.shape[2] == 4:  # a picture with transparent parts (a round pad)
            col = cv2.warpPerspective(p.image[..., :3], Hm, size, flags=cv2.INTER_LINEAR)
            alpha = cv2.warpPerspective(p.image[..., 3], Hm, size, flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(np.float32) / 255
            a = alpha[..., None]
            roi[:] = (roi.astype(np.float32) * (1 - a) + col.astype(np.float32) * a).astype(np.uint8)
        else:
            cv2.warpPerspective(p.image, Hm, size, roi, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
        if depth is not None:
            if alpha is None:
                alpha = cv2.warpPerspective(np.ones((h, w), np.float32), Hm, size, flags=cv2.INTER_NEAREST,
                                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            m = alpha > 0.5
            depth[y0:y1, x0:x1][m] = self._plane_depth(pc, x0, y0, x1, y1)[m]

    def _draw_poly(self, img: np.ndarray, p: SolidPoly, pc: np.ndarray, depth=None) -> None:
        uv = self.cam.project(pc)
        if uv is None:
            return
        uv = np.clip(uv, -20000, 20000)
        pts = np.round(uv * 16).astype(np.int32)
        cv2.fillPoly(img, [pts], p.color, cv2.LINE_AA, 4)
        if depth is not None:
            H_img, W_img = img.shape[:2]
            x0, y0 = max(int(np.floor(uv[:, 0].min())), 0), max(int(np.floor(uv[:, 1].min())), 0)
            x1, y1 = min(int(np.ceil(uv[:, 0].max())) + 1, W_img), min(int(np.ceil(uv[:, 1].max())) + 1, H_img)
            if x1 <= x0 or y1 <= y0:
                return
            mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
            cv2.fillPoly(mask, [pts - np.array([x0 * 16, y0 * 16], np.int32)], 1, cv2.LINE_8, 4)
            m = mask > 0
            depth[y0:y1, x0:x1][m] = self._plane_depth(pc, x0, y0, x1, y1)[m]
