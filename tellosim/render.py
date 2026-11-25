"""Draws what the camera sees.

1. The floor, by ray casting: every pixel's ray (with the real lens distortion) is intersected
   with z = 0 and the floor texture is sampled there (cv2.remap). This works for the forward
   camera and for the mirror (downward) camera alike.
2. Objects, far to near (painter's algorithm): flat pictures (markers, pads) with a homography,
   colored polygons (gates, boxes, walls, panels) clipped against the near plane.
3. Lens blur and sensor noise, then the mirror flip if the camera has one.

Rendering a 960x720 frame takes ~10–25 ms on a laptop.
"""
from __future__ import annotations

import cv2
import numpy as np

from .camera import Camera
from .objects import SolidPoly, TexturedQuad, bgr
from .world import World

NEAR_CM = 5.0


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
        """Primitives, and for each flat picture (marker) the surface it is stuck on, if any: it
        is drawn right after that surface. Sorting by mean depth alone puts a big wall in front
        of the markers on its lower half."""
        prims = self.world.primitives()
        self._host: dict[int, int] = {}
        for i, p in enumerate(prims):
            if not isinstance(p, TexturedQuad):
                continue
            for j, q in enumerate(prims):
                if isinstance(q, SolidPoly) and len(q.vertices) >= 3:
                    n = np.cross(q.vertices[1] - q.vertices[0], q.vertices[2] - q.vertices[0])
                    n = n / (np.linalg.norm(n) + 1e-9)
                    if np.abs((p.corners - q.vertices[0]) @ n).max() < 3.0:
                        self._host[i] = j
                        break
        self._prims = prims

    def _floor(self, O: np.ndarray, r, d, f) -> np.ndarray:
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
        if img.shape[1] != self.cam.width:
            img = cv2.resize(img, (self.cam.width, self.cam.height), interpolation=cv2.INTER_LINEAR)
        return img

    def render(self, x: float, y: float, z: float, yaw_deg: float) -> np.ndarray:
        """BGR image as the camera delivers it (flipped in mirror mode)."""
        cam = self.cam
        O = np.array([x, y, z], float)
        r, d, f = cam.basis(yaw_deg)
        img = self._floor(O, r, d, f)
        if self._prims is None:
            self._prepare()
        R = np.stack([r, d, f])
        jobs, decals = [], {}
        for i, p in enumerate(self._prims):
            if isinstance(p, TexturedQuad):
                if (O - p.corners.mean(axis=0)) @ p.normal <= 0:
                    continue  # seen from behind
                pc = (p.corners - O) @ R.T
                if (pc[:, 2] < 10).any():
                    continue
                if i in self._host:
                    decals.setdefault(self._host[i], []).append((p, pc))
                else:
                    jobs.append((float(pc[:, 2].mean()), p, pc, i))
            else:
                if p.normal is not None and (O - p.vertices.mean(axis=0)) @ p.normal <= 0:
                    continue
                pc = _clip_near((p.vertices - O) @ R.T)
                if len(pc) < 3:
                    continue
                jobs.append((float(pc[:, 2].mean()), p, pc, i))
        jobs.sort(key=lambda j: -j[0])
        for _, p, pc, i in jobs:
            if isinstance(p, TexturedQuad):
                self._draw_quad(img, p, pc)
            else:
                self._draw_poly(img, p, pc)
                for dp, dpc in decals.get(i, []):
                    self._draw_quad(img, dp, dpc)
        if cam.blur_sigma > 0:
            img = cv2.GaussianBlur(img, (3, 3), cam.blur_sigma)
        if self._noise:
            self._k = (self._k + 1) % len(self._noise)
            img = np.clip(img.astype(np.float32) + self._noise[self._k], 0, 255).astype(np.uint8)
        return cam.finish(img)

    def _draw_quad(self, img: np.ndarray, p: TexturedQuad, pc: np.ndarray) -> None:
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
        cv2.warpPerspective(p.image, Hm, (x1 - x0, y1 - y0), roi, flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_TRANSPARENT)

    def _draw_poly(self, img: np.ndarray, p: SolidPoly, pc: np.ndarray) -> None:
        uv = self.cam.project(pc)
        if uv is None:
            return
        uv = np.clip(uv, -20000, 20000)
        cv2.fillPoly(img, [np.round(uv * 16).astype(np.int32)], p.color, cv2.LINE_AA, 4)
