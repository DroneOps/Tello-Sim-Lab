"""How far the drone moved over the floor, from a downward camera (a mirror on the Tello).

Phase correlation between each frame and the previous one, converted to cm with the ToF height.
Position and velocity are in the drone frame: (right, forward), in cm and cm/s.

Notes from the lab:
- It needs texture on the floor (the lab tape grid works; a plain shiny floor does not).
- It does not measure while turning (the image rotates): pass ``turning=True``.
- When climbing or descending the floor grows or shrinks in the image, and the correlation
  confused that with motion (+10 cm sideways going from 30 to 100 cm). The previous frame is
  scaled by the height change first, around the pixel right below the drone
  (:func:`point_below_drone`), which depends on the mirror tilt.
"""
from __future__ import annotations

import math
import threading
from collections import deque

import cv2
import numpy as np


def point_below_drone(K: np.ndarray, view_forward_deg: float, view_side_deg: float = 0.0) -> tuple[float, float]:
    """Pixel of the (flipped back) mirror image right below the drone: the optical center of the
    mirrored camera shifted by the view tilt. Tilted forward, what is below the drone appears
    lower in the image; tilted right, more to the left."""
    return (float(K[0, 2] - K[0, 0] * math.tan(math.radians(view_side_deg))),
            float(K[1, 2] + K[1, 1] * math.tan(math.radians(view_forward_deg))))


class FloorFlow:
    """``flow.update(gray, t, alt_cm, turning)`` on every frame (the image already flipped back);
    ``pos, vel = flow.state()``; ``flow.reset(pos)`` to set where you are now (e.g. (0, 0) at
    takeoff, or the drone position relative to a tag)."""

    SIZE = (240, 180)  # working resolution: ~2 ms per frame

    def __init__(self, fx: float, min_quality: float = 0.1, center: tuple[float, float] = (480.0, 360.0),
                 image_width: int = 960) -> None:
        self.k = self.SIZE[0] / image_width
        self.f = fx * self.k
        self.c = (center[0] * self.k, center[1] * self.k)
        self.min_quality = min_quality
        self.win = cv2.createHanningWindow(self.SIZE, cv2.CV_32F)
        self.prev, self.t_prev = None, 0.0
        self.alts: deque = deque()
        self.pos = np.zeros(2)
        self.vel = np.zeros(2)
        self.quality, self.t_ok = 0.0, -math.inf
        self.lock = threading.Lock()

    def _vz(self) -> float:
        """Vertical speed (cm/s) robust to isolated ToF readings: median of the slopes."""
        h = self.alts
        s = [(h[j][1] - h[i][1]) / (h[j][0] - h[i][0]) for i in range(len(h)) for j in range(i + 1, len(h))
             if h[j][0] - h[i][0] > 0.05]
        return float(np.median(s)) if len(s) >= 3 else 0.0

    def update(self, gray: np.ndarray, t: float, alt_cm: float, turning: bool = False) -> None:
        small = cv2.resize(gray, self.SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)
        if not self.alts or self.alts[-1][1] != alt_cm or t - self.alts[-1][0] > 0.09:
            self.alts.append((t, alt_cm))
        while self.alts and t - self.alts[0][0] > 0.6:
            self.alts.popleft()
        if self.prev is not None and not turning and alt_cm > 25 and t > self.t_prev:
            a = float(np.median([x[1] for x in self.alts]))
            m = min(1.1, max(0.9, (a - self._vz() * (t - self.t_prev)) / a))
            ref = self.prev
            if abs(m - 1) > 1e-4:  # the previous frame at today's scale
                M = np.float32([[m, 0, (1 - m) * self.c[0]], [0, m, (1 - m) * self.c[1]]])
                ref = cv2.warpAffine(self.prev, M, self.SIZE, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            (dx, dy), q = cv2.phaseCorrelate(ref, small, self.win)
            self.quality = q
            dt, cm = t - self.t_prev, alt_cm / self.f
            v = np.array([-dx * cm / dt, dy * cm / dt])  # the floor moves opposite to the drone
            if q >= self.min_quality and np.hypot(*v) < 150:  # faster than 1.5 m/s: wrong match
                with self.lock:
                    self.pos += v * dt
                    self.vel += 0.25 * (v - self.vel)
                    self.t_ok = t
        self.prev, self.t_prev = (None if turning else small), t

    def reset(self, pos=(0.0, 0.0)) -> np.ndarray:
        """Take the current position as ``pos``. Returns how much the origin moved."""
        with self.lock:
            delta = np.asarray(pos, float) - self.pos
            self.pos[:] = pos
            return delta

    def state(self) -> tuple[np.ndarray, np.ndarray]:
        """(position, velocity) as (right, forward) arrays, cm and cm/s."""
        with self.lock:
            return self.pos.copy(), self.vel.copy()

    def valid(self, now: float, max_age_s: float = 0.5) -> bool:
        """A good measurement arrived recently."""
        return now - self.t_ok < max_age_s
