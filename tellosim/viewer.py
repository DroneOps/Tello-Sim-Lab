"""A view of the simulation: the map from above (where the drone really is, what the camera sees,
its trail) next to the camera image. Live in a window and/or recorded to an mp4.

    tello = Tello(world="color_gates")
    viewer = tello.sim.viewer(record="run.mp4")   # starts drawing (10 fps) in the background
    ...
    viewer.stop()                                 # closes the window and finishes the video

In simulated time (realtime=False) nothing runs by itself: call ``viewer.draw()`` after each
``tello.sim.advance(...)``.
"""
from __future__ import annotations

import math
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .objects import Box, ColorPanel, FloorLine, Gate, LandingPad, Marker, Wall, bgr, compass

MAP_PX = 540
W_OUT, H_OUT = 960, 540
RED, WHITE, CYAN = (60, 60, 230), (240, 240, 240), (200, 200, 60)


class Viewer:
    def __init__(self, sim, show: bool = True, record: str | None = None, fps: float = 10.0,
                 title: str = "tellosim") -> None:
        self.sim, self.show, self.fps, self.title = sim, show, fps, title
        self.trail: list = []
        self._writer = None
        self._record = record
        self._stop = threading.Event()
        self._thread = None
        self._bounds()
        self._background = self._floor_image()

    # ------------------------------------------------------------------ map geometry
    def _bounds(self) -> None:
        pts = [self.sim.world.start[:2]]
        for o in self.sim.world.objects:
            for k in ("position", "start", "end"):
                v = getattr(o, k, None)
                if v is not None:
                    pts.append(tuple(v[:2]))
            if isinstance(o, FloorLine):
                pts += [tuple(p) for p in o.points]
            if o.motion and getattr(o, "position", None) is not None:  # the whole run of a moving object
                m_ = o.motion
                d = compass(m_.direction_deg)[:2] * m_.distance_cm
                x, y = o.position[:2]
                pts += [(x + d[0] / 2, y + d[1] / 2), (x - d[0] / 2, y - d[1] / 2), (x + d[0], y + d[1])]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        m = 120
        self.x0, self.x1, self.y0, self.y1 = min(xs) - m, max(xs) + m, min(ys) - m, max(ys) + m
        span = max(self.x1 - self.x0, self.y1 - self.y0)
        cx, cy = (self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2
        self.x0, self.x1, self.y0, self.y1 = cx - span / 2, cx + span / 2, cy - span / 2, cy + span / 2
        self.k = (MAP_PX - 20) / span

    def px(self, x: float, y: float) -> tuple[int, int]:
        return int(10 + (x - self.x0) * self.k), int(10 + (self.y1 - y) * self.k)

    def _floor_image(self) -> np.ndarray:
        w = self.sim.world
        tex = w.texture
        u0, v0 = w.to_px(self.x0, self.y1)
        u1, v1 = w.to_px(self.x1, self.y0)
        h, wd = tex.shape[:2]
        canvas = np.full((MAP_PX, MAP_PX, 3), bgr(w.floor.color), np.uint8)
        a0, b0, a1, b1 = max(0, int(u0)), max(0, int(v0)), min(wd, int(u1)), min(h, int(v1))
        if a1 > a0 and b1 > b0:
            crop = tex[b0:b1, a0:a1]
            sx, sy = (MAP_PX - 20) / (u1 - u0), (MAP_PX - 20) / (v1 - v0)
            dst = cv2.resize(crop, (max(1, int((a1 - a0) * sx)), max(1, int((b1 - b0) * sy))), interpolation=cv2.INTER_AREA)
            ox, oy = 10 + int((a0 - u0) * sx), 10 + int((b0 - v0) * sy)
            hh, ww = min(dst.shape[0], MAP_PX - oy), min(dst.shape[1], MAP_PX - ox)
            canvas[oy:oy + hh, ox:ox + ww] = dst[:hh, :ww]
        for o in w.objects:
            if not o.motion:  # moving objects are drawn on every frame
                self._draw_object(canvas, o)
        return canvas

    def _draw_object(self, img, o, off=(0.0, 0.0)) -> None:
        def P(x, y):
            return self.px(x + off[0], y + off[1])
        if isinstance(o, Marker):
            c = P(*o.position[:2])
            s = max(3, int(o.size_cm * self.k / 2))
            cv2.rectangle(img, (c[0] - s, c[1] - s), (c[0] + s, c[1] + s), WHITE, -1)
            cv2.putText(img, str(o.id), (c[0] + s + 2, c[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1, cv2.LINE_AA)
        elif isinstance(o, LandingPad):
            c, s = P(*o.position[:2]), max(3, int(o.size_cm * self.k / 2))
            if o.shape == "circle":
                cv2.circle(img, c, s, bgr(o.border_color), 2, cv2.LINE_AA)
            else:
                cv2.rectangle(img, (c[0] - s, c[1] - s), (c[0] + s, c[1] + s), bgr(o.border_color), 2)
        elif isinstance(o, (Gate, ColorPanel)):
            c, n = np.array(o.position, float), compass(o.facing_deg)
            right = np.cross(np.array([0, 0, 1.0]), n)
            half = o.width_cm / 2 + getattr(o, "bar_cm", 0)
            a, b = c - half * right, c + half * right
            cv2.line(img, P(a[0], a[1]), P(b[0], b[1]), bgr(o.color), 4, cv2.LINE_AA)
        elif isinstance(o, Box):
            v = o._corners()[:4]
            cv2.fillPoly(img, [np.array([P(p[0], p[1]) for p in v], np.int32)], bgr(o.color))
        elif isinstance(o, Wall):
            cv2.line(img, P(*o.start), P(*o.end), bgr(o.color), 5)

    def _footprint(self, x, y, z, yaw):
        """Where the image corners hit the floor."""
        cam = self.sim.camera
        r, d, f = cam.basis(yaw)
        K, dist = cam.render_K
        pts = []
        for u, v in ((0, 0), (cam.width, 0), (cam.width, cam.height), (0, cam.height)):
            xn, yn = cv2.undistortPoints(np.float32([[[u, v]]]), K, dist).ravel()
            ray = xn * r + yn * d + f
            if ray[2] >= -0.05:
                return None
            t = -max(z, 1) / ray[2]
            pts.append((x + t * ray[0], y + t * ray[1]))
        return pts

    # ------------------------------------------------------------------ drawing
    def draw(self) -> np.ndarray:
        """Draw one frame (and show/record it). Returns the BGR image."""
        x, y, z, yaw = self.sim.pose()
        if not self.trail or math.hypot(x - self.trail[-1][0], y - self.trail[-1][1]) > 2:
            self.trail.append((x, y))
        out = np.full((H_OUT, W_OUT, 3), 30, np.uint8)
        m = self._background.copy()
        t_now = self.sim.clock.now() - self.sim.drone.t0
        for o in self.sim.world.objects:
            if o.motion:
                self._draw_object(m, o, o.motion.offset(t_now))
        fp = self._footprint(x, y, z, yaw)
        if fp and z > 5:
            cv2.polylines(m, [np.array([self.px(*p) for p in fp], np.int32)], True, CYAN, 1, cv2.LINE_AA)
        if len(self.trail) > 1:
            cv2.polylines(m, [np.array([self.px(*p) for p in self.trail], np.int32)], False, RED, 1, cv2.LINE_AA)
        c = self.px(x, y)
        h = compass(yaw)
        cv2.circle(m, c, 7, RED, 2, cv2.LINE_AA)
        cv2.line(m, c, self.px(x + 25 * h[0], y + 25 * h[1]), RED, 2, cv2.LINE_AA)
        out[:, :MAP_PX] = m
        cam = self.sim.capture()
        if cam is not None:
            img = cv2.cvtColor(cam, cv2.COLOR_RGB2BGR)
            out[10:310, MAP_PX + 10:MAP_PX + 410] = cv2.resize(img, (400, 300), interpolation=cv2.INTER_AREA)
        d = self.sim.drone
        lines = [self.sim.world.name or "world",
                 f"t {self.sim.clock.now() - d.t0:5.1f} s   {d.mode}", f"x {x:+.0f}  y {y:+.0f}  z {z:.0f} cm  yaw {yaw:+.0f}",
                 f"rc {d.rc}   battery {d.bat:.0f}%", f"camera: {self.sim.camera.mode}"]
        if d.crash:
            lines.append(f"CRASH: {d.crash}")
        for i, t in enumerate(lines):
            cv2.putText(out, t, (MAP_PX + 12, 340 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (90, 90, 240) if t.startswith("CRASH") else WHITE, 1, cv2.LINE_AA)
        if self.show:
            cv2.imshow(self.title, out)
            cv2.waitKey(1)
        if self._record:
            if self._writer is None:
                Path(self._record).parent.mkdir(parents=True, exist_ok=True)
                self._writer = cv2.VideoWriter(self._record, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (W_OUT, H_OUT))
            self._writer.write(out)
        return out

    def start(self) -> Viewer:
        if self.sim.realtime:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def _loop(self) -> None:
        period = 1.0 / self.fps
        while not self._stop.is_set():
            t0 = time.monotonic()
            self.draw()
            time.sleep(max(0.0, period - (time.monotonic() - t0)))

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        if self._writer:
            self._writer.release()
            self._writer = None
        if self.show:
            try:
                cv2.destroyWindow(self.title)
            except cv2.error:
                pass
