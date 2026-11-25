"""The simulated camera: intrinsics, where it looks, and image effects.

Two mountings:

- ``"forward"``: the Tello's own camera. It looks forward, tilted ``pitch_deg`` below the
  horizon (11° measured on the lab Tellos).
- ``"mirror"``: a mirror clipped in front of the lens makes it look DOWN (the AprilTag-Detection
  strategy). The view is tilted ``mirror_tilt_deg`` from straight down (+ = towards the drone's
  nose; the lab clip measured -10.3°, i.e. a bit towards the back), and the image comes out
  flipped like the real one (``mirror_axis``: "v" flips top/bottom, "h" left/right). Flip it back
  before detecting tags: a mirrored tag does not decode.

See ``docs/camera.md`` for calibrating another Tello and every field.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

CAMERAS_DIR = Path(__file__).resolve().parent / "cameras"


@dataclass
class Camera:
    K: np.ndarray  # 3x3 intrinsics of the image as the camera delivers it
    dist: np.ndarray  # OpenCV distortion (k1, k2, p1, p2, k3)
    width: int = 960
    height: int = 720
    mode: str = "forward"  # "forward" or "mirror"
    pitch_deg: float = 11.0  # forward: optical axis below the horizon
    mirror_axis: str = "v"  # mirror: how the image is flipped
    mirror_tilt_deg: float = -10.3  # mirror: view tilt from straight down (+ = towards the nose)
    fps: float = 30.0
    noise_std: float = 3.0  # gray levels of sensor noise
    blur_sigma: float = 0.6  # slight lens/compression blur
    description: str = ""
    _rays: np.ndarray | None = field(default=None, repr=False)

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, camera: str | Path | dict | Camera = "lab_tello", **overrides) -> Camera:
        """A built-in calibration name ("lab_tello"), a JSON file with "K" and "dist" (the
        format of camera calibration tools), a dict, or a Camera. ``overrides`` change any field,
        e.g. ``Camera.load(mode="mirror")``."""
        if isinstance(camera, Camera):
            cam = camera
        else:
            if isinstance(camera, dict):
                data = dict(camera)
            else:
                p = Path(camera)
                if not (p.suffix == ".json" and p.exists()):
                    p = CAMERAS_DIR / f"{camera}.json"
                if not p.exists():
                    names = sorted(x.stem for x in CAMERAS_DIR.glob("*.json"))
                    raise FileNotFoundError(f"camera {camera!r} not found (built-in: {', '.join(names)})")
                data = json.loads(p.read_text())
            w, h = data.get("resolution", (data.get("width", 960), data.get("height", 720)))
            cam = cls(K=np.array(data["K"], float), dist=np.array(data["dist"], float).ravel(),
                      width=int(w), height=int(h), pitch_deg=float(data.get("pitch_deg", 11.0)),
                      description=data.get("description", ""))
        for k, v in overrides.items():
            if not hasattr(cam, k) or k.startswith("_"):
                raise ValueError(f"unknown camera field {k!r}")
            setattr(cam, k, v)
        cam._rays = None
        return cam

    # ------------------------------------------------------------------ geometry
    @property
    def render_K(self) -> tuple[np.ndarray, np.ndarray]:
        """Intrinsics of the image the renderer draws. In mirror mode it draws the virtual
        downward camera and flips the result, so the optical center and one tangential
        distortion term go to the other side (same as PoseEstimator.mirrored in
        AprilTag-Detection)."""
        K, d = self.K.copy(), self.dist.copy()
        if self.mode == "mirror":
            if self.mirror_axis == "v":
                K[1, 2], d[2] = self.height - 1 - K[1, 2], -d[2]
            else:
                K[0, 2], d[3] = self.width - 1 - K[0, 2], -d[3]
        return K, d

    @property
    def detection_K(self) -> tuple[np.ndarray, np.ndarray]:
        """Intrinsics to use for pose estimation (solvePnP) on the image AFTER flipping it back
        (mirror mode), or on the raw image (forward mode)."""
        return self.render_K

    def basis(self, yaw_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Camera axes in world coordinates for a drone heading ``yaw_deg`` (clockwise from +y):
        (right, down, forward/optical axis) of the image the renderer draws."""
        ps = math.radians(yaw_deg)
        heading = np.array([math.sin(ps), math.cos(ps), 0.0])
        right = np.array([math.cos(ps), -math.sin(ps), 0.0])
        if self.mode == "mirror":  # looking down; top of the (flipped back) image = the drone's nose
            tau = math.radians(self.mirror_tilt_deg)
            fwd = math.cos(tau) * np.array([0.0, 0.0, -1.0]) + math.sin(tau) * heading
        else:
            pitch = math.radians(self.pitch_deg)
            fwd = math.cos(pitch) * heading + np.array([0.0, 0.0, -math.sin(pitch)])
        down = np.cross(fwd, right)
        return right, down, fwd

    def rays(self) -> np.ndarray:
        """(H, W, 2) undistorted normalized coordinates of every pixel of the drawn image: the
        ray through pixel (u, v) is x*right + y*down + forward. Computed once."""
        if self._rays is None:
            K, d = self.render_K
            u, v = np.meshgrid(np.arange(self.width, dtype=np.float32), np.arange(self.height, dtype=np.float32))
            pts = np.stack([u.ravel(), v.ravel()], axis=1).reshape(-1, 1, 2)
            und = cv2.undistortPoints(pts, K, d).reshape(self.height, self.width, 2)
            self._rays = und.astype(np.float32)
        return self._rays

    def project(self, pts_cam: np.ndarray) -> np.ndarray | None:
        """Pixels of camera-frame points (N, 3), with lens distortion. None if a point is behind
        the camera or so far outside the view that the distortion model folds it back."""
        pts_cam = np.asarray(pts_cam, float)
        if (pts_cam[:, 2] <= 1e-3).any():
            return None
        K, d = self.render_K
        nx, ny = pts_cam[:, 0] / pts_cam[:, 2], pts_cam[:, 1] / pts_cam[:, 2]
        lim_x, lim_y = 1.15 * self.width / 2 / K[0, 0], 1.15 * self.height / 2 / K[1, 1]
        if (np.abs(nx) <= lim_x).all() and (np.abs(ny) <= lim_y).all():
            uv, _ = cv2.projectPoints(pts_cam.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, d)
            return uv.reshape(-1, 2)
        # outside the view: plain pinhole (distortion is only valid inside the image)
        return np.stack([K[0, 0] * nx + K[0, 2], K[1, 1] * ny + K[1, 2]], axis=1)

    def finish(self, img: np.ndarray) -> np.ndarray:
        """What the real camera delivers from a drawn image: in mirror mode, flipped."""
        if self.mode == "mirror":
            img = cv2.flip(img, 0 if self.mirror_axis == "v" else 1)
        return img
