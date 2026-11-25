"""Where things are relative to the drone, from the camera image.

Two mountings (same as tellosim.Camera):
- forward camera, tilted ``pitch_deg`` below the horizon (11° on the lab Tellos);
- downward mirror: the image must be flipped back first (``unmirror``), and the view is tilted
  ``view_forward_deg`` from straight down (+ = towards the nose; the lab clip: -10.3°).

Drone frame: (forward, right, up) in cm, origin at the camera.
"""
from __future__ import annotations

import math

import cv2
import numpy as np


def unmirror(img: np.ndarray, axis: str = "v") -> np.ndarray:
    """Undo the mirror flip (a mirrored marker does not decode)."""
    return cv2.flip(img, 0 if axis == "v" else 1)


def mirrored_intrinsics(K: np.ndarray, dist: np.ndarray, size=(960, 720), axis: str = "v"):
    """Intrinsics of the flipped-back image from the camera's own calibration."""
    K, d = np.array(K, float).copy(), np.array(dist, float).ravel().copy()
    w, h = size
    if axis == "v":
        K[1, 2], d[2] = h - 1 - K[1, 2], -d[2]
    else:
        K[0, 2], d[3] = w - 1 - K[0, 2], -d[3]
    return K, d


def camera_axes(mode: str = "forward", pitch_deg: float = 11.0, view_forward_deg: float = -10.3):
    """(right, down, optical axis) of the image, in the drone frame (forward, right, up)."""
    right = np.array([0.0, 1.0, 0.0])
    if mode == "mirror":
        tau = math.radians(view_forward_deg)
        f = math.cos(tau) * np.array([0.0, 0.0, -1.0]) + math.sin(tau) * np.array([1.0, 0.0, 0.0])
    else:
        p = math.radians(pitch_deg)
        f = math.cos(p) * np.array([1.0, 0.0, 0.0]) + math.sin(p) * np.array([0.0, 0.0, -1.0])
    # (forward, right, up) is a left-handed frame: image-down = right x optical-axis here
    # (mirror looking straight down: image-down points backwards; forward camera: down)
    down = np.cross(right, f)
    return right, down, f


def camera_to_drone(p_cam, mode: str = "forward", pitch_deg: float = 11.0, view_forward_deg: float = -10.3):
    """A point in camera coordinates (x right, y down, z along the optical axis) -> drone frame
    (forward, right, up)."""
    r, d, f = camera_axes(mode, pitch_deg, view_forward_deg)
    x, y, z = p_cam
    return x * r + y * d + z * f


def floor_point_from_pixel(u: float, v: float, K, dist, height_cm: float, mode: str = "mirror",
                           pitch_deg: float = 11.0, view_forward_deg: float = -10.3):
    """(forward, right) in cm of the floor point seen at pixel (u, v), with the camera
    ``height_cm`` above the floor (tof - 10). None if that pixel looks above the horizon."""
    xn, yn = cv2.undistortPoints(np.float32([[[u, v]]]), np.asarray(K, float), np.asarray(dist, float)).ravel()
    ray = camera_to_drone((xn, yn, 1.0), mode, pitch_deg, view_forward_deg)
    if ray[2] >= -1e-6:
        return None
    t = -height_cm / ray[2]
    return float(t * ray[0]), float(t * ray[1])


def detect_markers(gray: np.ndarray, dictionary: str = "DICT_APRILTAG_25h9", ids=None):
    """[(id, corners (4, 2))] of the markers found. ids: keep only these (rejects false
    positives: on the lab floor a stray ID 29 appeared once)."""
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    corners, found, _ = cv2.aruco.ArucoDetector(d, params).detectMarkers(gray)
    out = []
    if found is not None:
        for c, i in zip(corners, found.ravel()):
            if ids is None or int(i) in ids:
                out.append((int(i), c.reshape(4, 2)))
    return out


def marker_position(corners, size_cm: float, K, dist, mode: str = "forward", pitch_deg: float = 11.0,
                    view_forward_deg: float = -10.3):
    """Center of a marker in the drone frame (forward, right, up), cm, with solvePnP."""
    s = size_cm / 2
    obj = np.float32([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]])
    ok, rvec, tvec = cv2.solvePnP(obj, np.asarray(corners, np.float32), np.asarray(K, float), np.asarray(dist, float),
                                  flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    return camera_to_drone(tvec.ravel(), mode, pitch_deg, view_forward_deg)
