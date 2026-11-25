"""A window with the video, status lines and a STOP button (keyboard and mouse).

Keys, with the window focused: SPACE / ENTER / l / ESC = land now; k = kill the motors (the
drone FALLS: only if it is heading at a person); q = quit (lands if flying). A click on the red
bar also lands. Somebody should always have a hand on the keyboard when flying.
"""
from __future__ import annotations

import cv2
import numpy as np


class StopWindow:
    BAR_H = 70

    def __init__(self, title: str = "tello") -> None:
        self.title = title
        self._clicked = False
        self.size = (960, 720)
        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(title, self._on_mouse)

    def _on_mouse(self, event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and y >= self.size[1] - self.BAR_H:
            self._clicked = True

    def show(self, frame_bgr: np.ndarray | None, lines: list[str] = (), flying: bool = True) -> str | None:
        """Draw and read the keyboard. Returns "land", "kill", "quit" or None."""
        img = frame_bgr.copy() if frame_bgr is not None else np.zeros((720, 960, 3), np.uint8)
        h, w = img.shape[:2]
        self.size = (w, h)
        for i, text in enumerate(lines):
            y = 30 + 28 * i
            cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        color = (0, 0, 200) if flying else (60, 60, 60)
        cv2.rectangle(img, (0, h - self.BAR_H), (w, h), color, -1)
        label = "STOP - click here or SPACE" if flying else "k = kill motors (emergency only)"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.putText(img, label, ((w - tw) // 2, h - (self.BAR_H - th) // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(self.title, img)
        key = cv2.waitKey(1) & 0xFF
        if self._clicked:
            self._clicked = False
            return "land"
        if key in (32, 13, ord("l"), 27):
            return "land"
        if key == ord("k"):
            return "kill"
        if key == ord("q"):
            return "quit"
        return None

    def close(self) -> None:
        cv2.destroyWindow(self.title)
