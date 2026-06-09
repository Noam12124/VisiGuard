from __future__ import annotations

import threading
import time
from typing import Generator

import cv2
import numpy as np

from config import (
    CAMERA_INDEX,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    JPEG_QUALITY,
    MAX_CAMERA_PROBE,
)


class CameraService:
    def __init__(self, index: int = CAMERA_INDEX) -> None:
        self._index = index
        self._cap: cv2.VideoCapture | None = None
        self._last_frame_ok = False
        self._lock = threading.RLock()

    @property
    def is_connected(self) -> bool:
        with self._lock:
            if self._cap is None:
                return False
            return self._cap.isOpened() and self._last_frame_ok

    @property
    def source_index(self) -> int:
        return self._index

    def release(self) -> None:
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            self._last_frame_ok = False

    def open(self) -> bool:
        with self._lock:
            if self._cap is not None and self._cap.isOpened():
                return True
            self._cap = cv2.VideoCapture(self._index, cv2.CAP_DSHOW)
            if not self._cap.isOpened():
                self._cap = cv2.VideoCapture(self._index)
            if self._cap.isOpened():
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
                return True
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            return False

    def set_source_index(self, index: int) -> tuple[bool, str]:
        """
        Release the current capture and switch to a new device index.
        """
        if index < 0:
            raise ValueError("Camera index must be non-negative")

        with self._lock:
            self.release()
            self._index = index
            opened = self.open()
            if opened:
                ok, _ = self._cap.read() if self._cap else (False, None)
                self._last_frame_ok = bool(ok)
                if ok:
                    return True, f"Switched to camera source {index}."
                return False, f"Opened source {index} but could not read a frame."
            return False, f"Could not open camera source {index}."

    def read_frame(self) -> np.ndarray | None:
        with self._lock:
            if not self.open():
                return None
            assert self._cap is not None
            ok, frame = self._cap.read()
            self._last_frame_ok = bool(ok)
            if not ok or frame is None:
                return None
            return frame

    @staticmethod
    def list_available_sources(max_probe: int = MAX_CAMERA_PROBE) -> list[dict]:
        """Probe camera indices and return those that can be opened."""
        sources: list[dict] = []
        for index in range(max_probe):
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(index)
            available = cap.isOpened()
            label = f"Camera {index}"
            if index == 0:
                label = "Default webcam"
            elif index == 1:
                label = "Alternative / Bluetooth bridge"
            if available:
                label = f"{label} (index {index})"
            sources.append(
                {
                    "index": index,
                    "label": label,
                    "available": available,
                }
            )
            if cap.isOpened():
                cap.release()
        return sources

    @staticmethod
    def _placeholder_frame(message: str = "Camera disconnected") -> np.ndarray:
        frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        frame[:] = (24, 24, 32)
        cv2.putText(
            frame,
            message,
            (40, FRAME_HEIGHT // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (200, 200, 210),
            2,
            cv2.LINE_AA,
        )
        return frame

    def _encode_jpeg(self, frame: np.ndarray) -> bytes:
        ok, buffer = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
        )
        if not ok:
            raise RuntimeError("Failed to encode frame")
        return buffer.tobytes()

    def generate_mjpeg(self) -> Generator[bytes, None, None]:
        """Yield multipart MJPEG chunks for StreamingResponse."""
        fail_streak = 0
        while True:
            frame = self.read_frame()
            if frame is None:
                fail_streak += 1
                label = (
                    "Camera disconnected — retrying..."
                    if fail_streak < 30
                    else "Camera unavailable"
                )
                frame = self._placeholder_frame(label)
                if fail_streak % 30 == 1:
                    with self._lock:
                        self.release()
                    time.sleep(0.5)
            else:
                fail_streak = 0

            jpg = self._encode_jpeg(frame)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            )
            time.sleep(0.04)
