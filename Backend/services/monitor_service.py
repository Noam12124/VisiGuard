from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from sqlalchemy.orm import Session

from config import (
    MONITOR_FRAME_MAX_WIDTH,
    RECOGNITION_LOG_COOLDOWN_SEC,
    SIMILARITY_THRESHOLD,
    UNKNOWN_LOG_COOLDOWN_SEC,
    UNKNOWN_UPLOADS_DIR,
)
from database.models import User
from repositories.recognition_repository import RecognitionRepository
from repositories.unknown_repository import UnknownRepository
from services.detector import FaceDetector
from services.embeddings import get_embedding
from services.gallery_service import GalleryService

logger = logging.getLogger(__name__)

_last_recognition_at: dict[tuple[int, str], float] = {}
_last_unknown_at: dict[int, float] = {}
_last_unknown_payload: dict[int, dict] = {}


class MonitorService:
    def __init__(self, db: Session, detector: FaceDetector) -> None:
        self._gallery = GalleryService(db)
        self._recognitions = RecognitionRepository(db)
        self._unknowns = UnknownRepository(db)
        self._detector = detector

    @staticmethod
    def _decode_frame(image_bytes: bytes) -> np.ndarray:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode frame image.")
        return img

    @staticmethod
    def _downscale(image_bgr: np.ndarray, max_width: int) -> np.ndarray:
        h, w = image_bgr.shape[:2]
        if w <= max_width:
            return image_bgr
        scale = max_width / w
        new_h = max(1, int(h * scale))
        return cv2.resize(image_bgr, (max_width, new_h), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _snapshot_path(user_id: int, filename: str) -> str:
        return f"unknowns/{user_id}/{filename}"

    @staticmethod
    def _should_log_recognition(user_id: int, person_name: str) -> bool:
        key = (user_id, person_name)
        now = time.monotonic()
        last = _last_recognition_at.get(key, 0.0)
        if now - last < RECOGNITION_LOG_COOLDOWN_SEC:
            return False
        _last_recognition_at[key] = now
        return True

    @staticmethod
    def _should_log_unknown(user_id: int) -> bool:
        now = time.monotonic()
        last = _last_unknown_at.get(user_id, 0.0)
        if now - last < UNKNOWN_LOG_COOLDOWN_SEC:
            return False
        _last_unknown_at[user_id] = now
        return True

    def process_frame(self, user: User, model, image_bytes: bytes) -> dict:
        try:
            image_bgr = self._downscale(
                self._decode_frame(image_bytes),
                MONITOR_FRAME_MAX_WIDTH,
            )
            h, w = image_bgr.shape[:2]
            logger.debug(
                "[monitor] frame user=%s bytes=%d shape=%dx%d",
                user.id,
                len(image_bytes),
                w,
                h,
            )
        except ValueError as exc:
            logger.error("[monitor] frame decode failed user=%s: %s", user.id, exc)
            raise

        boxes = self._detector.detect_faces(image_bgr)
        logger.debug(
            "[monitor] detection user=%s yolo=%s faces=%d",
            user.id,
            self._detector.is_available,
            len(boxes),
        )

        if not boxes:
            return {
                "detections": [],
                "threshold": SIMILARITY_THRESHOLD,
                "faces_detected": 0,
            }

        detections: list[dict] = []

        for idx, box in enumerate(boxes):
            try:
                face_bytes = self._detector.crop_and_align(image_bgr, box)
            except Exception as exc:
                logger.warning("[monitor] crop failed user=%s face=%d: %s", user.id, idx, exc)
                continue

            try:
                embedding = get_embedding(model, face_bytes)
            except Exception as exc:
                logger.warning("[monitor] embed failed user=%s face=%d: %s", user.id, idx, exc)
                continue

            status, person_name, confidence = self._gallery.identify(
                user.id, embedding, threshold=SIMILARITY_THRESHOLD
            )

            if status == "known" and person_name:
                if self._should_log_recognition(user.id, person_name):
                    self._recognitions.create(
                        user_id=user.id,
                        person_name=person_name,
                        confidence=confidence,
                    )
                detections.append(
                    {
                        "status": "known",
                        "person_name": person_name,
                        "confidence": confidence,
                    }
                )
            else:
                if self._should_log_unknown(user.id):
                    filename = f"{uuid.uuid4().hex}.jpg"
                    abs_path = UNKNOWN_UPLOADS_DIR / str(user.id) / filename
                    self._detector.save_snapshot(image_bgr, box, abs_path)
                    rel_path = self._snapshot_path(user.id, filename)
                    unknown = self._unknowns.create(
                        user_id=user.id,
                        snapshot_path=rel_path,
                        confidence=confidence,
                    )
                    payload = {
                        "status": "unknown",
                        "confidence": confidence,
                        "snapshot_path": rel_path,
                        "unknown_id": unknown.id,
                    }
                    _last_unknown_payload[user.id] = payload
                    detections.append(payload)
                elif user.id in _last_unknown_payload:
                    detections.append(_last_unknown_payload[user.id])
                else:
                    detections.append(
                        {
                            "status": "unknown",
                            "confidence": confidence,
                            "snapshot_path": "",
                            "unknown_id": 0,
                        }
                    )

        return {
            "detections": detections,
            "threshold": SIMILARITY_THRESHOLD,
            "faces_detected": len(boxes),
        }
