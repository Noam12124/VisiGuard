from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path

import cv2
import numpy as np
from sqlalchemy.orm import Session

from config import GALLERY_UPLOADS_DIR, SIMILARITY_THRESHOLD
from database.models import GalleryEntry, User
from repositories.gallery_repository import GalleryRepository
from services.detector import FaceDetector
from services.embeddings import compute_cosine_similarity, get_embedding

logger = logging.getLogger(__name__)


class GalleryService:
    def __init__(self, db: Session) -> None:
        self._repo = GalleryRepository(db)

    @staticmethod
    def _ensure_upload_dir(user_id: int) -> Path:
        path = GALLERY_UPLOADS_DIR / str(user_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _embedding_to_json(embedding: np.ndarray) -> str:
        vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vec.size == 0:
            raise ValueError("Cannot serialize empty embedding.")
        return json.dumps(vec.tolist())

    @staticmethod
    def _json_to_embedding(raw: str) -> np.ndarray:
        data = json.loads(raw)
        if not data:
            raise ValueError("Stored embedding is empty.")
        vec = np.asarray(data, dtype=np.float32).reshape(-1)
        return vec

    def list_entries(self, user_id: int) -> list[GalleryEntry]:
        return self._repo.list_for_user(user_id)

    def get_entry(self, user_id: int, entry_id: int) -> GalleryEntry | None:
        return self._repo.get_for_user(user_id, entry_id)

    def add_from_bytes(
        self,
        user: User,
        model,
        person_name: str,
        image_bytes: bytes,
        *,
        detector: FaceDetector | None = None,
        use_face_detection: bool = True,
        save_image: bool = True,
    ) -> GalleryEntry:
        person_name = person_name.strip()
        if not person_name:
            raise ValueError("Person name is required.")

        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError("Could not decode gallery image.")

        face_bytes = image_bytes
        if use_face_detection and detector is not None and detector.is_available:
            boxes = detector.detect_faces(image_bgr)
            if not boxes:
                raise ValueError("No face detected in gallery image.")
            largest = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            face_bytes = detector.crop_and_align(image_bgr, largest)
            logger.info(
                "Gallery enroll: YOLO crop for user=%s name=%r box=%s",
                user.id,
                person_name,
                largest,
            )
        elif use_face_detection:
            logger.warning(
                "Gallery enroll: YOLO unavailable — embedding full image for user=%s",
                user.id,
            )
        else:
            logger.info(
                "Gallery enroll: pre-cropped image for user=%s name=%r",
                user.id,
                person_name,
            )

        embedding = get_embedding(model, face_bytes)
        image_path = ""

        if save_image:
            upload_dir = self._ensure_upload_dir(user.id)
            filename = f"{uuid.uuid4().hex}.jpg"
            file_path = upload_dir / filename
            file_path.write_bytes(face_bytes)
            image_path = f"gallery/{user.id}/{filename}"

        entry = self._repo.create(
            user_id=user.id,
            person_name=person_name,
            embedding_json=self._embedding_to_json(embedding),
            image_path=image_path,
        )
        logger.info(
            "Gallery entry created id=%s user=%s name=%r dims=%d",
            entry.id,
            user.id,
            person_name,
            embedding.size,
        )
        return entry

    def add_from_embedding(
        self,
        user: User,
        person_name: str,
        embedding: list[float],
        image_path: str = "",
    ) -> GalleryEntry:
        person_name = person_name.strip()
        if not person_name:
            raise ValueError("Person name is required.")
        vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vec.size == 0:
            raise ValueError("Embedding vector is empty.")

        return self._repo.create(
            user_id=user.id,
            person_name=person_name,
            embedding_json=self._embedding_to_json(vec),
            image_path=image_path,
        )

    def update_entry(
        self,
        user_id: int,
        entry_id: int,
        person_name: str,
    ) -> GalleryEntry:
        entry = self._repo.get_for_user(user_id, entry_id)
        if entry is None:
            raise ValueError("Gallery entry not found.")
        return self._repo.update(entry, person_name=person_name.strip())

    def delete_entry(self, user_id: int, entry_id: int) -> None:
        entry = self._repo.get_for_user(user_id, entry_id)
        if entry is None:
            raise ValueError("Gallery entry not found.")
        self._repo.delete(entry)

    def identify(
        self,
        user_id: int,
        probe_embedding: np.ndarray,
        threshold: float = SIMILARITY_THRESHOLD,
    ) -> tuple[str, str | None, float]:
        entries = self._repo.list_for_user(user_id)
        probe = np.asarray(probe_embedding, dtype=np.float32).reshape(-1)

        if probe.size == 0:
            logger.error("[identify] user=%s probe embedding is empty", user_id)
            return "unknown", None, 0.0

        if not entries:
            logger.debug("[identify] user=%s gallery empty -> unknown", user_id)
            return "unknown", None, 0.0

        best_name: str | None = None
        best_score = -1.0
        scores: list[tuple[str, float]] = []

        for entry in entries:
            try:
                stored = self._json_to_embedding(entry.embedding)
            except (ValueError, json.JSONDecodeError) as exc:
                logger.error(
                    "[identify] user=%s entry=%s corrupt embedding: %s",
                    user_id,
                    entry.id,
                    exc,
                )
                continue

            if stored.shape != probe.shape:
                logger.error(
                    "[identify] user=%s entry=%s dim mismatch probe=%d stored=%d",
                    user_id,
                    entry.id,
                    probe.size,
                    stored.size,
                )
                continue

            score = compute_cosine_similarity(probe, stored)
            scores.append((entry.person_name, score))
            if score > best_score:
                best_score = score
                best_name = entry.person_name

        scores.sort(key=lambda item: item[1], reverse=True)
        logger.debug(
            "[identify] user=%s threshold=%.3f scores=%s",
            user_id,
            threshold,
            [(n, round(s, 4)) for n, s in scores],
        )

        if best_score >= threshold and best_name:
            return "known", best_name, float(best_score)

        confidence = float(best_score) if best_score >= 0 else 0.0
        return "unknown", None, confidence
