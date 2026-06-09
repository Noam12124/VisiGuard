from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy.orm import Session

from database.models import User
from repositories.gallery_repository import GalleryRepository
from repositories.security_repository import SecurityRepository
from repositories.unknown_repository import UnknownRepository
from services.gallery_service import GalleryService

logger = logging.getLogger(__name__)


class UnknownService:
    def __init__(self, db: Session) -> None:
        self._unknowns = UnknownRepository(db)
        self._security = SecurityRepository(db)
        self._gallery = GalleryService(db)
        self._gallery_repo = GalleryRepository(db)

    def add_to_gallery(
        self,
        user: User,
        model,
        unknown_id: int,
        person_name: str,
    ) -> dict:
        record = self._unknowns.get_for_user(user.id, unknown_id)
        if record is None:
            raise ValueError("Unknown person record not found.")

        from config import UPLOADS_DIR

        snapshot_path = UPLOADS_DIR / record.snapshot_path
        if not snapshot_path.exists():
            raise ValueError(f"Snapshot file not found: {record.snapshot_path}")

        from app_state import face_detector

        image_bytes = snapshot_path.read_bytes()
        entry = self._gallery.add_from_bytes(
            user=user,
            model=model,
            person_name=person_name,
            image_bytes=image_bytes,
            detector=face_detector,
            use_face_detection=False,
            save_image=True,
        )
        self._unknowns.update_action(record, "added_to_gallery")
        logger.info("Unknown %s added to gallery as '%s'", unknown_id, person_name)
        return {
            "gallery_id": entry.id,
            "person_name": entry.person_name,
            "message": f"'{person_name}' added to gallery.",
        }

    def trigger_alarm(self, user: User, unknown_id: int) -> dict:
        record = self._unknowns.get_for_user(user.id, unknown_id)
        if record is None:
            raise ValueError("Unknown person record not found.")

        self._unknowns.update_action(record, "alarm_triggered")
        event = self._security.create(
            user_id=user.id,
            event_type="alarm",
            description=f"Manual alarm triggered for unknown person #{unknown_id} (confidence={record.confidence:.2f})",
        )
        logger.warning("Alarm triggered by user %s for unknown %s", user.username, unknown_id)
        return {
            "logged": True,
            "security_event_id": event.id,
            "message": "Security alarm logged.",
        }

    def ignore(self, user: User, unknown_id: int) -> dict:
        record = self._unknowns.get_for_user(user.id, unknown_id)
        if record is None:
            raise ValueError("Unknown person record not found.")

        self._unknowns.update_action(record, "ignored")
        logger.info("Unknown %s ignored by user %s", unknown_id, user.username)
        return {"ignored": True, "message": "Event ignored."}
