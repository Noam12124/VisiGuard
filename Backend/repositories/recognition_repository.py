from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from database.models import RecognitionEvent


class RecognitionRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    def create(
        self,
        user_id: int,
        person_name: str,
        confidence: float,
        snapshot_path: str | None = None,
    ) -> RecognitionEvent:
        event = RecognitionEvent(
            user_id=user_id,
            person_name=person_name,
            confidence=confidence,
            snapshot_path=snapshot_path,
        )
        self._db.add(event)
        self._db.commit()
        self._db.refresh(event)
        return event

    def list_for_user(self, user_id: int, limit: int = 100) -> list[RecognitionEvent]:
        stmt = (
            select(RecognitionEvent)
            .where(RecognitionEvent.user_id == user_id)
            .order_by(RecognitionEvent.timestamp.desc())
            .limit(limit)
        )
        return list(self._db.scalars(stmt).all())
