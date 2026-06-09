from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from database.models import SecurityEvent


class SecurityRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    def create(
        self,
        user_id: int,
        event_type: str,
        description: str,
    ) -> SecurityEvent:
        event = SecurityEvent(
            user_id=user_id,
            event_type=event_type,
            description=description,
        )
        self._db.add(event)
        self._db.commit()
        self._db.refresh(event)
        return event

    def list_for_user(self, user_id: int, limit: int = 100) -> list[SecurityEvent]:
        stmt = (
            select(SecurityEvent)
            .where(SecurityEvent.user_id == user_id)
            .order_by(SecurityEvent.timestamp.desc())
            .limit(limit)
        )
        return list(self._db.scalars(stmt).all())
