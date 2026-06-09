from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from database.models import UnknownPerson


class UnknownRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    def create(
        self,
        user_id: int,
        snapshot_path: str,
        confidence: float,
    ) -> UnknownPerson:
        record = UnknownPerson(
            user_id=user_id,
            snapshot_path=snapshot_path,
            confidence=confidence,
        )
        self._db.add(record)
        self._db.commit()
        self._db.refresh(record)
        return record

    def get_for_user(self, user_id: int, unknown_id: int) -> UnknownPerson | None:
        stmt = select(UnknownPerson).where(
            UnknownPerson.user_id == user_id,
            UnknownPerson.id == unknown_id,
        )
        return self._db.scalar(stmt)

    def list_for_user(self, user_id: int, limit: int = 100) -> list[UnknownPerson]:
        stmt = (
            select(UnknownPerson)
            .where(UnknownPerson.user_id == user_id)
            .order_by(UnknownPerson.timestamp.desc())
            .limit(limit)
        )
        return list(self._db.scalars(stmt).all())

    def update_action(self, record: UnknownPerson, action_taken: str) -> UnknownPerson:
        record.action_taken = action_taken
        self._db.commit()
        self._db.refresh(record)
        return record
