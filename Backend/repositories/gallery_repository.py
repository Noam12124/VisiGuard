from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from database.models import GalleryEntry


class GalleryRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    def list_for_user(self, user_id: int) -> list[GalleryEntry]:
        stmt = (
            select(GalleryEntry)
            .where(GalleryEntry.user_id == user_id)
            .order_by(GalleryEntry.created_at.desc())
        )
        return list(self._db.scalars(stmt).all())

    def get_for_user(self, user_id: int, entry_id: int) -> GalleryEntry | None:
        stmt = select(GalleryEntry).where(
            GalleryEntry.user_id == user_id,
            GalleryEntry.id == entry_id,
        )
        return self._db.scalar(stmt)

    def create(
        self,
        user_id: int,
        person_name: str,
        embedding_json: str,
        image_path: str,
    ) -> GalleryEntry:
        entry = GalleryEntry(
            user_id=user_id,
            person_name=person_name,
            embedding=embedding_json,
            image_path=image_path,
        )
        self._db.add(entry)
        self._db.commit()
        self._db.refresh(entry)
        return entry

    def update(self, entry: GalleryEntry, person_name: str | None = None) -> GalleryEntry:
        if person_name is not None:
            entry.person_name = person_name
        self._db.commit()
        self._db.refresh(entry)
        return entry

    def delete(self, entry: GalleryEntry) -> None:
        self._db.delete(entry)
        self._db.commit()
