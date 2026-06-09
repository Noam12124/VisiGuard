from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.base import Base


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)

    gallery_entries: Mapped[list[GalleryEntry]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    recognition_events: Mapped[list[RecognitionEvent]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    unknown_persons: Mapped[list[UnknownPerson]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    security_events: Mapped[list[SecurityEvent]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class GalleryEntry(Base):
    __tablename__ = "gallery"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    person_name: Mapped[str] = mapped_column(String(120))
    embedding: Mapped[str] = mapped_column(Text)
    image_path: Mapped[str] = mapped_column(String(512))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)

    user: Mapped[User] = relationship(back_populates="gallery_entries")


class RecognitionEvent(Base):
    __tablename__ = "recognition_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    person_name: Mapped[str] = mapped_column(String(120))
    confidence: Mapped[float] = mapped_column(Float)
    snapshot_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)

    user: Mapped[User] = relationship(back_populates="recognition_events")


class UnknownPerson(Base):
    __tablename__ = "unknown_persons"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    snapshot_path: Mapped[str] = mapped_column(String(512))
    confidence: Mapped[float] = mapped_column(Float)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)
    action_taken: Mapped[str | None] = mapped_column(String(64), nullable=True)

    user: Mapped[User] = relationship(back_populates="unknown_persons")


class SecurityEvent(Base):
    __tablename__ = "security_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    event_type: Mapped[str] = mapped_column(String(64))
    description: Mapped[str] = mapped_column(Text)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)

    user: Mapped[User] = relationship(back_populates="security_events")
