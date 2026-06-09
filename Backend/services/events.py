from __future__ import annotations

import threading
import uuid
from collections import deque
from datetime import datetime, timezone

from config import MAX_EVENTS
from schemas import AlertItem, RecognizedItem


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class EventStore:
    def __init__(self, maxlen: int = MAX_EVENTS) -> None:
        self._lock = threading.Lock()
        self._alerts: deque[AlertItem] = deque(maxlen=maxlen)
        self._recognized: deque[RecognizedItem] = deque(maxlen=maxlen)

    def add_alert(
        self,
        *,
        person_label: str = "Unknown person",
        confidence: float | None = None,
        camera_id: str = "default",
        severity: str = "high",
    ) -> AlertItem:
        item = AlertItem(
            id=str(uuid.uuid4()),
            timestamp=_utc_now(),
            person_label=person_label,
            confidence=confidence,
            camera_id=camera_id,
            severity=severity,  # type: ignore[arg-type]
        )
        with self._lock:
            self._alerts.appendleft(item)
        return item

    def add_recognized(
        self,
        *,
        person_name: str,
        similarity_score: float,
        camera_id: str = "default",
    ) -> RecognizedItem:
        item = RecognizedItem(
            id=str(uuid.uuid4()),
            timestamp=_utc_now(),
            person_name=person_name,
            similarity_score=similarity_score,
            camera_id=camera_id,
        )
        with self._lock:
            self._recognized.appendleft(item)
        return item

    def list_alerts(self, limit: int = 50) -> list[AlertItem]:
        with self._lock:
            return list(self._alerts)[:limit]

    def list_recognized(self, limit: int = 20) -> list[RecognizedItem]:
        with self._lock:
            return list(self._recognized)[:limit]

    def seed_demo_events(self) -> None:
        """Populate sample events so the dashboard is usable before live detection."""
        if self._alerts or self._recognized:
            return
        self.add_recognized(person_name="Family Member", similarity_score=0.91)
        self.add_recognized(person_name="Family Member", similarity_score=0.87)
        self.add_alert(person_label="Unknown person at entrance", confidence=0.72)
