from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from config import KNOWN_USERS_PATH, MATCH_THRESHOLD
from services.embeddings import compute_cosine_similarity


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class KnownUsersGallery:
    """
    Persistent known-users store.

    File layout (Backend/data/known_users.json):
    {
      "version": 1,
      "users": [
        {
          "id": "<uuid>",
          "name": "Noam",
          "embedding": [0.012, -0.034, ...],  // flat float32 vector
          "created_at": "2026-06-04T12:00:00+00:00"
        }
      ]
    }
    """

    def __init__(self, path: Path = KNOWN_USERS_PATH) -> None:
        self._path = path
        self._lock = threading.RLock()
        self._data: dict[str, Any] = {"version": 1, "users": []}
        self._load()

    def _load(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            if not self._path.exists():
                self._save_unlocked()
                return
            try:
                raw = json.loads(self._path.read_text(encoding="utf-8"))
                if isinstance(raw, dict) and isinstance(raw.get("users"), list):
                    self._data = raw
                else:
                    self._data = {"version": 1, "users": []}
            except (json.JSONDecodeError, OSError):
                self._data = {"version": 1, "users": []}

    def _save_unlocked(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(self._data, indent=2),
            encoding="utf-8",
        )

    def save(self) -> None:
        with self._lock:
            self._save_unlocked()

    def list_users(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._data.get("users", []))

    def add_user(self, name: str, embedding: np.ndarray) -> dict[str, Any]:
        name = name.strip()
        if not name:
            raise ValueError("Name is required.")

        vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vec.size == 0:
            raise ValueError("Embedding vector is empty.")

        record = {
            "id": str(uuid.uuid4()),
            "name": name,
            "embedding": vec.tolist(),
            "created_at": _utc_now_iso(),
        }
        with self._lock:
            self._data.setdefault("users", []).append(record)
            self._save_unlocked()
        return record

    def identify(
        self, probe_embedding: np.ndarray
    ) -> tuple[str, str | None, float | None]:
        """
        Returns (status, name, best_score).
        status is 'authorized' or 'unknown'.
        """
        probe = np.asarray(probe_embedding, dtype=np.float32).reshape(-1)
        with self._lock:
            users = list(self._data.get("users", []))

        if not users:
            return "unknown", None, None

        best_name: str | None = None
        best_score = -1.0

        for user in users:
            stored = np.asarray(user["embedding"], dtype=np.float32).reshape(-1)
            score = compute_cosine_similarity(probe, stored)
            if score > best_score:
                best_score = score
                best_name = user["name"]

        if best_score >= MATCH_THRESHOLD and best_name:
            return "authorized", best_name, float(best_score)

        return "unknown", None, float(best_score) if best_score >= 0 else None
