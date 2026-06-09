"""Shared application state (ML models, detector) to avoid circular imports."""

from __future__ import annotations

from typing import Any

ml_models: dict[str, Any] = {}
face_detector: Any = None
