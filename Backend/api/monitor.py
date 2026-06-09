from __future__ import annotations

import asyncio
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from database.models import User
from database.session import get_db
from dependencies import get_current_user
from schemas_auth import MonitorFrameResponse
from services.monitor_service import MonitorService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/monitor", tags=["monitor"])


def _get_model():
    from app_state import face_detector, ml_models

    model = ml_models.get("prediction_model")
    if model is None:
        raise HTTPException(
            status_code=400,
            detail="Embedding model is not loaded.",
        )
    return model, face_detector


@router.post("/frame", response_model=MonitorFrameResponse)
async def analyze_monitor_frame(
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
    frame: UploadFile = File(...),
):
    model, detector = _get_model()
    image_bytes = await frame.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Frame image is required.")

    try:
        service = MonitorService(db, detector)
        result = await asyncio.to_thread(
            service.process_frame,
            user,
            model,
            image_bytes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Monitor frame processing failed")
        raise HTTPException(status_code=500, detail=f"Frame processing failed: {exc}") from exc

    logger.debug(
        "POST /monitor/frame user=%s faces=%d statuses=%s",
        user.id,
        result.get("faces_detected"),
        [d.get("status") for d in result.get("detections", [])],
    )
    return MonitorFrameResponse(**result)
