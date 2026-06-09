from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from database.models import User
from database.session import get_db
from dependencies import get_current_user
from schemas_auth import (
    ActionMessageResponse,
    AddToGalleryRequest,
    UnknownActionRequest,
)
from services.unknown_service import UnknownService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/unknowns", tags=["unknowns"])


def _get_model():
    from app_state import ml_models

    model = ml_models.get("prediction_model")
    if model is None:
        raise HTTPException(status_code=400, detail="Embedding model is not loaded.")
    return model


@router.post("/add-to-gallery")
def add_unknown_to_gallery(
    body: AddToGalleryRequest,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
):
    model = _get_model()
    try:
        result = UnknownService(db).add_to_gallery(
            user=user,
            model=model,
            unknown_id=body.unknown_id,
            person_name=body.person_name,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return result


@router.post("/trigger-alarm", response_model=ActionMessageResponse)
def trigger_alarm(
    body: UnknownActionRequest,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
):
    try:
        result = UnknownService(db).trigger_alarm(user, body.unknown_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ActionMessageResponse(message=result["message"])


@router.post("/ignore", response_model=ActionMessageResponse)
def ignore_unknown(
    body: UnknownActionRequest,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
):
    try:
        result = UnknownService(db).ignore(user, body.unknown_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ActionMessageResponse(message=result["message"])
