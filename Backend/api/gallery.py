from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from database.models import User
from database.session import get_db
from dependencies import get_current_user
from schemas_auth import (
    GalleryCreateRequest,
    GalleryListResponse,
    GalleryPersonResponse,
    GalleryUpdateRequest,
)
from services.gallery_service import GalleryService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/gallery", tags=["gallery"])


def _get_model_and_detector():
    from app_state import face_detector, ml_models

    model = ml_models.get("prediction_model")
    if model is None:
        raise HTTPException(
            status_code=400,
            detail="Embedding model is not loaded.",
        )
    return model, face_detector


@router.get("", response_model=GalleryListResponse)
def list_gallery(
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
):
    entries = GalleryService(db).list_entries(user.id)
    items = [GalleryPersonResponse.model_validate(e) for e in entries]
    return GalleryListResponse(count=len(items), items=items)


@router.post("", response_model=GalleryPersonResponse)
async def add_gallery_person(
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
    person_name: Annotated[str, Form()],
    image: UploadFile = File(...),
):
    model, detector = _get_model_and_detector()
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Image file is required.")

    try:
        entry = GalleryService(db).add_from_bytes(
            user=user,
            model=model,
            person_name=person_name,
            image_bytes=image_bytes,
            detector=detector,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return GalleryPersonResponse.model_validate(entry)


@router.post("/enroll", response_model=GalleryPersonResponse)
async def enroll_gallery_person(
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
    body: GalleryCreateRequest,
    image: UploadFile = File(...),
):
    """JSON body variant with multipart image upload."""
    model, detector = _get_model_and_detector()
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Image file is required.")

    try:
        entry = GalleryService(db).add_from_bytes(
            user=user,
            model=model,
            person_name=body.person_name,
            image_bytes=image_bytes,
            detector=detector,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return GalleryPersonResponse.model_validate(entry)


@router.put("/{entry_id}", response_model=GalleryPersonResponse)
def update_gallery_person(
    entry_id: int,
    body: GalleryUpdateRequest,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
):
    try:
        entry = GalleryService(db).update_entry(user.id, entry_id, body.person_name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return GalleryPersonResponse.model_validate(entry)


@router.delete("/{entry_id}")
def delete_gallery_person(
    entry_id: int,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
):
    try:
        GalleryService(db).delete_entry(user.id, entry_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"deleted": True, "message": "Gallery person removed."}
