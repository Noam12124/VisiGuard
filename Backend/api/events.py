from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from config import DEFAULT_LIST_LIMIT
from database.models import User
from database.session import get_db
from dependencies import get_current_user
from repositories.recognition_repository import RecognitionRepository
from repositories.security_repository import SecurityRepository
from repositories.unknown_repository import UnknownRepository
from schemas_auth import (
    RecognitionEventResponse,
    RecognitionListResponse,
    SecurityEventListResponse,
    SecurityEventResponse,
    UnknownListResponse,
    UnknownPersonResponse,
)

router = APIRouter(tags=["events"])


@router.get("/recognitions", response_model=RecognitionListResponse)
def list_recognitions(
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
    limit: int = DEFAULT_LIST_LIMIT,
):
    items = RecognitionRepository(db).list_for_user(user.id, limit=limit)
    response_items = [RecognitionEventResponse.model_validate(i) for i in items]
    return RecognitionListResponse(count=len(response_items), items=response_items)


@router.get("/unknowns", response_model=UnknownListResponse)
def list_unknowns(
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
    limit: int = DEFAULT_LIST_LIMIT,
):
    items = UnknownRepository(db).list_for_user(user.id, limit=limit)
    response_items = [UnknownPersonResponse.model_validate(i) for i in items]
    return UnknownListResponse(count=len(response_items), items=response_items)


@router.get("/security-events", response_model=SecurityEventListResponse)
def list_security_events(
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
    limit: int = DEFAULT_LIST_LIMIT,
):
    items = SecurityRepository(db).list_for_user(user.id, limit=limit)
    response_items = [SecurityEventResponse.model_validate(i) for i in items]
    return SecurityEventListResponse(count=len(response_items), items=response_items)
