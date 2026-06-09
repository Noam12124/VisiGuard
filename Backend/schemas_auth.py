from datetime import datetime

from pydantic import BaseModel, EmailStr, Field


class SignUpRequest(BaseModel):
    username: str = Field(min_length=3, max_length=80)
    email: EmailStr
    password: str = Field(min_length=6, max_length=128)


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=80)
    password: str = Field(min_length=1, max_length=128)


class UserPublic(BaseModel):
    id: int
    username: str
    email: str
    created_at: datetime

    model_config = {"from_attributes": True}


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserPublic


class GalleryPersonResponse(BaseModel):
    id: int
    person_name: str
    image_path: str
    created_at: datetime

    model_config = {"from_attributes": True}


class GalleryListResponse(BaseModel):
    count: int
    items: list[GalleryPersonResponse]


class GalleryCreateRequest(BaseModel):
    person_name: str = Field(min_length=1, max_length=120)


class GalleryUpdateRequest(BaseModel):
    person_name: str = Field(min_length=1, max_length=120)


class KnownDetection(BaseModel):
    status: str = "known"
    person_name: str
    confidence: float


class UnknownDetection(BaseModel):
    status: str = "unknown"
    confidence: float
    snapshot_path: str
    unknown_id: int


class MonitorFrameResponse(BaseModel):
    detections: list[KnownDetection | UnknownDetection | dict]
    threshold: float
    faces_detected: int


class RecognitionEventResponse(BaseModel):
    id: int
    person_name: str
    confidence: float
    snapshot_path: str | None
    timestamp: datetime

    model_config = {"from_attributes": True}


class RecognitionListResponse(BaseModel):
    count: int
    items: list[RecognitionEventResponse]


class UnknownPersonResponse(BaseModel):
    id: int
    snapshot_path: str
    confidence: float
    timestamp: datetime
    action_taken: str | None

    model_config = {"from_attributes": True}


class UnknownListResponse(BaseModel):
    count: int
    items: list[UnknownPersonResponse]


class SecurityEventResponse(BaseModel):
    id: int
    event_type: str
    description: str
    timestamp: datetime

    model_config = {"from_attributes": True}


class SecurityEventListResponse(BaseModel):
    count: int
    items: list[SecurityEventResponse]


class UnknownActionRequest(BaseModel):
    unknown_id: int


class AddToGalleryRequest(BaseModel):
    unknown_id: int
    person_name: str = Field(min_length=1, max_length=120)


class ActionMessageResponse(BaseModel):
    message: str
