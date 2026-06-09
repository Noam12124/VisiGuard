from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class RootResponse(BaseModel):
    message: str


class PredictResponse(BaseModel):
    status: str


class CompareFacesResponse(BaseModel):
    is_match: bool
    similarity_score: float = Field(ge=-1.0, le=1.0)
    threshold: float
    match_percent: float = Field(ge=-100.0, le=100.0)
    identity_label: Literal["Match", "No Match"]


class CameraSourceItem(BaseModel):
    index: int
    label: str
    available: bool


class CamerasListResponse(BaseModel):
    sources: list[CameraSourceItem]
    active_index: int


class SetCameraRequest(BaseModel):
    source_index: int = Field(ge=0)


class SetCameraResponse(BaseModel):
    success: bool
    source_index: int
    connected: bool
    message: str


class VideoFeedStatusResponse(BaseModel):
    connected: bool
    source_index: int


class AlertItem(BaseModel):
    id: str
    timestamp: datetime
    person_label: str
    confidence: float | None = None
    camera_id: str = "default"
    thumbnail_url: str | None = None
    severity: Literal["high", "medium", "low"] = "high"


class AlertsResponse(BaseModel):
    items: list[AlertItem]


class RecognizedItem(BaseModel):
    id: str
    timestamp: datetime
    person_name: str
    similarity_score: float
    camera_id: str = "default"
    status: Literal["authorized"] = "authorized"


class RecognizedResponse(BaseModel):
    items: list[RecognizedItem]


class AnalyzeFrameResponse(BaseModel):
    status: Literal["authorized", "unknown"]
    name: str | None = None
    similarity_score: float | None = None
    threshold: float
    embedding: list[float] | None = None


class AddKnownUserRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    embedding: list[float] = Field(min_length=1)


class AddKnownUserResponse(BaseModel):
    id: str
    name: str
    message: str


class ReportIntrusionRequest(BaseModel):
    similarity_score: float | None = None


class ReportIntrusionResponse(BaseModel):
    logged: bool
    message: str


class KnownUsersListResponse(BaseModel):
    count: int
    users: list[dict]
