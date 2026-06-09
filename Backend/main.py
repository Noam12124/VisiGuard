from contextlib import asynccontextmanager
import logging
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

import app_state
from api import auth_router, events_router, gallery_router, monitor_router, unknowns_router
from config import (
    CORS_ORIGINS,
    DATA_DIR,
    DEFAULT_ALERTS_LIMIT,
    DEFAULT_RECOGNIZED_LIMIT,
    MODEL_PATH,
    UPLOADS_DIR,
)
from database import init_db
from schemas import (
    AddKnownUserRequest,
    AddKnownUserResponse,
    AlertsResponse,
    AnalyzeFrameResponse,
    CamerasListResponse,
    CameraSourceItem,
    CompareFacesResponse,
    KnownUsersListResponse,
    PredictResponse,
    RecognizedResponse,
    ReportIntrusionRequest,
    ReportIntrusionResponse,
    RootResponse,
    SetCameraRequest,
    SetCameraResponse,
    VideoFeedStatusResponse,
)
from services.analyze import analyze_frame
from services.camera import CameraService
from services.compare import compare_two_faces
from services.detector import FaceDetector
from services.events import EventStore
from services.gallery import KnownUsersGallery

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

camera_service = CameraService()
event_store = EventStore()
known_gallery = KnownUsersGallery()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing database...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    init_db()

    logger.info("Loading embedding model...")
    try:
        if MODEL_PATH.exists():
            app_state.ml_models["prediction_model"] = tf.keras.models.load_model(MODEL_PATH)
            logger.info("Embedding model loaded successfully.")
        else:
            logger.warning("Model not found at %s — ML endpoints will return 400.", MODEL_PATH)
    except Exception as exc:
        logger.error("Error loading embedding model: %s", exc)

    logger.info("Loading YOLO face detector...")
    app_state.face_detector = FaceDetector()

    logger.info("Known users gallery (legacy): %s user(s)", len(known_gallery.list_users()))
    event_store.seed_demo_events()
    yield

    camera_service.release()
    app_state.ml_models.clear()
    app_state.face_detector = None


app = FastAPI(lifespan=lifespan, title="VisiGuard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Authenticated API routers
app.include_router(auth_router)
app.include_router(gallery_router)
app.include_router(monitor_router)
app.include_router(events_router)
app.include_router(unknowns_router)

# Static uploads (gallery images, unknown snapshots)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")


def _get_model():
    model = app_state.ml_models.get("prediction_model")
    if model is None:
        raise HTTPException(
            status_code=400,
            detail="Embedding model is not loaded. Place best_embedding_model.keras in the Backend folder.",
        )
    return model


@app.get("/", response_model=RootResponse)
def read_root():
    model_ready = app_state.ml_models.get("prediction_model") is not None
    if model_ready:
        message = "VisiGuard API is live and model is ready."
    else:
        message = "VisiGuard API is live. Model not loaded — surveillance disabled."
    return RootResponse(message=message)


@app.get("/video-feed")
def video_feed():
    return StreamingResponse(
        camera_service.generate_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/video-feed/status", response_model=VideoFeedStatusResponse)
def video_feed_status():
    camera_service.open()
    return VideoFeedStatusResponse(
        connected=camera_service.is_connected,
        source_index=camera_service.source_index,
    )


@app.get("/video-feed/snapshot")
def video_feed_snapshot(compact: bool = True):
    """Single JPEG frame for background monitoring (compact=true reduces payload)."""
    from config import MONITOR_SNAPSHOT_MAX_WIDTH

    camera_service.open()
    frame = camera_service.read_frame()
    if frame is None:
        frame = camera_service._placeholder_frame("Camera unavailable")
    if compact and MONITOR_SNAPSHOT_MAX_WIDTH > 0:
        h, w = frame.shape[:2]
        if w > MONITOR_SNAPSHOT_MAX_WIDTH:
            scale = MONITOR_SNAPSHOT_MAX_WIDTH / w
            frame = cv2.resize(
                frame,
                (MONITOR_SNAPSHOT_MAX_WIDTH, max(1, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
    jpg = camera_service._encode_jpeg(frame)
    return Response(content=jpg, media_type="image/jpeg")


@app.get("/cameras", response_model=CamerasListResponse)
def list_cameras():
    raw = CameraService.list_available_sources()
    return CamerasListResponse(
        sources=[CameraSourceItem(**item) for item in raw],
        active_index=camera_service.source_index,
    )


@app.post("/set-camera", response_model=SetCameraResponse)
def set_camera(body: SetCameraRequest):
    try:
        connected, message = camera_service.set_source_index(body.source_index)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return SetCameraResponse(
        success=connected,
        source_index=body.source_index,
        connected=connected,
        message=message,
    )


@app.post("/compare-faces", response_model=CompareFacesResponse)
async def compare_faces(
    image_a: UploadFile = File(...),
    image_b: UploadFile = File(...),
):
    model = _get_model()
    bytes_a = await image_a.read()
    bytes_b = await image_b.read()

    if not bytes_a or not bytes_b:
        raise HTTPException(status_code=400, detail="Both image files are required.")

    return compare_two_faces(model, bytes_a, bytes_b)


@app.post("/analyze-frame", response_model=AnalyzeFrameResponse)
async def analyze_frame_endpoint(frame: UploadFile = File(...)):
    model = _get_model()
    image_bytes = await frame.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Frame image is required.")

    result = analyze_frame(model, known_gallery, image_bytes)

    if result.status == "authorized" and result.name:
        event_store.add_recognized(
            person_name=result.name,
            similarity_score=result.similarity_score or 0.0,
        )

    return result


@app.post("/add-known-user", response_model=AddKnownUserResponse)
def add_known_user(body: AddKnownUserRequest):
    try:
        record = known_gallery.add_user(body.name, np.asarray(body.embedding))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return AddKnownUserResponse(
        id=record["id"],
        name=record["name"],
        message=f"User '{record['name']}' added to Known Users Gallery.",
    )


@app.get("/known-users", response_model=KnownUsersListResponse)
def list_known_users():
    users = known_gallery.list_users()
    public = [
        {"id": u["id"], "name": u["name"], "created_at": u["created_at"]}
        for u in users
    ]
    return KnownUsersListResponse(count=len(public), users=public)


@app.post("/report-intrusion", response_model=ReportIntrusionResponse)
def report_intrusion(body: ReportIntrusionRequest):
    event_store.add_alert(
        person_label="Unauthorized Intrusion",
        confidence=body.similarity_score,
        severity="high",
    )
    return ReportIntrusionResponse(
        logged=True,
        message="Unauthorized intrusion logged.",
    )


@app.get("/alerts", response_model=AlertsResponse)
def list_alerts(limit: int = DEFAULT_ALERTS_LIMIT):
    return AlertsResponse(items=event_store.list_alerts(limit=limit))


@app.get("/recognized", response_model=RecognizedResponse)
def list_recognized(limit: int = DEFAULT_RECOGNIZED_LIMIT):
    return RecognizedResponse(items=event_store.list_recognized(limit=limit))


@app.post("/predict", response_model=PredictResponse)
async def predict():
    if app_state.ml_models.get("prediction_model") is not None:
        event_store.add_recognized(
            person_name="Authorized (pipeline test)",
            similarity_score=0.95,
        )
        return PredictResponse(status="Prediction recorded — authorized event logged.")
    event_store.add_alert(
        person_label="Pipeline test — model offline",
        confidence=None,
    )
    return PredictResponse(status="Endpoint ready — model offline, alert logged for demo.")
