import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "best_embedding_model.keras"

# OpenCV webcam index (0 = default camera)
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
MAX_CAMERA_PROBE = int(os.getenv("MAX_CAMERA_PROBE", "10"))
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "80"))
FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", "640"))
FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", "480"))

# Face embedding input (matches project training notes)
FACE_IMAGE_SIZE = (112, 112)

# Legacy JSON gallery threshold (existing endpoints)
MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "0.78"))

# User-scoped monitoring threshold (new authenticated pipeline)
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.65"))

# Known users gallery persisted to disk (legacy)
DATA_DIR = BASE_DIR / "data"
KNOWN_USERS_PATH = DATA_DIR / "known_users.json"

# SQLite database
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{(DATA_DIR / 'visiguard.db').as_posix()}")

# Uploads for gallery faces and unknown snapshots
UPLOADS_DIR = Path(os.getenv("UPLOADS_DIR", str(DATA_DIR / "uploads")))
GALLERY_UPLOADS_DIR = UPLOADS_DIR / "gallery"
UNKNOWN_UPLOADS_DIR = UPLOADS_DIR / "unknowns"

# Surveillance
SURVEILLANCE_INTERVAL_SEC = float(os.getenv("SURVEILLANCE_INTERVAL_SEC", "3.0"))
MONITOR_FRAME_MAX_WIDTH = int(os.getenv("MONITOR_FRAME_MAX_WIDTH", "480"))
MONITOR_SNAPSHOT_MAX_WIDTH = int(os.getenv("MONITOR_SNAPSHOT_MAX_WIDTH", "320"))
RECOGNITION_LOG_COOLDOWN_SEC = float(os.getenv("RECOGNITION_LOG_COOLDOWN_SEC", "12.0"))
UNKNOWN_LOG_COOLDOWN_SEC = float(os.getenv("UNKNOWN_LOG_COOLDOWN_SEC", "15.0"))

# In-memory event store capacity (legacy)
MAX_EVENTS = int(os.getenv("MAX_EVENTS", "200"))

# Polling-friendly list limits
DEFAULT_ALERTS_LIMIT = int(os.getenv("DEFAULT_ALERTS_LIMIT", "50"))
DEFAULT_RECOGNIZED_LIMIT = int(os.getenv("DEFAULT_RECOGNIZED_LIMIT", "20"))
DEFAULT_LIST_LIMIT = int(os.getenv("DEFAULT_LIST_LIMIT", "100"))

# JWT authentication
JWT_SECRET = os.getenv("JWT_SECRET", "visiguard-dev-secret-change-in-production")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "1440"))

# YOLO face detector (weights auto-downloaded to data/checkpoints on first run)
_DEFAULT_YOLO = (DATA_DIR / "checkpoints" / "model.pt").as_posix()
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", _DEFAULT_YOLO)
YOLO_CONFIDENCE = float(os.getenv("YOLO_CONFIDENCE", "0.5"))

# CORS
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]
