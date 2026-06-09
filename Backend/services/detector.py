from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from config import BASE_DIR, FACE_IMAGE_SIZE, YOLO_CONFIDENCE, YOLO_MODEL_PATH

logger = logging.getLogger(__name__)

HF_FACE_REPO = "arnabdhar/YOLOv8-Face-Detection"
HF_FACE_FILENAME = "model.pt"


def _resolve_yolo_path() -> Path:
    path = Path(YOLO_MODEL_PATH)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path


def _download_yolo_weights(dest: Path) -> Path:
    """Download YOLOv8-face weights from HuggingFace if missing."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("YOLO weights not found at %s — downloading from HuggingFace…", dest)
    try:
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            repo_id=HF_FACE_REPO,
            filename=HF_FACE_FILENAME,
            local_dir=str(dest.parent),
        )
        downloaded_path = Path(downloaded)
        if downloaded_path.resolve() != dest.resolve():
            downloaded_path.replace(dest)
        logger.info("YOLO weights downloaded to %s", dest)
        return dest
    except Exception as exc:
        logger.error(
            "YOLO auto-download failed (%s). Place %s at %s manually.",
            exc,
            HF_FACE_FILENAME,
            dest,
        )
        raise


class FaceDetector:
    """YOLO-based face detector with full-frame fallback."""

    def __init__(self) -> None:
        self._model = None
        self._available = False
        self._weights_path = _resolve_yolo_path()
        self._load()

    def _load(self) -> None:
        try:
            from ultralytics import YOLO

            weights = self._weights_path
            if not weights.exists():
                weights = _download_yolo_weights(weights)

            self._model = YOLO(str(weights))
            self._available = True
            logger.info("YOLO face detector loaded: %s", weights)
        except Exception as exc:
            logger.warning(
                "YOLO detector unavailable (%s) — using full-frame fallback.",
                exc,
            )
            self._model = None
            self._available = False

    @property
    def is_available(self) -> bool:
        return self._available

    def detect_faces(self, image_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Return list of (x1, y1, x2, y2) bounding boxes."""
        if self._model is None:
            h, w = image_bgr.shape[:2]
            logger.warning("YOLO unavailable — full-frame fallback (%dx%d)", w, h)
            return [(0, 0, w, h)]

        try:
            results = self._model(image_bgr, conf=YOLO_CONFIDENCE, verbose=False)
            boxes: list[tuple[int, int, int, int]] = []
            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = map(int, box[:4])
                    boxes.append((x1, y1, x2, y2))
            if boxes:
                logger.debug("YOLO detected %d face(s)", len(boxes))
                return boxes
            logger.debug("YOLO found no faces — full-frame fallback")
        except Exception as exc:
            logger.warning("YOLO inference failed: %s — full-frame fallback", exc)

        h, w = image_bgr.shape[:2]
        return [(0, 0, w, h)]

    @staticmethod
    def crop_and_align(image_bgr: np.ndarray, box: tuple[int, int, int, int]) -> bytes:
        """Crop face region, convert to RGB 112x112 JPEG bytes for embedding."""
        h, w = image_bgr.shape[:2]
        x1, y1, x2, y2 = box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            logger.warning("Invalid face box %s — using full frame crop", box)
            crop = image_bgr
        else:
            crop = image_bgr[y1:y2, x1:x2]

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, FACE_IMAGE_SIZE)

        bgr_out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(".jpg", bgr_out, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            raise ValueError("Failed to encode face crop.")
        return encoded.tobytes()

    @staticmethod
    def save_snapshot(image_bgr: np.ndarray, box: tuple[int, int, int, int], dest: Path) -> None:
        h, w = image_bgr.shape[:2]
        x1, y1, x2, y2 = box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = image_bgr[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else image_bgr
        dest.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dest), crop)
