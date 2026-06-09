"""
detector.py — YOLOv8-face detection wrapper.

Returns bounding boxes AND 5-point facial keypoints so that aligner.py
can do proper eye-based alignment.

Model used:  yolov8n-face  (nano — fast, good enough for detection)
             Loaded from  checkpoints/yolov8n_face.pt
             or auto-downloaded from HuggingFace on first run.

Each detection result contains:
    {
        "bbox":      [x1, y1, x2, y2],   # pixel coords in original image
        "confidence": float,
        "keypoints": np.ndarray (5, 2),   # left-eye, right-eye, nose,
                                           #  mouth-left, mouth-right
        "face_crop": np.ndarray (112,112,3)  # aligned + preprocessed
    }
"""

import os
import numpy as np
import cv2
import config
from aligner import FaceAligner


# ── Lazy import Ultralytics (installed at runtime in Colab) ────────────────

def _load_yolo(weights_path: str):
    """Load YOLO model; download weights if missing."""
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "Ultralytics not installed. Run:  pip install ultralytics"
        )

    if not os.path.exists(weights_path):
        print(f"[detector] Weights not found at {weights_path}. "
              "Attempting to download yolov8n-face from HuggingFace…")
        try:
            from huggingface_hub import hf_hub_download
            dl_path = hf_hub_download(
                repo_id   = "arnabdhar/YOLOv8-Face-Detection",
                filename  = "model.pt",
                local_dir = os.path.dirname(weights_path),
            )
            os.rename(dl_path, weights_path)
            print(f"[detector] Downloaded to {weights_path}")
        except Exception as e:
            print(f"[detector] Auto-download failed ({e}). "
                  "Please manually download 'model.pt' from "
                  "https://huggingface.co/arnabdhar/YOLOv8-Face-Detection "
                  f"and place it at:  {weights_path}")
            raise FileNotFoundError(f"YOLO weights missing: {weights_path}")

    model = YOLO(weights_path)
    print(f"[detector] Loaded YOLOv8-face from {weights_path}")
    return model


# ── Main detector class ────────────────────────────────────────────────────

class FaceDetector:
    """
    YOLOv8-face wrapper.

    Args:
        weights_path:  Path to the .pt weights file.
        conf_threshold: Minimum detection confidence.
        min_face_size:  Minimum face height/width in pixels to accept.
    """

    def __init__(
        self,
        weights_path:   str   = config.YOLO_WEIGHTS,
        conf_threshold: float = config.FACE_CONF_THRESHOLD,
        min_face_size:  int   = config.MIN_FACE_SIZE,
    ):
        self.conf_threshold = conf_threshold
        self.min_face_size  = min_face_size
        self.aligner        = FaceAligner()
        self._model         = None          # lazy load
        self._weights_path  = weights_path

    def _ensure_loaded(self):
        if self._model is None:
            self._model = _load_yolo(self._weights_path)

    # ── Core detection ─────────────────────────────────────────────────────

    def detect(self, image_bgr: np.ndarray) -> list[dict]:
        """
        Detect all faces in an image.

        Args:
            image_bgr: OpenCV BGR image (H, W, 3) uint8.

        Returns:
            List of dicts, one per detected face (sorted by confidence desc).
            Each dict: {"bbox", "confidence", "keypoints", "face_crop"}
        """
        self._ensure_loaded()

        results = self._model(
            image_bgr,
            conf    = self.conf_threshold,
            verbose = False,
        )

        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            # Optional keypoints (5-point landmarks)
            kpts_data = None
            if result.keypoints is not None:
                kpts_data = result.keypoints.xy.cpu().numpy()  # (N, 5, 2)

            for i in range(len(boxes)):
                conf = float(boxes.conf[i].cpu().numpy())
                if conf < self.conf_threshold:
                    continue

                # Bounding box in xyxy format
                xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy

                face_w = x2 - x1
                face_h = y2 - y1

                # Skip very small faces
                if face_w < self.min_face_size or face_h < self.min_face_size:
                    continue

                # Keypoints for this detection (5 × 2 array or None)
                kpts = None
                if kpts_data is not None and i < len(kpts_data):
                    kpts = kpts_data[i]   # shape (5, 2)

                # Align + preprocess the face crop
                face_crop = self.aligner.align_from_bbox_and_kpts(
                    image_bgr, (x1, y1, x2, y2), kpts
                )

                detections.append({
                    "bbox":       [x1, y1, x2, y2],
                    "confidence": conf,
                    "keypoints":  kpts,
                    "face_crop":  face_crop,      # (112, 112, 3) uint8 BGR
                })

        # Sort by confidence (highest first)
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections

    def detect_largest(self, image_bgr: np.ndarray) -> dict | None:
        """
        Return only the largest (by area) detected face, or None.
        Useful for single-face scenarios.
        """
        dets = self.detect(image_bgr)
        if not dets:
            return None
        return max(
            dets,
            key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1])
        )

    # ── Visualisation helper ───────────────────────────────────────────────

    def draw_detections(
        self,
        image_bgr: np.ndarray,
        detections: list[dict],
        labels: list[str] | None = None,
    ) -> np.ndarray:
        """
        Draw bounding boxes and optional identity labels on the image.

        Returns a copy with annotations.
        """
        vis = image_bgr.copy()
        for idx, det in enumerate(detections):
            x1, y1, x2, y2 = det["bbox"]
            conf            = det["confidence"]
            label           = labels[idx] if labels and idx < len(labels) else ""
            kpts            = det.get("keypoints")

            # Box
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 220, 100), 2)

            # Confidence + optional identity
            text = f"{label} ({conf:.2f})" if label else f"{conf:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 220, 100), -1)
            cv2.putText(vis, text, (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

            # Keypoints (draw eyes in distinct colour)
            if kpts is not None:
                colours = [(0, 120, 255), (0, 200, 255),
                           (200, 200, 0), (200, 0, 200), (0, 200, 200)]
                for pt_idx, (px, py) in enumerate(kpts):
                    if px > 0 and py > 0:
                        colour = colours[pt_idx % len(colours)]
                        cv2.circle(vis, (int(px), int(py)), 3, colour, -1)

        return vis


# ── Module-level singleton ─────────────────────────────────────────────────

_default_detector: FaceDetector | None = None


def get_detector() -> FaceDetector:
    """Return (and lazily create) the module-level default detector."""
    global _default_detector
    if _default_detector is None:
        _default_detector = FaceDetector()
    return _default_detector
