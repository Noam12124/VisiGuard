"""End-to-end monitor pipeline test (no webcam required)."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

import tensorflow as tf  # noqa: E402
from config import MODEL_PATH, SIMILARITY_THRESHOLD  # noqa: E402
from database.session import SessionLocal, init_db  # noqa: E402
from database.models import User  # noqa: E402
from services.detector import FaceDetector  # noqa: E402
from services.gallery_service import GalleryService  # noqa: E402
from services.monitor_service import MonitorService  # noqa: E402


def make_face_jpeg(width=640, height=480, center=(320, 240), radius=(90, 110)) -> bytes:
    """Synthetic portrait patch for pipeline mechanics testing."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.ellipse(img, center, radius, 0, 0, 360, (195, 170, 145), -1)
    cv2.ellipse(img, (center[0] - 30, center[1] - 20), (12, 8), 0, 0, 360, (40, 40, 40), -1)
    cv2.ellipse(img, (center[0] + 30, center[1] - 20), (12, 8), 0, 0, 360, (40, 40, 40), -1)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def main() -> None:
    init_db()
    db = SessionLocal()
    user = db.query(User).order_by(User.id).first()
    if user is None:
        print("FAIL: No users in DB")
        return

    if not MODEL_PATH.exists():
        print("FAIL: Model missing:", MODEL_PATH)
        return

    model = tf.keras.models.load_model(MODEL_PATH)
    detector = FaceDetector()
    print(f"YOLO available: {detector.is_available}")

    gallery = GalleryService(db)
    service = MonitorService(db, detector)

    # Clear test gallery entries for this user (optional — only names starting with E2E_)
    for entry in gallery.list_entries(user.id):
        if entry.person_name.startswith("E2E_"):
            gallery.delete_entry(user.id, entry.id)

    frame_a = make_face_jpeg(center=(200, 240))
    frame_b = make_face_jpeg(center=(440, 240))

    print("\n=== TEST 1: Unknown person (empty or no match) ===")
    result1 = service.process_frame(user, model, frame_a)
    print(json.dumps(result1, indent=2))
    assert result1["faces_detected"] >= 1, "Expected at least one face"
    assert any(d["status"] == "unknown" for d in result1["detections"]), "Expected unknown detection"

    print("\n=== TEST 2: Enroll via gallery (YOLO crop) ===")
    entry = gallery.add_from_bytes(
        user=user,
        model=model,
        person_name="E2E_TestPerson",
        image_bytes=frame_a,
        detector=detector,
        use_face_detection=True,
    )
    emb = gallery._json_to_embedding(entry.embedding)
    print(f"Enrolled id={entry.id} dims={emb.size}")

    print("\n=== TEST 3: Re-detect same face -> known ===")
    result2 = service.process_frame(user, model, frame_a)
    print(json.dumps(result2, indent=2))
    known = [d for d in result2["detections"] if d["status"] == "known"]
    assert known, f"Expected known detection, got: {result2['detections']}"
    print(f"PASS: Recognized as {known[0]['person_name']} @ {known[0]['confidence']:.4f}")

    print("\n=== TEST 4: Different face position ===")
    result3 = service.process_frame(user, model, frame_b)
    print(json.dumps(result3, indent=2))
    assert result3["detections"], "Expected at least one detection"

    print("\n=== TEST 5: Self-match similarity ===")
    status, name, score = gallery.identify(user.id, emb, SIMILARITY_THRESHOLD)
    print(f"Self-match: status={status} name={name} score={score}")
    assert status == "known" and score >= SIMILARITY_THRESHOLD

    # Cleanup
    gallery.delete_entry(user.id, entry.id)
    db.close()
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
