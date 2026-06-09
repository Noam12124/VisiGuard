"""HTTP integration test for POST /monitor/frame."""
from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from fastapi.testclient import TestClient  # noqa: E402
from main import app  # noqa: E402
from database.session import SessionLocal  # noqa: E402
from database.models import User  # noqa: E402
from services.auth_service import AuthService  # noqa: E402


def make_jpeg() -> bytes:
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.ellipse(img, (320, 240), (90, 110), 0, 0, 360, (195, 170, 145), -1)
    ok, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def main() -> None:
    db = SessionLocal()
    user = db.query(User).order_by(User.id).first()
    if not user:
        print("FAIL: no user")
        return
    token = AuthService(db).create_access_token(user.id, user.username)
    db.close()

    jpeg = make_jpeg()
    files = {"frame": ("snapshot.jpg", BytesIO(jpeg), "image/jpeg")}
    headers = {"Authorization": f"Bearer {token}"}

    with TestClient(app) as client:
        res = client.post("/monitor/frame", files=files, headers=headers)
        print("Status:", res.status_code)
        print("Body:", res.json())
        assert res.status_code == 200
        body = res.json()
        assert "detections" in body
        assert "threshold" in body
        assert body["faces_detected"] >= 1
        assert len(body["detections"]) >= 1
        status = body["detections"][0]["status"]
        print(f"First detection status: {status}")
    print("API TEST PASSED")


if __name__ == "__main__":
    main()
