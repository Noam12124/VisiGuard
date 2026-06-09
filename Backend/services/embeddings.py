from __future__ import annotations

import logging

import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


def load_and_preprocess_image(file_bytes: bytes) -> np.ndarray:
    """
    Decode upload bytes and preprocess exactly per project spec:
    BGR -> RGB, resize 112x112, float32 (0-255 scale, not /255).
    """
    if not file_bytes:
        raise ValueError("Empty image file")

    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image. Use a valid JPEG or PNG file.")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (112, 112))
    img = img.astype("float32")
    return img


def get_embedding(model, file_bytes: bytes) -> np.ndarray:
    """Run model.predict on a single preprocessed face image; returns flat 512-d vector."""
    img = load_and_preprocess_image(file_bytes)
    raw = model.predict(np.expand_dims(img, axis=0), verbose=0)
    embedding = np.asarray(raw, dtype=np.float32).reshape(-1)
    if embedding.size == 0:
        raise ValueError("Model returned an empty embedding vector.")
    logger.debug("Embedding generated: dims=%d", embedding.size)
    return embedding


def compute_cosine_similarity(embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
    """Cosine similarity via scikit-learn (stable for 2D embedding matrices)."""
    a = np.asarray(embedding_a, dtype=np.float32).reshape(1, -1)
    b = np.asarray(embedding_b, dtype=np.float32).reshape(1, -1)
    score = cosine_similarity(a, b)[0][0]
    return float(np.clip(score, -1.0, 1.0))
