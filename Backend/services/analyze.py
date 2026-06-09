from __future__ import annotations

from fastapi import HTTPException

from config import MATCH_THRESHOLD
from schemas import AnalyzeFrameResponse
from services.embeddings import get_embedding
from services.gallery import KnownUsersGallery


def analyze_frame(
    model,
    gallery: KnownUsersGallery,
    image_bytes: bytes,
) -> AnalyzeFrameResponse:
    try:
        embedding = get_embedding(model, image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to analyze frame: {exc}",
        ) from exc

    status, name, score = gallery.identify(embedding)
    flat_embedding = embedding.reshape(-1).tolist()

    if status == "authorized":
        return AnalyzeFrameResponse(
            status="authorized",
            name=name,
            similarity_score=score,
            threshold=MATCH_THRESHOLD,
            embedding=None,
        )

    return AnalyzeFrameResponse(
        status="unknown",
        name=None,
        similarity_score=score,
        threshold=MATCH_THRESHOLD,
        embedding=flat_embedding,
    )
