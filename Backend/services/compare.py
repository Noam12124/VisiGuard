from __future__ import annotations

from fastapi import HTTPException

from config import MATCH_THRESHOLD
from schemas import CompareFacesResponse
from services.embeddings import (
    compute_cosine_similarity,
    get_embedding,
)


def compare_two_faces(model, image_a_bytes: bytes, image_b_bytes: bytes) -> CompareFacesResponse:
    try:
        embed_a = get_embedding(model, image_a_bytes)
        embed_b = get_embedding(model, image_b_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to process images for comparison: {exc}",
        ) from exc

    score = compute_cosine_similarity(embed_a, embed_b)
    is_match = score >= MATCH_THRESHOLD
    match_percent = round(score * 100.0, 2)

    return CompareFacesResponse(
        is_match=is_match,
        similarity_score=round(score, 6),
        threshold=MATCH_THRESHOLD,
        match_percent=match_percent,
        identity_label="Match" if is_match else "No Match",
    )
