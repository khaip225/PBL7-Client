"""Cross-modal retrieval API: audio ↔ image search in shared embedding space."""

import os
from fastapi import APIRouter, HTTPException, Request, Query

router = APIRouter()


@router.post("/api/retrieval/audio-to-image")
async def audio_to_image(
    request: Request,
    audio_path: str = Query(..., description="Path to audio file on disk"),
    top_k: int = Query(5, ge=1, le=50),
):
    """Find top-k X-ray images most similar to an audio file."""
    service = _get_service(request)
    result = service.audio_to_image(audio_path, top_k=top_k)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


@router.post("/api/retrieval/image-to-audio")
async def image_to_audio(
    request: Request,
    image_path: str = Query(..., description="Path to image file on disk"),
    top_k: int = Query(5, ge=1, le=50),
):
    """Find top-k audio files most similar to an image file."""
    service = _get_service(request)
    result = service.image_to_audio(image_path, top_k=top_k)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


@router.post("/api/retrieval/prototype-similarities")
async def prototype_similarities(
    request: Request,
    modality: str = Query(..., description="image or audio"),
    file_path: str = Query(..., description="Path to file on disk"),
):
    """Get cosine similarity of a file to all prototypes (for radar/spider chart)."""
    if modality not in ("image", "audio"):
        raise HTTPException(400, "modality must be 'image' or 'audio'")
    service = _get_service(request)
    result = service.get_prototype_similarities(modality, file_path)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


@router.post("/api/retrieval/build-index")
async def build_index(request: Request):
    """Rebuild the cross-modal embedding index from stored files."""
    service = _get_service(request)
    count = service.build_index()
    return {"indexed": count}


def _get_service(request: Request):
    svc = getattr(request.app.state, "retrieval_service", None)
    if svc is None:
        raise HTTPException(503, "Retrieval service not initialized")
    return svc
