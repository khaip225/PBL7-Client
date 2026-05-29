import os
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse
from ..models.schemas import HistoryListResponse

router = APIRouter()


@router.get("/api/history", response_model=HistoryListResponse)
async def list_history(
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    service = request.app.state.history_service
    return HistoryListResponse(**service.list_records(page=page, page_size=page_size))


@router.get("/api/history/{record_id}/image")
async def get_history_image(record_id: str, request: Request):
    service = request.app.state.history_service
    path = service.get_image_path(record_id)
    if not path or not os.path.exists(path):
        raise HTTPException(404, "Image not found")
    return FileResponse(path, media_type="image/jpeg")


@router.get("/api/history/{record_id}/audio")
async def get_history_audio(record_id: str, request: Request):
    service = request.app.state.history_service
    path = service.get_audio_path(record_id)
    if not path or not os.path.exists(path):
        raise HTTPException(404, "Audio not found")
    return FileResponse(path, media_type="audio/wav")


@router.get("/api/heatmap")
async def get_heatmap(path: str, request: Request):
    """Serve heatmap image. Chi chap nhan path nam trong Local_Data."""
    if not path or not os.path.exists(path):
        raise HTTPException(404, "Heatmap not found")
    # Chi cho phep truy cap file trong thu muc Local_Data
    allowed_base = os.path.abspath("./Local_Data")
    target = os.path.abspath(path)
    if not target.startswith(allowed_base):
        raise HTTPException(403, "Access denied")
    return FileResponse(path, media_type="image/png")
