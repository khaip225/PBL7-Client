from fastapi import APIRouter, HTTPException, Query, Request
from ..models.schemas import ReviewApproveRequest, ReviewApproveResponse, ReviewListResponse, ReviewStateResponse

router = APIRouter()


@router.get("/api/review/pending", response_model=ReviewListResponse)
async def list_pending(
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    service = request.app.state.review_service
    return ReviewListResponse(**service.list_pending(page=page, page_size=page_size))


@router.post("/api/review/{record_id}/approve", response_model=ReviewApproveResponse)
async def approve_record(record_id: str, body: ReviewApproveRequest, request: Request):
    service = request.app.state.review_service
    try:
        result = service.approve(record_id, body.labels)
    except ValueError as e:
        raise HTTPException(404, str(e))
    return ReviewApproveResponse(**result)


@router.post("/api/review/{record_id}/reject")
async def reject_record(record_id: str, request: Request):
    """Chuyển record pending vào thùng rác."""
    service = request.app.state.review_service
    try:
        result = service.reject(record_id)
    except ValueError as e:
        raise HTTPException(404, str(e))
    return result


# ---------------------------------------------------------------------------
# Thùng rác
# ---------------------------------------------------------------------------

@router.get("/api/review/trash")
async def list_trash(request: Request):
    """Liệt kê dữ liệu trong thùng rác."""
    service = request.app.state.review_service
    return service.list_trash()


@router.post("/api/review/trash/{record_id}/restore")
async def restore_from_trash(record_id: str, request: Request):
    """Khôi phục dữ liệu từ thùng rác về pending."""
    service = request.app.state.review_service
    try:
        result = service.restore_from_trash(record_id)
    except ValueError as e:
        raise HTTPException(404, str(e))
    return result


@router.post("/api/review/trash/{record_id}/delete")
async def delete_permanently(record_id: str, request: Request):
    """Xóa vĩnh viễn dữ liệu khỏi thùng rác."""
    service = request.app.state.review_service
    try:
        result = service.delete_permanently(record_id)
    except ValueError as e:
        raise HTTPException(404, str(e))
    return result


@router.get("/api/review/trash/{record_id}/image")
async def trash_image(record_id: str, request: Request):
    """Serve ảnh X-quang từ thùng rác."""
    import os
    from fastapi.responses import FileResponse
    service = request.app.state.review_service
    trash_items = service.list_trash()
    for item in trash_items:
        if item["id"] == record_id:
            path = item.get("image_path")
            if path and os.path.exists(path):
                return FileResponse(path, media_type="image/jpeg")
    raise HTTPException(404, "Image not found")


@router.get("/api/review/trash/{record_id}/audio")
async def trash_audio(record_id: str, request: Request):
    """Serve audio từ thùng rác."""
    import os
    from fastapi.responses import FileResponse
    service = request.app.state.review_service
    trash_items = service.list_trash()
    for item in trash_items:
        if item["id"] == record_id:
            path = item.get("audio_path")
            if path and os.path.exists(path):
                return FileResponse(path, media_type="audio/wav")
    raise HTTPException(404, "Audio not found")


@router.get("/api/review/state", response_model=ReviewStateResponse)
async def get_state(request: Request):
    service = request.app.state.review_service
    return ReviewStateResponse(**service.get_state())


@router.post("/api/review/advance-batch", response_model=ReviewStateResponse)
async def advance_batch(request: Request):
    service = request.app.state.review_service
    return ReviewStateResponse(**service.advance_batch())
