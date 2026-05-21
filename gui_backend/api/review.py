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
        result = service.approve(record_id, body.label)
    except ValueError as e:
        raise HTTPException(404, str(e))
    return ReviewApproveResponse(**result)


@router.get("/api/review/state", response_model=ReviewStateResponse)
async def get_state(request: Request):
    service = request.app.state.review_service
    return ReviewStateResponse(**service.get_state())


@router.post("/api/review/advance-batch", response_model=ReviewStateResponse)
async def advance_batch(request: Request):
    service = request.app.state.review_service
    return ReviewStateResponse(**service.advance_batch())
