from fastapi import APIRouter, Request, HTTPException
from ..models.schemas import (
    TrainingStartRequest, TrainingStartResponse, TrainingStopResponse,
    TrainingStateResponse, AvailableJob,
)

router = APIRouter()


@router.get("/api/training/state", response_model=TrainingStateResponse)
async def get_training_state(request: Request):
    service = request.app.state.training_service
    return TrainingStateResponse(**service.get_state())


@router.get("/api/training/available-jobs")
async def get_available_jobs(request: Request):
    """Lấy danh sách job training từ VPS mà client có thể tham gia."""
    service = request.app.state.training_service
    jobs = service.get_available_jobs()
    return jobs


@router.post("/api/training/start", response_model=TrainingStartResponse)
async def start_training(body: TrainingStartRequest, request: Request):
    service = request.app.state.training_service
    try:
        pid = service.start(
            modality=body.modality,
            total_rounds=body.total_rounds,
            total_epochs=body.total_epochs,
            server_address=body.server_address,
            job_id=body.job_id,
        )
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return TrainingStartResponse(
        status="started",
        pid=pid,
        modality=body.modality,
        total_rounds=body.total_rounds,
        total_epochs=body.total_epochs,
        job_id=body.job_id,
    )


@router.post("/api/training/stop", response_model=TrainingStopResponse)
async def stop_training(request: Request):
    service = request.app.state.training_service
    pid = service._process.pid if service._process else None
    try:
        service.stop()
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return TrainingStopResponse(status="stopping", pid=pid)
