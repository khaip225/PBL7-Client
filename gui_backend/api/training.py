from fastapi import APIRouter, Request, HTTPException
from ..models.schemas import TrainingStartRequest, TrainingStartResponse, TrainingStopResponse, TrainingStateResponse

router = APIRouter()


@router.get("/api/training/state", response_model=TrainingStateResponse)
async def get_training_state(request: Request):
    service = request.app.state.training_service
    return TrainingStateResponse(**service.get_state())


@router.post("/api/training/start", response_model=TrainingStartResponse)
async def start_training(body: TrainingStartRequest, request: Request):
    service = request.app.state.training_service
    try:
        pid = service.start(
            modality=body.modality,
            total_rounds=body.total_rounds,
            total_epochs=body.total_epochs,
            server_address=body.server_address,
        )
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return TrainingStartResponse(
        status="started",
        pid=pid,
        modality=body.modality,
        total_rounds=body.total_rounds,
        total_epochs=body.total_epochs,
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
