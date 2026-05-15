import os
from fastapi import APIRouter, Request
from ..models.schemas import HealthResponse

router = APIRouter()


@router.get("/api/health", response_model=HealthResponse)
async def health(request: Request):
    models_loaded = {
        "audio": os.path.exists(request.app.state.audio_model_path),
        "image": os.path.exists(request.app.state.image_model_path),
    }
    return HealthResponse(
        status="ok",
        models_loaded=models_loaded,
        client_name=request.app.state.client_name,
        client_id=request.app.state.client_id,
    )
