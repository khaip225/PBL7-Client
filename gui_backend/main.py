import os
import sys
import signal
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure PBL7-Client is on the path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from config import config
from ai_engines.audio_engine.predictor import AudioPredictor
from ai_engines.image_engine.predictor import ImagePredictor
from ai_engines.fusion import OntologyFusion
from local_managers.storage_manager import StorageManager
from local_managers.data_sync_manager import DataSyncManager
from gui_backend.services.diagnosis_service import DiagnosisService
from gui_backend.services.training_service import TrainingService
from gui_backend.services.history_service import HistoryService
from gui_backend.services.review_service import ReviewService
from gui_backend.api.health import router as health_router
from gui_backend.api.diagnosis import router as diagnosis_router
from gui_backend.api.training import router as training_router
from gui_backend.api.history import router as history_router
from gui_backend.api.review import router as review_router
from fl_worker.api_client import api_client

AUDIO_MODEL = os.path.join(BASE_DIR, "ai_engines", "current_weights", "best_global_audio.pth")
IMAGE_MODEL = os.path.join(BASE_DIR, "ai_engines", "current_weights", "best_global_image.pth")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[GUI Backend] Initializing AI models...")
    app.state.audio_model_path = AUDIO_MODEL
    app.state.image_model_path = IMAGE_MODEL
    app.state.client_name = config.CLIENT_NAME
    app.state.client_id = None

    audio_predictor = AudioPredictor(AUDIO_MODEL, threshold=config.PREDICTION_THRESHOLD)
    image_predictor = ImagePredictor(IMAGE_MODEL)
    fusion_engine = OntologyFusion(audio_weight=0.4)
    storage_manager = StorageManager(client_id=config.CLIENT_ID)
    data_sync_manager = DataSyncManager(client_id=config.CLIENT_ID)

    app.state.diagnosis_service = DiagnosisService(
        audio_predictor, image_predictor, fusion_engine, storage_manager
    )
    app.state.training_service = TrainingService()
    app.state.history_service = HistoryService()
    app.state.review_service = ReviewService(app.state.history_service, data_sync_manager)

    # Connect to VPS server (register + start heartbeat)
    print("[GUI Backend] Connecting to VPS server...")
    try:
        client_uuid = api_client.register()
        if client_uuid:
            app.state.client_id = str(client_uuid)
            api_client.start_heartbeat()
            print(f"[GUI Backend] Connected to VPS as {client_uuid}")
        else:
            print("[GUI Backend] WARNING: Could not register with VPS — continuing offline")
    except Exception as e:
        print(f"[GUI Backend] WARNING: VPS connection failed: {e}")

    print("[GUI Backend] Ready.")
    yield

    print("[GUI Backend] Shutting down...")
    if app.state.training_service.is_running:
        try:
            app.state.training_service.stop()
        except Exception:
            pass

    # Notify VPS that we're going offline
    try:
        api_client.shutdown()
        print("[GUI Backend] Sent offline notification to VPS")
    except Exception as e:
        print(f"[GUI Backend] WARNING: Failed to send offline: {e}")


app = FastAPI(title="PBL7 Client GUI Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(diagnosis_router)
app.include_router(training_router)
app.include_router(history_router)
app.include_router(review_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("gui_backend.main:app", host="127.0.0.1", port=8001, reload=True)
