import os
import sys
import signal
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ── Path setup ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from config import config
from ai_engines.pipeline_engine import PipelineEngine
from local_managers.storage_manager import StorageManager
from local_managers.data_sync_manager import DataSyncManager
from gui_backend.services.diagnosis_service import DiagnosisService
from gui_backend.services.training_service import TrainingService
from gui_backend.services.history_service import HistoryService
from gui_backend.services.review_service import ReviewService
from gui_backend.services.retrieval_service import RetrievalService
from gui_backend.api.health import router as health_router
from gui_backend.api.diagnosis import router as diagnosis_router
from gui_backend.api.training import router as training_router
from gui_backend.api.history import router as history_router
from gui_backend.api.review import router as review_router
from gui_backend.api.retrieval import router as retrieval_router
from gui_backend.api.tsne import router as tsne_router
from fl_worker.api_client import api_client

# ── Model paths ─────────────────────────────────────────────────────────────
STAGE4_PATH = os.path.join(BASE_DIR, "models", "stage4_best_model.pth")
IMAGE_HEAD_PATH = os.path.join(BASE_DIR, "models", "Global_Image_best.pth")
AUDIO_HEAD_PATH = os.path.join(BASE_DIR, "models", "Global_Audio_best.pth")

# Database paths (optional — created by scripts/build_database.py)
AUDIO_DB_PATH = os.path.join(BASE_DIR, "Local_Data", "databases", "audio_database.npy")
IMAGE_DB_PATH = os.path.join(BASE_DIR, "Local_Data", "databases", "image_database.npy")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Khởi tạo PipelineEngine và các services."""
    print("[GUI Backend] Initializing PipelineEngine...")

    device = "cuda" if __import__('torch').cuda.is_available() else "cpu"
    print(f"[GUI Backend] Device: {device}")

    # ── PipelineEngine ───────────────────────────────────────────────────
    pipeline = PipelineEngine(
        stage4_path=STAGE4_PATH,
        image_head_path=IMAGE_HEAD_PATH,
        audio_head_path=AUDIO_HEAD_PATH,
        audio_db_path=AUDIO_DB_PATH if os.path.exists(AUDIO_DB_PATH) else None,
        image_db_path=IMAGE_DB_PATH if os.path.exists(IMAGE_DB_PATH) else None,
        device=device,
        threshold=config.PREDICTION_THRESHOLD,
    )
    app.state.pipeline_engine = pipeline
    print("[GUI Backend] PipelineEngine initialized")

    # ── Storage & Sync ───────────────────────────────────────────────────
    storage_manager = StorageManager(client_id=config.CLIENT_ID)
    data_sync_manager = DataSyncManager(client_id=config.CLIENT_ID)

    # ── Diagnosis Service (dùng PipelineEngine) ─────────────────────────
    app.state.diagnosis_service = DiagnosisService(pipeline, storage_manager)

    # ── Training ─────────────────────────────────────────────────────────
    app.state.training_service = TrainingService()

    # ── History & Review ─────────────────────────────────────────────────
    app.state.history_service = HistoryService()
    app.state.review_service = ReviewService(app.state.history_service, data_sync_manager, pipeline)

    # ── Retrieval Service (dùng PipelineEngine) ─────────────────────────
    retrieval_svc = RetrievalService(pipeline, storage_manager)
    app.state.retrieval_service = retrieval_svc
    print("[GUI Backend] Retrieval service initialized")

    # ── Model path info ──────────────────────────────────────────────────
    app.state.audio_model_path = STAGE4_PATH
    app.state.image_model_path = STAGE4_PATH
    app.state.client_name = config.CLIENT_NAME
    app.state.client_id = None

    # ── Connect to VPS server (background, không chặn startup) ───────────
    import threading
    def _connect_vps():
        try:
            client_uuid = api_client.register()
            if client_uuid:
                app.state.client_id = str(client_uuid)
                api_client.start_heartbeat()
                print(f"[GUI Backend] Connected to VPS as {client_uuid}")
            else:
                print("[GUI Backend] WARNING: Could not register with VPS — continuing offline")
        except Exception as e:
            print(f"[GUI Backend] WARNING: VPS not available — running in offline mode")
    threading.Thread(target=_connect_vps, daemon=True, name="vps-connect").start()

    print("[GUI Backend] Ready.")
    yield

    print("[GUI Backend] Shutting down...")
    if app.state.training_service.is_running:
        try:
            app.state.training_service.stop()
        except Exception:
            pass

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
app.include_router(retrieval_router)
app.include_router(tsne_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("gui_backend.main:app", host="127.0.0.1", port=8001, reload=True)
