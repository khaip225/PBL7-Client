import os
from dotenv import load_dotenv

load_dotenv()


def _env_bool(key: str, default: bool = False) -> bool:
    val = os.getenv(key, str(default)).lower()
    return val in ("1", "true", "yes", "on")


class Config:
    FASTAPI_URL: str = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000")
    CLIENT_NAME: str = os.getenv("CLIENT_NAME", "Hospital-01")
    CLIENT_ID: str = os.getenv("CLIENT_ID", "1")
    CLIENT_HOST: str = os.getenv("CLIENT_HOST", "127.0.0.1")
    CLIENT_MODALITY: str = os.getenv("CLIENT_MODALITY", "audio")

    FLOWER_SERVER_ADDRESS: str = os.getenv("FLOWER_SERVER_ADDRESS", "")
    CLIENT_API_KEY: str = os.getenv("CLIENT_API_KEY", "pbl7-client-api-key-change-in-production")

    PREDICTION_THRESHOLD: float = float(os.getenv("PREDICTION_THRESHOLD", "0.5"))
    HEARTBEAT_INTERVAL: int = int(os.getenv("HEARTBEAT_INTERVAL", "30"))

    AUDIO_CHUNK_DURATION: float = float(os.getenv("AUDIO_CHUNK_DURATION", "5"))
    AUDIO_CHUNK_OVERLAP: float = float(os.getenv("AUDIO_CHUNK_OVERLAP", "0.5"))
    AUDIO_SEGMENT_THRESHOLD: float = float(os.getenv("AUDIO_SEGMENT_THRESHOLD", "0.5"))

    STATE_FILE: str = os.getenv("STATE_FILE", "./client_state.json")
    FL_DATA_DIR: str = os.getenv("FL_DATA_DIR", "./Local_Data/fl_data")
    FL_SYNC_ENABLED: bool = _env_bool("FL_SYNC_ENABLED", False)

    # Multi-label config — khớp với notebook Stage 5 (4 image classes, 3 audio classes)
    IMAGE_NUM_CLASSES: int = 4
    IMAGE_CLASS_NAMES: list = ["Normal", "Pneumonia", "COPD_Emphysema", "Fibrosis"]
    AUDIO_NUM_CLASSES: int = 3
    AUDIO_CLASS_NAMES: list = ["normal", "crackle", "wheeze"]
    N_MELS: int = 128


config = Config()
