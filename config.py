import os
from dotenv import load_dotenv

load_dotenv()


def _env_bool(key: str, default: bool = False) -> bool:
    val = os.getenv(key, str(default)).lower()
    return val in ("1", "true", "yes", "on")


class Config:
    FASTAPI_URL: str = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000")
    CLIENT_NAME: str = os.getenv("CLIENT_NAME", "Hospital-01")
    CLIENT_HOST: str = os.getenv("CLIENT_HOST", "127.0.0.1")
    CLIENT_MODALITY: str = os.getenv("CLIENT_MODALITY", "audio")

    FLOWER_SERVER_ADDRESS: str = os.getenv("FLOWER_SERVER_ADDRESS", "")
    CLIENT_API_KEY: str = os.getenv("CLIENT_API_KEY", "pbl7-client-api-key-change-in-production")

    PREDICTION_THRESHOLD: float = float(os.getenv("PREDICTION_THRESHOLD", "0.5"))
    HEARTBEAT_INTERVAL: int = int(os.getenv("HEARTBEAT_INTERVAL", "30"))

    STATE_FILE: str = os.getenv("STATE_FILE", "./client_state.json")
    FL_DATA_DIR: str = os.getenv("FL_DATA_DIR", "./fl_worker/fl_data")

    # Multi-label config
    IMAGE_NUM_CLASSES: int = 3
    IMAGE_CLASS_NAMES: list = ["Pneumonia", "COPD_Emphysema", "Fibrosis"]
    AUDIO_NUM_CLASSES: int = 2
    AUDIO_CLASS_NAMES: list = ["Crackle", "Wheeze"]
    N_MELS: int = 128


config = Config()
