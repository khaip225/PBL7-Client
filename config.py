import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    FASTAPI_URL: str = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000")
    CLIENT_NAME: str = os.getenv("CLIENT_NAME", "Hospital-01")
    CLIENT_ID: str = os.getenv("CLIENT_ID", "1")
    CLIENT_HOST: str = os.getenv("CLIENT_HOST", "127.0.0.1")
    CLIENT_MODALITY: str = os.getenv("CLIENT_MODALITY", "audio")

    FLOWER_SERVER_ADDRESS: str = os.getenv("FLOWER_SERVER_ADDRESS", "")
    CLIENT_API_KEY: str = os.getenv("CLIENT_API_KEY", "")

    # FL runtime state & heartbeat
    STATE_FILE: str = os.getenv("STATE_FILE", "./client_state.json")
    HEARTBEAT_INTERVAL: int = int(os.getenv("HEARTBEAT_INTERVAL", "30"))
    FL_DATA_DIR: str = os.getenv("FL_DATA_DIR", "./Local_Data/fl_data")

    # Prediction
    PREDICTION_THRESHOLD: float = float(os.getenv("PREDICTION_THRESHOLD", "0.5"))

    # Audio segmentation
    AUDIO_CHUNK_DURATION: float = float(os.getenv("AUDIO_CHUNK_DURATION", "5.0"))
    AUDIO_CHUNK_OVERLAP: float = float(os.getenv("AUDIO_CHUNK_OVERLAP", "1.0"))
    AUDIO_SEGMENT_THRESHOLD: float = float(os.getenv("AUDIO_SEGMENT_THRESHOLD", "0.5"))


config = Config()
