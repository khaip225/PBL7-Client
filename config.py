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


config = Config()
