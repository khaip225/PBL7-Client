import json
import os
import signal
import subprocess
import sys
from pathlib import Path

from config import config
from fl_worker.api_client import api_client

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _check_data_for_modality(modality: str) -> bool:
    """Kiểm tra client có dữ liệu training cho modality này không."""
    base = Path(config.FL_DATA_DIR)
    if modality == "audio":
        csv_path = base / "metadata" / "audio_fl" / "client_1_train.csv"
        return csv_path.exists()
    elif modality == "image":
        img_dir = base / "fl_image"
        return img_dir.exists() and any(img_dir.iterdir())
    return False


class TrainingService:
    def __init__(self):
        self._process: subprocess.Popen | None = None
        self._modality: str = "image"
        self._total_rounds: int = 10
        self._job_id: str | None = None

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def get_available_jobs(self) -> list[dict]:
        """Lấy danh sách job training từ VPS, có kiểm tra data."""
        raw = api_client.get_available_jobs()
        enriched = []
        for job in raw:
            job["has_data"] = _check_data_for_modality(job.get("task_type", ""))
            enriched.append(job)
        return enriched

    def start(self, modality: str = "image", total_rounds: int = 10,
              server_address: str | None = None, job_id: str | None = None) -> int:
        if self.is_running:
            raise RuntimeError(f"Training already running (pid: {self._process.pid})")

        if job_id:
            # Flow mới: join job trên VPS trước
            join_result = api_client.join_job(job_id)
            if not join_result:
                raise RuntimeError("Không thể tham gia job training trên server")
            server_address = join_result.get("server_address", server_address)
            modality = join_result.get("task_type", modality)
            total_rounds = join_result.get("num_rounds", total_rounds)

        if server_address is None:
            port = 8080 if modality == "audio" else 8081
            if "://" in config.FASTAPI_URL:
                host = config.FASTAPI_URL.split("://")[1].split(":")[0]
            else:
                host = "127.0.0.1"
            server_address = f"{host}:{port}"

        cmd = [
            sys.executable, "fl_worker/client.py",
            "--client_id", "1",
            "--modality", modality,
            "--total_rounds", str(total_rounds),
            "--server_address", server_address,
        ]

        self._process = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._modality = modality
        self._total_rounds = total_rounds
        self._job_id = job_id
        return self._process.pid

    def stop(self):
        if not self.is_running:
            raise RuntimeError("No training running")
        self._process.send_signal(signal.SIGTERM)
        try:
            self._process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()

    def get_state(self) -> dict:
        state_file = config.STATE_FILE
        if os.path.exists(state_file):
            with open(state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
        else:
            state = {}

        training_active = self.is_running

        return {
            "client_id": state.get("client_id"),
            "client_name": state.get("client_name", config.CLIENT_NAME),
            "modality": state.get("modality", config.CLIENT_MODALITY),
            "status": state.get("status", "offline"),
            "connected_to_server": state.get("connected_to_server", False),
            "connected_to_flower": state.get("connected_to_flower", False),
            "current_round": state.get("current_round", 0),
            "total_rounds": state.get("total_rounds", self._total_rounds),
            "loss": state.get("loss"),
            "accuracy": state.get("accuracy"),
            "last_heartbeat": state.get("last_heartbeat"),
            "latency_ms": state.get("latency_ms", 0),
            "training_active": training_active,
            "job_id": self._job_id,
            "dataset_info": state.get("dataset_info", {}),
            "system": state.get("system", {
                "cpu_percent": 0, "ram_percent": 0, "gpu_percent": 0,
                "gpu_temp": None, "disk_percent": 0, "latency_ms": 0,
            }),
            "recent_logs": state.get("logs", [])[-50:],
        }
