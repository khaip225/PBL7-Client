import os
import signal
import subprocess
import sys
from pathlib import Path

from config import config
from fl_worker.api_client import api_client
from shared.fl_utils import resolve_fl_data_dir, check_data_for_modality

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class TrainingService:
    def __init__(self):
        self._process: subprocess.Popen | None = None
        self._modality: str = "image"
        self._total_rounds: int = 10
        self._job_id: str | None = None
        self._log_file = None

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def get_available_jobs(self) -> list[dict]:
        """Lấy danh sách job training từ VPS, có kiểm tra data."""
        raw = api_client.get_available_jobs()
        enriched = []
        for job in raw:
            job["has_data"] = check_data_for_modality(resolve_fl_data_dir(), job.get("task_type", ""))
            enriched.append(job)
        return enriched

    def start(
        self,
        modality: str = "image",
        total_rounds: int = 10,
        total_epochs: int = 2,
        server_address: str | None = None,
        job_id: str | None = None,
    ) -> int:
        if self.is_running:
            raise RuntimeError(f"Training already running (pid: {self._process.pid})")

        if job_id:
            # Flow mới: join job trên VPS trước
            join_result = api_client.join_job(job_id)
            if not join_result:
                raise RuntimeError("Unable to join training job on server")
            server_address = join_result.get("server_address", server_address)
            modality = join_result.get("task_type", modality)
            total_rounds = join_result.get("num_rounds", total_rounds)

        if server_address is None:
            port_map = {"audio": 8080, "image": 8081, "alignment": 8082, "multimodal": 8082}
            port = port_map.get(modality, 8081)
            if "://" in config.FASTAPI_URL:
                host = config.FASTAPI_URL.split("://")[1].split(":")[0]
            else:
                host = "127.0.0.1"
            server_address = f"{host}:{port}"

        client_id = self._normalize_client_id(config.CLIENT_ID)
        cmd = [
            sys.executable, "fl_worker/client.py",
            "--client_id", str(client_id),
            "--modality", modality,
            "--total_rounds", str(total_rounds),
            "--total_epochs", str(total_epochs),
            "--server_address", server_address,
        ]

        log_path = PROJECT_ROOT / "training.log"
        if self._log_file:
            self._log_file.close()
            self._log_file = None
        self._log_file = open(log_path, "w", encoding="utf-8", buffering=1)

        self._process = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=self._log_file,
            stderr=self._log_file,
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
        if self._log_file:
            self._log_file.close()
            self._log_file = None

    def get_state(self) -> dict:
        state_file = config.STATE_FILE
        if os.path.exists(state_file):
            with open(state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
        else:
            state = {}

        training_active = self.is_running

        default_system = {
            "cpu_percent": 0,
            "ram_percent": 0,
            "gpu_percent": 0,
            "gpu_temp": None,
            "disk_percent": 0,
            "latency_ms": 0,
        }
        system = {**default_system, **(state.get("system") or {})}

        return {
            "client_id": state.get("client_id"),
            "client_name": state.get("client_name", config.CLIENT_NAME),
            "modality": state.get("modality", config.CLIENT_MODALITY),
            "status": state.get("status", "offline"),
            "connected_to_server": state.get("connected_to_server", False),
            "connected_to_flower": state.get("connected_to_flower", False),
            "current_round": state.get("current_round", 0),
            "total_rounds": state.get("total_rounds", self._total_rounds),
            "current_epoch": state.get("current_epoch", 0),
            "total_epochs": state.get("total_epochs", 0),
            "loss": state.get("loss"),
            "accuracy": state.get("accuracy"),
            "train_loss": state.get("train_loss"),
            "train_accuracy": state.get("train_accuracy"),
            "val_loss": state.get("val_loss"),
            "val_accuracy": state.get("val_accuracy"),
            "precision": state.get("precision"),
            "recall": state.get("recall"),
            "f1": state.get("f1"),
            "auc": state.get("auc"),
            "last_heartbeat": state.get("last_heartbeat"),
            "latency_ms": state.get("latency_ms", 0),
            "training_active": training_active,
            "system": system,
            "dataset_info": state.get("dataset_info", {}),
            "recent_logs": state.get("logs", [])[-50:],
        }

    @staticmethod
    def _normalize_client_id(value: str | int) -> int:
        if isinstance(value, int):
            return value
        text = str(value).strip()
        if text.startswith("client_"):
            text = text.split("client_", 1)[1]
        return int(text) if text.isdigit() else 1
