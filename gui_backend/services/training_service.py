import json
import os
import signal
import subprocess
import sys
from pathlib import Path

from config import config


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class TrainingService:
    def __init__(self):
        self._process: subprocess.Popen | None = None
        self._modality: str = "image"
        self._total_rounds: int = 10

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self, modality: str = "image", total_rounds: int = 10, server_address: str | None = None) -> int:
        if self.is_running:
            raise RuntimeError(f"Training already running (pid: {self._process.pid})")

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
            "system": state.get("system", {
                "cpu_percent": 0, "ram_percent": 0, "gpu_percent": 0,
                "gpu_temp": None, "disk_percent": 0, "latency_ms": 0,
            }),
            "recent_logs": state.get("logs", [])[-50:],
        }
