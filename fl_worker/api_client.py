import json
import logging
import os
import signal
import sys
import threading
import time
import uuid
from datetime import datetime, timezone

import requests

from config import config
from fl_worker.monitor import SystemMonitor, scan_dataset_info

logger = logging.getLogger("api_client")


class FastAPIClient:
    """HTTP client for FastAPI backend — handles register, heartbeat, offline."""

    def __init__(self):
        self.base_url = config.FASTAPI_URL.rstrip("/")
        self.client_id: uuid.UUID | None = None
        self._heartbeat_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._monitor = SystemMonitor()
        self._session = requests.Session()
        self._session.headers.update({
            "Content-Type": "application/json",
            "X-API-Key": config.CLIENT_API_KEY,
        })

        # Shared state written to disk for dashboard
        self._state: dict = {
            "client_id": None,
            "client_name": config.CLIENT_NAME,
            "modality": config.CLIENT_MODALITY,
            "status": "offline",
            "connected_to_server": False,
            "connected_to_flower": False,
            "current_round": 0,
            "total_rounds": 0,
            "current_epoch": 0,
            "total_epochs": 0,
            "loss": None,
            "accuracy": None,
            "train_loss": None,
            "train_accuracy": None,
            "val_loss": None,
            "val_accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "auc": None,
            "last_heartbeat": None,
            "latency_ms": 0,
            "system": {},
            "logs": [],
        }
        self._load_state()
        self._setup_signal_handlers()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self, task_type: str | None = None, hardware_info: dict | None = None,
                 dataset_info: dict | None = None) -> uuid.UUID | None:
        """Register this client with FastAPI (idempotent by client_name). Returns the assigned UUID."""
        if hardware_info is None:
            hardware_info = self._monitor.get_hardware_info()
        if dataset_info is None:
            dataset_info = scan_dataset_info(config.FL_DATA_DIR)

        payload = {
            "client_name": config.CLIENT_NAME,
            "client_host": config.CLIENT_HOST,
            "task_type": task_type,
            "hardware_info": hardware_info,
            "dataset_info": dataset_info,
        }

        try:
            resp = self._session.post(
                f"{self.base_url}/api/clients/register",
                json=payload,
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            self.client_id = uuid.UUID(data["id"])
            self._update_state(
                client_id=str(self.client_id),
                connected_to_server=True,
                status="online",
                dataset_info=dataset_info,
            )
            self._add_log("info", f"Registered with server as {self.client_id}")
            logger.info("Registered client_id=%s", self.client_id)
            return self.client_id
        except Exception as e:
            logger.error("Registration failed: %s", e)
            self._add_log("error", f"Registration failed: {e}")
            # If we have a cached client_id from previous session, try reusing it
            if self._state.get("client_id"):
                self.client_id = uuid.UUID(self._state["client_id"])
                logger.info("Reusing cached client_id=%s", self.client_id)
                return self.client_id
            return None

    def start_heartbeat(self):
        """Launch background heartbeat thread."""
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            return
        self._stop_event.clear()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True, name="heartbeat"
        )
        self._heartbeat_thread.start()
        logger.info("Heartbeat thread started (interval=%ds)", config.HEARTBEAT_INTERVAL)

    def send_offline(self):
        """Notify server that this client is going offline."""
        self._stop_event.set()
        if not self.client_id:
            return
        try:
            self._session.post(
                f"{self.base_url}/api/clients/{self.client_id}/offline",
                timeout=5,
            )
            self._update_state(connected_to_server=False, status="offline")
            self._add_log("info", "Sent offline notification")
            logger.info("Offline notification sent for %s", self.client_id)
        except Exception as e:
            logger.warning("Failed to send offline: %s", e)

    def get_available_jobs(self) -> list[dict]:
        """Poll VPS for available training jobs."""
        try:
            resp = self._session.get(
                f"{self.base_url}/api/jobs/available",
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning("Failed to fetch available jobs: %s", e)
            return []

    def join_job(self, job_id: str) -> dict | None:
        """Join a training job on VPS, get connection info."""
        try:
            resp = self._session.post(
                f"{self.base_url}/api/jobs/{job_id}/join",
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error("Failed to join job %s: %s", job_id, e)
            return None

    def update_training_state(self, **kwargs):
        """Update local training state (called from FL client callbacks)."""
        self._update_state(**kwargs)

    def shutdown(self):
        """Clean shutdown: stop heartbeat + notify offline."""
        self.send_offline()
        self._save_state()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _heartbeat_loop(self):
        while not self._stop_event.wait(timeout=config.HEARTBEAT_INTERVAL):
            if not self.client_id:
                continue
            try:
                metrics = self._monitor.collect()
                payload = {
                    "latency_ms": metrics.get("latency_ms", 0),
                    "hardware_info": {
                        "cpu_percent": metrics.get("cpu_percent", 0),
                        "ram_percent": metrics.get("ram_percent", 0),
                        "gpu_percent": metrics.get("gpu_percent", 0),
                        "gpu_temp": metrics.get("gpu_temp"),
                    },
                    "task_type": (self._state.get("modality") or "").replace("multimodal", "alignment"),
                    "dataset_info": self._state.get("dataset_info", {}),
                }
                resp = self._session.post(
                    f"{self.base_url}/api/clients/{self.client_id}/heartbeat",
                    json=payload,
                    timeout=10,
                )
                resp.raise_for_status()
                self._update_state(
                    last_heartbeat=datetime.now(timezone.utc).isoformat(),
                    latency_ms=metrics.get("latency_ms", 0),
                    system=metrics,
                )
            except Exception as e:
                logger.warning("Heartbeat failed: %s", e)
                self._update_state(connected_to_server=False)
                self._add_log("warning", f"Heartbeat failed: {e}")

    def _update_state(self, **kwargs):
        self._state.update(kwargs)
        self._save_state()

    def _add_log(self, level: str, message: str):
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "message": message,
        }
        self._state.setdefault("logs", []).append(entry)
        if len(self._state["logs"]) > 200:
            self._state["logs"] = self._state["logs"][-200:]
        self._save_state()

    def _save_state(self):
        try:
            with open(config.STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(self._state, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.debug("Failed to save state: %s", e)

    def _load_state(self):
        if os.path.exists(config.STATE_FILE):
            try:
                with open(config.STATE_FILE, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    self._state.update(loaded)
            except Exception as e:
                logger.warning("Failed to load state from %s: %s", config.STATE_FILE, e)

    def _setup_signal_handlers(self):
        def _handler(signum, frame):
            logger.info("Received signal %d, shutting down...", signum)
            self.shutdown()
            sys.exit(0)

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, _handler)
            except Exception as e:
                logger.warning("Failed to register signal handler for %d: %s", sig, e)


# Singleton
api_client = FastAPIClient()
