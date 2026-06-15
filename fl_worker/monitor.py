import logging
import os
import platform
import time
from pathlib import Path

import psutil

logger = logging.getLogger(__name__)


def _resolve_batch_dir(fl_data_dir: str) -> str:
    """Resolve current batch directory from fl_state.json."""
    from shared.fl_utils import resolve_fl_data_dir
    return str(resolve_fl_data_dir(base_dir=fl_data_dir))


def scan_dataset_info(fl_data_dir: str = "./fl_worker/fl_data") -> dict:
    """Đếm số lượng sample audio và image trong thư mục FL data."""
    base = _resolve_batch_dir(fl_data_dir)
    info = {
        "total_samples": 0,
        "audio_samples": 0,
        "image_samples": 0,
        "has_audio": False,
        "has_image": False,
    }

    # Audio: đếm dòng trong CSV metadata
    audio_metadata_dir = os.path.join(base, "metadata", "audio_fl")
    if os.path.isdir(audio_metadata_dir):
        import csv
        for fname in os.listdir(audio_metadata_dir):
            if fname.endswith(".csv"):
                fpath = os.path.join(audio_metadata_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        count = sum(1 for _ in f) - 1  # trừ header
                        if count > 0:
                            info["audio_samples"] += count
                except Exception as e:
                    logger.warning("Failed to read audio metadata CSV %s: %s", fpath, e)

    # Image: đếm file ảnh trong thư mục fl_image
    image_dir = os.path.join(base, "fl_image")
    if os.path.isdir(image_dir):
        count = 0
        for root, dirs, files in os.walk(image_dir):
            for f in files:
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    count += 1
        info["image_samples"] = count

    info["total_samples"] = info["audio_samples"] + info["image_samples"]
    info["has_audio"] = info["audio_samples"] > 0
    info["has_image"] = info["image_samples"] > 0
    return info


class SystemMonitor:
    """Collects system metrics: CPU, RAM, GPU (if available), temperature."""

    def __init__(self):
        self._gpu_available = False
        try:
            import torch
            self._torch = torch
            self._gpu_available = torch.cuda.is_available()
        except ImportError:
            self._torch = None

    def get_hardware_info(self) -> dict:
        """One-time hardware snapshot for registration."""
        info = {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "cpu_count": psutil.cpu_count(logical=True),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "ram_total_gb": round(psutil.virtual_memory().total / (1024 ** 3), 1),
            "gpu_available": self._gpu_available,
        }
        if self._gpu_available and self._torch:
            info["gpu_name"] = self._torch.cuda.get_device_name(0)
            info["gpu_count"] = self._torch.cuda.device_count()
        return info

    def collect(self) -> dict:
        """Collect current system metrics. Called on each heartbeat."""
        cpu = psutil.cpu_percent(interval=0.5)
        ram = psutil.virtual_memory().percent

        metrics = {
            "cpu_percent": cpu,
            "ram_percent": ram,
            "gpu_percent": 0,
            "gpu_temp": None,
            "latency_ms": 0,
        }

        if self._gpu_available and self._torch:
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                metrics["gpu_percent"] = util.gpu
                try:
                    metrics["gpu_temp"] = pynvml.nvmlDeviceGetTemperature(
                        handle, pynvml.NVML_TEMPERATURE_GPU
                    )
                except Exception as e:
                    logger.debug("Failed to read GPU temp: %s", e)
                pynvml.nvmlShutdown()
            except ImportError:
                # No pynvml — report GPU memory usage if measurable
                try:
                    reserved = self._torch.cuda.memory_reserved(0)
                    total = self._torch.cuda.get_device_properties(0).total_memory
                    if total > 0:
                        metrics["gpu_percent"] = round((reserved / total) * 100, 1)
                except Exception as e:
                    logger.debug("Failed to read CUDA memory: %s", e)
            except Exception as e:
                logger.warning("GPU monitoring error: %s", e)

        # Disk usage on current drive
        try:
            disk = psutil.disk_usage("/" if platform.system() != "Windows" else "C:\\")
            metrics["disk_percent"] = disk.percent
        except Exception:
            metrics["disk_percent"] = 0

        return metrics
