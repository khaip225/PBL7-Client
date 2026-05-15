import platform
import time

import psutil


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
                # pynvml for detailed GPU stats
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                metrics["gpu_percent"] = util.gpu
                try:
                    metrics["gpu_temp"] = pynvml.nvmlDeviceGetTemperature(
                        handle, pynvml.NVML_TEMPERATURE_GPU
                    )
                except Exception:
                    pass
                pynvml.nvmlShutdown()
            except ImportError:
                # Fallback: just report GPU memory usage
                mem = self._torch.cuda.memory_reserved(0) / self._torch.cuda.max_memory_reserved(0)
                metrics["gpu_percent"] = round(mem * 100, 1) if mem > 0 else 0
            except Exception:
                pass

        # Disk usage on current drive
        try:
            disk = psutil.disk_usage("/" if platform.system() != "Windows" else "C:\\")
            metrics["disk_percent"] = disk.percent
        except Exception:
            metrics["disk_percent"] = 0

        return metrics
