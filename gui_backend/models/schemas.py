from __future__ import annotations
from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    models_loaded: dict[str, bool]
    client_name: str
    client_id: str | None = None


class DiagnosisRequest(BaseModel):
    mode: str  # "fusion", "audio", "image"


class ScoreDetail(BaseModel):
    audio_score: float | None = None
    image_score: float | None = None
    fusion_score: float | None = None


class DiagnosisResult(BaseModel):
    label: str
    confidence: float
    threshold: float


class SavedPaths(BaseModel):
    audio_dest: str | None = None
    image_dest: str | None = None


class DiagnosisResponse(BaseModel):
    mode: str
    result: DiagnosisResult
    scores: ScoreDetail
    saved: SavedPaths
    timestamp: str


class TrainingStartRequest(BaseModel):
    modality: str = "image"
    total_rounds: int = 10
    server_address: str | None = None


class TrainingStartResponse(BaseModel):
    status: str
    pid: int
    modality: str
    total_rounds: int


class TrainingStopResponse(BaseModel):
    status: str
    pid: int | None = None


class SystemMetrics(BaseModel):
    cpu_percent: float
    ram_percent: float
    gpu_percent: float
    gpu_temp: float | None = None
    disk_percent: float
    latency_ms: int = 0


class LogEntry(BaseModel):
    timestamp: str
    level: str
    message: str


class TrainingStateResponse(BaseModel):
    client_id: str | None = None
    client_name: str
    modality: str
    status: str
    connected_to_server: bool
    connected_to_flower: bool
    current_round: int
    total_rounds: int
    loss: float | None = None
    accuracy: float | None = None
    last_heartbeat: str | None = None
    latency_ms: int = 0
    training_active: bool
    system: SystemMetrics
    recent_logs: list[LogEntry] = []


class HistoryRecord(BaseModel):
    id: str
    timestamp: str
    label: str
    mode: str
    audio_file: str | None = None
    image_file: str | None = None
    audio_path: str | None = None
    image_path: str | None = None
    confidence: float | None = None


class HistoryListResponse(BaseModel):
    items: list[HistoryRecord]
    total: int
    page: int
    page_size: int
