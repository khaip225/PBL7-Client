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
    audio_scores: dict | None = None    # {"Crackle": 0.7, "Wheeze": 0.1}
    image_scores: dict | None = None    # {"Pneumonia": 0.8, "COPD_Emphysema": 0.1, "Fibrosis": 0.05}
    fusion_scores: dict | None = None   # {"Pneumonia": 0.8, "COPD_Emphysema": 0.1, "Fibrosis": 0.05, "Normal": 0.2}


class DiagnosisResult(BaseModel):
    labels: list[str]       # multi-label: all classes with prob >= threshold
    confidence: float
    threshold: float


class SavedPaths(BaseModel):
    audio_dest: str | None = None
    image_dest: str | None = None


# ── 5-Action Pipeline Response Types ──────────────────────────────────────

class RetrievalItem(BaseModel):
    file_path: str
    file_name: str
    similarity: float
    case_id: str = ""
    disease_label: str = ""
    acoustic_label: str = ""


class CrossModalResult(BaseModel):
    scores: dict          # acoustic hoặc disease probabilities
    message: str          # cảnh báo cho bác sĩ


class LateFusionResultDetail(BaseModel):
    primary_diagnosis: str
    confidence: float
    confidence_level: str   # "Rất cao" / "Cao" / "Trung bình" / "Thấp"
    agreement: str
    fusion_scores: dict
    is_normal: bool


class DiagnosisResponse(BaseModel):
    mode: str
    result: DiagnosisResult
    scores: ScoreDetail
    saved: SavedPaths
    heatmap_path: str | None = None
    # ── 5-Action Pipeline fields ─────────────────────────────────────
    cross_modal: CrossModalResult | None = None       # HĐ2
    retrieval: list[RetrievalItem] = []               # HĐ3
    late_fusion: LateFusionResultDetail | None = None  # HĐ5
    attention_map_path: str | None = None             # HĐ4 (audio)
    timestamp: str


class AvailableJob(BaseModel):
    job_id: str
    name: str
    task_type: str
    num_rounds: int
    min_clients: int
    joined_clients: list[str] = []
    port: int
    strategy: str
    has_data: bool = False


class TrainingStartRequest(BaseModel):
    modality: str = "image"
    total_rounds: int = 10
    total_epochs: int = 2
    server_address: str | None = None
    job_id: str | None = None


class TrainingStartResponse(BaseModel):
    status: str
    pid: int
    modality: str
    total_rounds: int
    total_epochs: int
    job_id: str | None = None


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


class DatasetInfo(BaseModel):
    total_samples: int = 0
    audio_samples: int = 0
    image_samples: int = 0
    has_audio: bool = False
    has_image: bool = False


class TrainingStateResponse(BaseModel):
    client_id: str | None = None
    client_name: str
    modality: str
    status: str
    connected_to_server: bool
    connected_to_flower: bool
    current_round: int
    total_rounds: int
    current_epoch: int = 0
    total_epochs: int = 0
    loss: float | None = None
    accuracy: float | None = None
    train_loss: float | None = None
    train_accuracy: float | None = None
    val_loss: float | None = None
    val_accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    auc: float | None = None
    last_heartbeat: str | None = None
    latency_ms: int = 0
    training_active: bool
    dataset_info: DatasetInfo = DatasetInfo()
    system: SystemMetrics
    recent_logs: list[LogEntry] = []


class HistoryRecord(BaseModel):
    id: str
    timestamp: str
    labels: list[str]       # multi-label: detected classes
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


class ReviewListResponse(HistoryListResponse):
    pass


class ReviewApproveRequest(BaseModel):
    labels: dict[str, bool]  # class name → true/false


class ReviewApproveResponse(BaseModel):
    record_id: str
    labels: dict[str, bool]
    audio_dest: str | None = None
    image_dest: str | None = None
    batch_dir: str
    csv_path: str | None = None


class ReviewStateResponse(BaseModel):
    current_batch: int = 1
    client_id: str = "1"
    threshold: int = 0
