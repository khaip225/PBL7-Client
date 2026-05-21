export interface HealthResponse {
  status: string;
  models_loaded: { audio: boolean; image: boolean };
  client_name: string;
  client_id: string | null;
}

export interface DiagnosisResult {
  mode: string;
  result: {
    label: string;
    confidence: number;
    threshold: number;
  };
  scores: {
    audio_score: number | null;
    image_score: number | null;
    fusion_score: number | null;
  };
  saved: {
    audio_dest: string | null;
    image_dest: string | null;
  };
  timestamp: string;
}

export interface SystemMetrics {
  cpu_percent: number;
  ram_percent: number;
  gpu_percent: number;
  gpu_temp: number | null;
  disk_percent: number;
  latency_ms: number;
}

export interface TrainingState {
  client_id: string | null;
  client_name: string;
  modality: string;
  status: string;
  connected_to_server: boolean;
  connected_to_flower: boolean;
  current_round: number;
  total_rounds: number;
  current_epoch: number;
  total_epochs: number;
  loss: number | null;
  accuracy: number | null;
  train_loss: number | null;
  train_accuracy: number | null;
  val_loss: number | null;
  val_accuracy: number | null;
  precision: number | null;
  recall: number | null;
  f1: number | null;
  auc: number | null;
  last_heartbeat: string | null;
  latency_ms: number;
  training_active: boolean;
  system: SystemMetrics;
  recent_logs: LogEntry[];
}

export interface LogEntry {
  timestamp: string;
  level: string;
  message: string;
}

export interface HistoryRecord {
  id: string;
  timestamp: string;
  label: string;
  mode: string;
  audio_file: string | null;
  image_file: string | null;
  audio_path: string | null;
  image_path: string | null;
  confidence: number | null;
}

export interface HistoryListResponse {
  items: HistoryRecord[];
  total: number;
  page: number;
  page_size: number;
}

export interface ReviewListResponse extends HistoryListResponse {}

export interface ReviewApproveRequest {
  label: string;
}

export interface ReviewApproveResponse {
  record_id: string;
  label: string;
  audio_dest: string | null;
  image_dest: string | null;
  batch_dir: string;
  csv_path: string | null;
}

export interface ReviewStateResponse {
  current_batch: number;
  client_id: string;
  threshold: number;
}

export interface TrainingStartRequest {
  modality: string;
  total_rounds: number;
  total_epochs: number;
  server_address?: string;
}

export interface TrainingStartResponse {
  status: string;
  pid: number;
  modality: string;
  total_rounds: number;
  total_epochs: number;
}
