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

export interface DatasetInfo {
  total_samples: number;
  audio_samples: number;
  image_samples: number;
  has_audio: boolean;
  has_image: boolean;
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
  loss: number | null;
  accuracy: number | null;
  last_heartbeat: string | null;
  latency_ms: number;
  training_active: boolean;
  dataset_info: DatasetInfo;
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

export interface AvailableJob {
  job_id: string;
  name: string;
  task_type: string;
  num_rounds: number;
  min_clients: number;
  joined_clients: string[];
  port: number;
  strategy: string;
  has_data: boolean;
}

export interface TrainingStartRequest {
  modality: string;
  total_rounds: number;
  server_address?: string;
  job_id?: string;
}

export interface TrainingStartResponse {
  status: string;
  pid: number;
  modality: string;
  total_rounds: number;
  job_id?: string | null;
}
