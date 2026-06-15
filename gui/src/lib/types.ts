export interface HealthResponse {
  status: string;
  models_loaded: { audio: boolean; image: boolean };
  client_name: string;
  client_id: string | null;
}

/** Multi-label probability per disease/acoustic attribute */
export interface ClassProbabilities {
  [className: string]: number;
}

/** Kết quả cross-modal zero-shot (HĐ2) */
export interface CrossModalResult {
  scores: ClassProbabilities;   // acoustic hoặc disease probabilities
  message: string;              // cảnh báo cho bác sĩ
}

/** Một mục retrieval (HĐ3) */
export interface RetrievalItem {
  file_path: string;
  file_name: string;
  similarity: number;
  case_id: string;
  disease_label: string;
  acoustic_label: string;
}

/** Kết quả late fusion (HĐ5) */
export interface LateFusionResultDetail {
  primary_diagnosis: string;
  confidence: number;
  confidence_level: string;   // "Very High" / "High" / "Medium" / "Low"
  agreement: string;
  fusion_scores: ClassProbabilities;
  is_normal: boolean;
}

export interface DiagnosisResult {
  mode: string;
  result: {
    labels: string[];         // all detected disease/acoustic classes
    confidence: number;
    threshold: number;
  };
  scores: {
    /** Multi-label: {"Crackle": 0.7, "Wheeze": 0.1} — null if mode != audio/fusion */
    audio_scores: ClassProbabilities | null;
    /** Multi-label: {"Pneumonia": 0.8, "COPD_Emphysema": 0.1, "Fibrosis": 0.05} — null if mode != image/fusion */
    image_scores: ClassProbabilities | null;
    /** Ontology-fused: {"Pneumonia": 0.8, "COPD_Emphysema": 0.1, "Fibrosis": 0.05, "Normal": 0.2} — only in fusion mode */
    fusion_scores: ClassProbabilities | null;
  };
  saved: {
    audio_dest: string | null;
    image_dest: string | null;
  };
  heatmap_path: string | null;
  // ── 5-Action Pipeline ──────────────────────────────────────────
  cross_modal: CrossModalResult | null;           // HĐ2
  retrieval: RetrievalItem[];                      // HĐ3
  late_fusion: LateFusionResultDetail | null;      // HĐ5
  attention_map_path: string | null;               // HĐ4 (audio)
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
  labels: string[];          // multi-label: detected classes
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

export interface ReviewListResponse extends HistoryListResponse {}

export interface ReviewApproveRequest {
  labels: Record<string, boolean>;  // class name → true/false
}

export interface ReviewApproveResponse {
  record_id: string;
  labels: Record<string, boolean>;
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
  job_id?: string;
}

export interface TrainingStartResponse {
  status: string;
  pid: number;
  modality: string;
  total_rounds: number;
  total_epochs: number;
  job_id?: string | null;
}

/** Disease & acoustic labels for display */
export const DISEASE_NAMES = ["Pneumonia", "COPD_Emphysema", "Fibrosis"];
export const DISEASE_COLORS: Record<string, string> = {
  Pneumonia: "#ef4444",
  COPD_Emphysema: "#f97316",
  Fibrosis: "#8b5cf6",
};
export const ACOUSTIC_NAMES = ["Crackle", "Wheeze"];
export const ACOUSTIC_COLORS: Record<string, string> = {
  Crackle: "#06b6d4",
  Wheeze: "#eab308",
};
export const ALL_LABELS = [...DISEASE_NAMES, ...ACOUSTIC_NAMES];
export const ALL_COLORS: Record<string, string> = { ...DISEASE_COLORS, ...ACOUSTIC_COLORS };
