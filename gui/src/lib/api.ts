import type {
  HealthResponse,
  DiagnosisResult,
  TrainingState,
  TrainingStartRequest,
  TrainingStartResponse,
  AvailableJob,
  HistoryRecord,
  HistoryListResponse,
  ReviewApproveRequest,
  ReviewApproveResponse,
  ReviewListResponse,
  ReviewStateResponse,
} from "./types";

const BASE = "";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, options);
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API error ${res.status}: ${body}`);
  }
  return res.json();
}

export const api = {
  health: () => request<HealthResponse>("/api/health"),

  diagnosis: {
    run: (formData: FormData) =>
      request<DiagnosisResult>("/api/diagnosis", {
        method: "POST",
        body: formData,
      }),
  },

  training: {
    getState: () => request<TrainingState>("/api/training/state"),
    availableJobs: () => request<AvailableJob[]>("/api/training/available-jobs"),
    start: (body: TrainingStartRequest) =>
      request<TrainingStartResponse>("/api/training/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }),
    stop: () =>
      request<{ status: string }>("/api/training/stop", {
        method: "POST",
      }),
  },

  history: {
    list: (page = 1, pageSize = 20) =>
      request<HistoryListResponse>(
        `/api/history?page=${page}&page_size=${pageSize}`
      ),
    imageUrl: (id: string) => `/api/history/${id}/image`,
    audioUrl: (id: string) => `/api/history/${id}/audio`,
  },

  review: {
    list: (page = 1, pageSize = 20) =>
      request<ReviewListResponse>(
        `/api/review/pending?page=${page}&page_size=${pageSize}`
      ),
    approve: (recordId: string, body: ReviewApproveRequest) =>
      request<ReviewApproveResponse>(`/api/review/${recordId}/approve`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }),
    reject: (recordId: string) =>
      request<{ record_id: string; moved_files: string[] }>(
        `/api/review/${recordId}/reject`,
        { method: "POST" }
      ),
    /** Trash */
    listTrash: () =>
      request<HistoryRecord[]>("/api/review/trash"),
    restoreFromTrash: (recordId: string) =>
      request<{ record_id: string; restored_files: string[] }>(
        `/api/review/trash/${recordId}/restore`,
        { method: "POST" }
      ),
    deletePermanently: (recordId: string) =>
      request<{ record_id: string; deleted_files: string[] }>(
        `/api/review/trash/${recordId}/delete`,
        { method: "POST" }
      ),
    trashImageUrl: (recordId: string) => `/api/review/trash/${recordId}/image`,
    trashAudioUrl: (recordId: string) => `/api/review/trash/${recordId}/audio`,
    getState: () => request<ReviewStateResponse>("/api/review/state"),
    advanceBatch: () =>
      request<ReviewStateResponse>("/api/review/advance-batch", {
        method: "POST",
      }),
  },
};
