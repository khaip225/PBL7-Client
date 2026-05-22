import type {
  HealthResponse,
  DiagnosisResult,
  TrainingState,
  TrainingStartRequest,
  TrainingStartResponse,
  AvailableJob,
  HistoryListResponse,
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
};
