import { useState, useCallback } from "react";
import type { TrainingState } from "../lib/types";
import { api } from "../lib/api";
import { useInterval } from "./useInterval";

const POLL_MS = 2500;

export function useTrainingState() {
  const [state, setState] = useState<TrainingState | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetch = useCallback(async () => {
    try {
      const data = await api.training.getState();
      setState(data);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Backend connection error");
    } finally {
      setLoading(false);
    }
  }, []);

  useInterval(fetch, POLL_MS);

  return { state, loading, error, refetch: fetch };
}
