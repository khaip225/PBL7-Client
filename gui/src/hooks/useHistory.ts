import { useState, useCallback } from "react";
import type { HistoryListResponse } from "../lib/types";
import { api } from "../lib/api";

export function useHistory() {
  const [data, setData] = useState<HistoryListResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetch = useCallback(async (page = 1, pageSize = 20) => {
    setLoading(true);
    try {
      const res = await api.history.list(page, pageSize);
      setData(res);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Lỗi tải lịch sử");
    } finally {
      setLoading(false);
    }
  }, []);

  return { data, loading, error, fetch };
}
