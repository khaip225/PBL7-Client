import { useCallback, useState } from "react";
import type { ReviewListResponse, ReviewStateResponse } from "../lib/types";
import { api } from "../lib/api";

export function useReview() {
  const [data, setData] = useState<ReviewListResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [state, setState] = useState<ReviewStateResponse | null>(null);

  const fetch = useCallback(async (page = 1, pageSize = 20) => {
    setLoading(true);
    try {
      const res = await api.review.list(page, pageSize);
      setData(res);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Loi tai danh sach pending");
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchState = useCallback(async () => {
    try {
      const res = await api.review.getState();
      setState(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Loi tai trang thai batch");
    }
  }, []);

  const approve = useCallback(async (recordId: string, label: string) => {
    return api.review.approve(recordId, { label });
  }, []);

  const advanceBatch = useCallback(async () => {
    const res = await api.review.advanceBatch();
    setState(res);
    return res;
  }, []);

  return { data, loading, error, state, fetch, fetchState, approve, advanceBatch };
}
