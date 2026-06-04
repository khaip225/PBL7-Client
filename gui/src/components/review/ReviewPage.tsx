import { useEffect, useMemo, useState } from "react";
import { Loader2 } from "lucide-react";
import type { HistoryRecord } from "../../lib/types";
import { useReview } from "../../hooks/useReview";
import ReviewTable from "./ReviewTable";
import ReviewDetail from "./ReviewDetail";

export default function ReviewPage() {
  const { data, loading, error, state, fetch, fetchState, approve, advanceBatch, reject } =
    useReview();
  const [selected, setSelected] = useState<HistoryRecord | null>(null);
  const [page, setPage] = useState(1);

  useEffect(() => {
    fetch(page);
  }, [fetch, page]);

  useEffect(() => {
    fetchState();
  }, [fetchState]);

  const hasItems = useMemo(() => (data?.items?.length ?? 0) > 0, [data]);

  const handleApprove = async (recordId: string, labels: Record<string, boolean>) => {
    await approve(recordId, labels);
    await fetch(page);
    setSelected(null);
  };

  const handleReject = async (recordId: string) => {
    await reject(recordId);
    await fetch(page);
  };

  const handleAdvance = async () => {
    await advanceBatch();
  };

  return (
    <div className="mx-auto max-w-6xl space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-xl font-bold text-white">Duyệt dữ liệu FL</h2>
          <p className="text-sm text-gray-400 mt-1">
            Kiểm duyệt dữ liệu pending trước khi đồng bộ vào FL.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <div className="rounded-lg border border-gray-800 bg-gray-900/60 px-3 py-2 text-xs text-gray-300">
            Batch hiện tại: <span className="text-white">{state?.current_batch ?? "-"}</span>
          </div>
          <button
            onClick={handleAdvance}
            className="rounded-lg border border-gray-700 px-3 py-2 text-xs text-gray-300 hover:bg-gray-800"
          >
            Tăng batch
          </button>
        </div>
      </div>

      {loading && !data && (
        <div className="flex items-center gap-2 text-gray-400">
          <Loader2 size={18} className="animate-spin" />
          Dang tai...
        </div>
      )}

      {error && (
        <div className="rounded-lg border border-red-600/30 bg-red-600/10 px-4 py-3 text-sm text-red-400">
          {error}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className={selected ? "lg:col-span-2" : "lg:col-span-3"}>
          {data && (
            <>
              <ReviewTable
                items={data.items}
                onSelect={setSelected}
                selectedId={selected?.id}
              />
              {data.total > data.page_size && (
                <div className="flex items-center justify-between mt-4 text-sm">
                  <span className="text-gray-500">
                    Tong: {data.total} | Trang {data.page}
                  </span>
                  <div className="flex gap-2">
                    <button
                      disabled={page <= 1}
                      onClick={() => setPage((p) => p - 1)}
                      className="rounded-lg px-3 py-1.5 text-gray-400 hover:text-white hover:bg-gray-800 disabled:opacity-30"
                    >
                      Truoc
                    </button>
                    <button
                      disabled={page * data.page_size >= data.total}
                      onClick={() => setPage((p) => p + 1)}
                      className="rounded-lg px-3 py-1.5 text-gray-400 hover:text-white hover:bg-gray-800 disabled:opacity-30"
                    >
                      Sau
                    </button>
                  </div>
                </div>
              )}
              {!hasItems && (
                <div className="card text-center text-gray-500 py-10">
                  Không có dữ liệu pending.
                </div>
              )}
            </>
          )}
        </div>
        {selected && (
          <div className="lg:col-span-1">
            <ReviewDetail item={selected} onApprove={handleApprove} onReject={handleReject} onClose={() => setSelected(null)} />
          </div>
        )}
      </div>
    </div>
  );
}
