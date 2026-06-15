import { useEffect, useState } from "react";
import type { HistoryRecord } from "../../lib/types";
import { useHistory } from "../../hooks/useHistory";
import HistoryTable from "./HistoryTable";
import HistoryDetail from "./HistoryDetail";
import { Loader2 } from "lucide-react";

export default function HistoryPage() {
  const { data, loading, error, fetch } = useHistory();
  const [selected, setSelected] = useState<HistoryRecord | null>(null);
  const [page, setPage] = useState(1);

  useEffect(() => {
    fetch(page);
  }, [fetch, page]);

  return (
    <div className="mx-auto max-w-5xl space-y-6">
      <div>
        <h2 className="text-xl font-bold text-white">Diagnosis History</h2>
        <p className="text-sm text-gray-400 mt-1">
          Review previous diagnoses.
        </p>
      </div>

      {loading && !data && (
        <div className="flex items-center gap-2 text-gray-400">
          <Loader2 size={18} className="animate-spin" />
          Loading...
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
              <HistoryTable
                items={data.items}
                onSelect={setSelected}
                selectedId={selected?.id}
              />
              {data.total > data.page_size && (
                <div className="flex items-center justify-between mt-4 text-sm">
                  <span className="text-gray-500">
                    Total: {data.total} | Page {data.page}
                  </span>
                  <div className="flex gap-2">
                    <button
                      disabled={page <= 1}
                      onClick={() => setPage((p) => p - 1)}
                      className="rounded-lg px-3 py-1.5 text-gray-400 hover:text-white hover:bg-gray-800 disabled:opacity-30"
                    >
                      Prev
                    </button>
                    <button
                      disabled={page * data.page_size >= data.total}
                      onClick={() => setPage((p) => p + 1)}
                      className="rounded-lg px-3 py-1.5 text-gray-400 hover:text-white hover:bg-gray-800 disabled:opacity-30"
                    >
                      Next
                    </button>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
        {selected && (
          <div className="lg:col-span-1">
            <HistoryDetail item={selected} onClose={() => setSelected(null)} />
          </div>
        )}
      </div>
    </div>
  );
}
