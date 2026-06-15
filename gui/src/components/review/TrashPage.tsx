import { useEffect, useMemo, useState } from "react";
import { Loader2, Trash2, RotateCcw } from "lucide-react";
import type { HistoryRecord } from "../../lib/types";
import { ALL_COLORS } from "../../lib/types";
import { api } from "../../lib/api";
import TrashDetail from "./TrashDetail";

export default function TrashPage() {
  const [items, setItems] = useState<HistoryRecord[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<HistoryRecord | null>(null);

  const fetch = async () => {
    setLoading(true);
    try {
      const data = await api.review.listTrash();
      setItems(data);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Trash load error");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetch();
  }, []);

  const hasItems = useMemo(() => (items?.length ?? 0) > 0, [items]);

  const handleRestore = async (recordId: string) => {
    await api.review.restoreFromTrash(recordId);
    await fetch();
    setSelected(null);
  };

  const handleDelete = async (recordId: string) => {
    await api.review.deletePermanently(recordId);
    await fetch();
    setSelected(null);
  };

  const handleEmptyTrash = async () => {
    if (!items) return;
    for (const item of items) {
      await api.review.deletePermanently(item.id);
    }
    await fetch();
  };

  return (
    <div className="mx-auto max-w-6xl space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-xl font-bold text-white flex items-center gap-2">
            <Trash2 size={22} className="text-yellow-500" />
            Trash
          </h2>
          <p className="text-sm text-gray-400 mt-1">
            Rejected data — restore or permanently delete.
          </p>
        </div>
        {hasItems && (
          <button
            onClick={handleEmptyTrash}
            className="rounded-lg border border-red-600/30 bg-red-600/10 px-3 py-2 text-xs text-red-400 hover:bg-red-600/20"
          >
            Delete All
          </button>
        )}
      </div>

      {loading && !items && (
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
          {items && items.length > 0 ? (
            <div className="card overflow-hidden !p-0">
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-gray-800 bg-gray-900/50">
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400">
                        Time
                      </th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400">
                        Label
                      </th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400">
                        Mode
                      </th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400">
                        Confidence
                      </th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400">
                        Actions
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {items.map((item) => {
                      const displayLabels = item.labels ?? [];
                      return (
                        <tr
                          key={item.id}
                          onClick={() => setSelected(item)}
                          className={`border-b border-gray-800/50 cursor-pointer transition-colors hover:bg-gray-800/50 ${
                            selected?.id === item.id ? "bg-yellow-600/10" : ""
                          }`}
                        >
                          <td className="px-4 py-3 text-gray-300 whitespace-nowrap">
                            {item.timestamp}
                          </td>
                          <td className="px-4 py-3">
                            <div className="flex flex-wrap gap-1">
                              {displayLabels.map((label) => (
                                <span
                                  key={label}
                                  className="rounded-full px-2 py-0.5 text-xs font-medium border"
                                  style={{
                                    backgroundColor:
                                      (ALL_COLORS[label] ?? "#6b7280") + "20",
                                    color: ALL_COLORS[label] ?? "#6b7280",
                                    borderColor:
                                      (ALL_COLORS[label] ?? "#6b7280") + "40",
                                  }}
                                >
                                  {label}
                                </span>
                              ))}
                            </div>
                          </td>
                          <td className="px-4 py-3 text-gray-400">{item.mode}</td>
                          <td className="px-4 py-3 text-gray-300">
                            {item.confidence != null
                              ? `${(item.confidence * 100).toFixed(1)}%`
                              : "—"}
                          </td>
                          <td className="px-4 py-3">
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                handleRestore(item.id);
                              }}
                              className="rounded-lg px-2 py-1 text-xs text-green-400 hover:bg-green-600/10 flex items-center gap-1"
                            >
                              <RotateCcw size={12} />
                              Restore
                            </button>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          ) : (
            items && (
              <div className="card text-center text-gray-500 py-10">
                <Trash2 size={32} className="mx-auto mb-3 opacity-30" />
                Trash is empty.
              </div>
            )
          )}
        </div>
        {selected && (
          <div className="lg:col-span-1">
            <TrashDetail
              item={selected}
              onRestore={handleRestore}
              onDelete={handleDelete}
              onClose={() => setSelected(null)}
            />
          </div>
        )}
      </div>
    </div>
  );
}
