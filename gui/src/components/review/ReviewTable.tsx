import type { HistoryRecord } from "../../lib/types";
import { ALL_COLORS } from "../../lib/types";

interface Props {
  items: HistoryRecord[];
  onSelect: (item: HistoryRecord) => void;
  selectedId?: string;
}

export default function ReviewTable({ items, onSelect, selectedId }: Props) {
  if (items.length === 0) {
    return (
      <div className="card text-center text-gray-500 py-10">
        Không có dữ liệu pending.
      </div>
    );
  }

  return (
    <div className="card overflow-hidden !p-0">
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-800 bg-gray-900/50">
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-400">Thời gian</th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-400">Nhãn gợi ý</th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-400">Mode</th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-400">Confidence</th>
            </tr>
          </thead>
          <tbody>
            {items.map((item) => {
              const displayLabels = item.labels ?? [];
              return (
                <tr
                  key={item.id}
                  onClick={() => onSelect(item)}
                  className={`border-b border-gray-800/50 cursor-pointer transition-colors hover:bg-gray-800/50 ${
                    selectedId === item.id ? "bg-blue-600/10" : ""
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
                            backgroundColor: (ALL_COLORS[label] ?? "#6b7280") + "20",
                            color: ALL_COLORS[label] ?? "#6b7280",
                            borderColor: (ALL_COLORS[label] ?? "#6b7280") + "40",
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
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
