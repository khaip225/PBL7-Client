import type { HistoryRecord } from "../../lib/types";
import { api } from "../../lib/api";
import { X } from "lucide-react";

interface Props {
  item: HistoryRecord;
  onClose: () => void;
}

export default function HistoryDetail({ item, onClose }: Props) {
  return (
    <div className="card relative">
      <button
        onClick={onClose}
        className="absolute top-4 right-4 text-gray-500 hover:text-white"
      >
        <X size={18} />
      </button>

      <h3 className="text-sm font-semibold text-gray-300 mb-4">Chi tiết</h3>

      <div className="space-y-4">
        <div className="grid grid-cols-2 gap-3 text-sm">
          <div>
            <span className="text-gray-500">ID:</span>
            <p className="text-gray-300 font-mono text-xs">{item.id}</p>
          </div>
          <div>
            <span className="text-gray-500">Thời gian:</span>
            <p className="text-gray-300">{item.timestamp}</p>
          </div>
          <div>
            <span className="text-gray-500">Kết quả:</span>
            <p
              className={
                item.label === "normal" ? "text-green-400" : "text-red-400"
              }
            >
              {item.label}
            </p>
          </div>
          <div>
            <span className="text-gray-500">Mode:</span>
            <p className="text-gray-300">{item.mode}</p>
          </div>
          <div>
            <span className="text-gray-500">Confidence:</span>
            <p className="text-gray-300">
              {item.confidence != null
                ? `${(item.confidence * 100).toFixed(1)}%`
                : "—"}
            </p>
          </div>
        </div>

        {(item.image_file || item.image_path) && (
          <div>
            <p className="text-xs text-gray-500 mb-2">Ảnh X-quang</p>
            <img
              src={api.history.imageUrl(item.id)}
              alt="X-quang"
              className="rounded-lg border border-gray-800 max-h-64 object-contain bg-black"
            />
          </div>
        )}

        {(item.audio_file || item.audio_path) && (
          <div>
            <p className="text-xs text-gray-500 mb-2">Audio</p>
            <audio controls src={api.history.audioUrl(item.id)} className="w-full" />
          </div>
        )}
      </div>
    </div>
  );
}
