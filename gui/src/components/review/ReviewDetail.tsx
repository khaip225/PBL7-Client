import { useMemo, useState } from "react";
import { X } from "lucide-react";
import type { HistoryRecord } from "../../lib/types";
import { api } from "../../lib/api";

interface Props {
  item: HistoryRecord;
  onApprove: (recordId: string, label: string) => Promise<void>;
  onClose: () => void;
}

function normalizeLabel(label: string) {
  const lower = label.toLowerCase();
  if (lower === "normal") return "Normal";
  if (lower === "abnormal") return "Abnormal";
  return "Normal";
}

export default function ReviewDetail({ item, onApprove, onClose }: Props) {
  const [label, setLabel] = useState(() => normalizeLabel(item.label));
  const [saving, setSaving] = useState(false);

  const displayLabel = useMemo(() => normalizeLabel(item.label), [item.label]);

  const handleApprove = async () => {
    setSaving(true);
    try {
      await onApprove(item.id, label);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="card relative">
      <button
        onClick={onClose}
        className="absolute top-4 right-4 text-gray-500 hover:text-white"
      >
        <X size={18} />
      </button>

      <h3 className="text-sm font-semibold text-gray-300 mb-4">Duyệt hồ sơ</h3>

      <div className="space-y-4">
        <div className="grid grid-cols-2 gap-3 text-sm">
          <div>
            <span className="text-gray-500">ID:</span>
            <p className="text-gray-300 font-mono text-xs break-all">{item.id}</p>
          </div>
          <div>
            <span className="text-gray-500">Thời gian:</span>
            <p className="text-gray-300">{item.timestamp}</p>
          </div>
          <div>
            <span className="text-gray-500">Nhãn gợi ý:</span>
            <p className="text-gray-300">{displayLabel}</p>
          </div>
          <div>
            <span className="text-gray-500">Mode:</span>
            <p className="text-gray-300">{item.mode}</p>
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

        <div className="rounded-lg border border-gray-800 bg-gray-900/40 p-3">
          <p className="text-xs text-gray-500 mb-2">Chọn nhãn chốt</p>
          <div className="flex gap-3 text-sm">
            <label className="flex items-center gap-2 text-gray-300">
              <input
                type="radio"
                name="final-label"
                value="Normal"
                checked={label === "Normal"}
                onChange={() => setLabel("Normal")}
              />
              Normal
            </label>
            <label className="flex items-center gap-2 text-gray-300">
              <input
                type="radio"
                name="final-label"
                value="Abnormal"
                checked={label === "Abnormal"}
                onChange={() => setLabel("Abnormal")}
              />
              Abnormal
            </label>
          </div>
        </div>

        <button
          onClick={handleApprove}
          disabled={saving}
          className="w-full rounded-lg bg-blue-600 px-3 py-2 text-sm font-semibold text-white hover:bg-blue-500 disabled:opacity-60"
        >
          {saving ? "Đang đồng bộ..." : "Lưu và đồng bộ"}
        </button>
      </div>
    </div>
  );
}
