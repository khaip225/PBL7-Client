import { useMemo, useState } from "react";
import { X } from "lucide-react";
import type { HistoryRecord } from "../../lib/types";
import { ALL_LABELS, ALL_COLORS } from "../../lib/types";
import { api } from "../../lib/api";

interface Props {
  item: HistoryRecord;
  onApprove: (recordId: string, labels: Record<string, boolean>) => Promise<void>;
  onReject?: (recordId: string) => Promise<void>;
  onClose: () => void;
}

export default function ReviewDetail({ item, onApprove, onReject, onClose }: Props) {
  // Initialize checkboxes: pre-check labels from AI suggestion
  const [labels, setLabels] = useState<Record<string, boolean>>(() => {
    const init: Record<string, boolean> = {};
    for (const name of ALL_LABELS) {
      init[name] = (item.labels ?? []).includes(name);
    }
    return init;
  });
  const [saving, setSaving] = useState(false);
  const [deleting, setDeleting] = useState(false);

  const suggestedLabels = useMemo(() => item.labels ?? [], [item.labels]);

  const toggleLabel = (name: string) => {
    setLabels((prev) => ({ ...prev, [name]: !prev[name] }));
  };

  const handleApprove = async () => {
    setSaving(true);
    try {
      await onApprove(item.id, labels);
    } finally {
      setSaving(false);
    }
  };

  const handleReject = async () => {
    if (!onReject) return;
    setDeleting(true);
    try {
      await onReject(item.id);
      onClose();
    } finally {
      setDeleting(false);
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

      <h3 className="text-sm font-semibold text-gray-300 mb-4">Review Record</h3>

      <div className="space-y-4">
        <div className="grid grid-cols-2 gap-3 text-sm">
          <div>
            <span className="text-gray-500">ID:</span>
            <p className="text-gray-300 font-mono text-xs break-all">{item.id}</p>
          </div>
          <div>
            <span className="text-gray-500">Time:</span>
            <p className="text-gray-300">{item.timestamp}</p>
          </div>
          <div>
            <span className="text-gray-500">AI Suggestion:</span>
            <div className="flex flex-wrap gap-1 mt-1">
              {suggestedLabels.map((label) => (
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
          </div>
          <div>
            <span className="text-gray-500">Mode:</span>
            <p className="text-gray-300">{item.mode}</p>
          </div>
        </div>

        {(item.image_file || item.image_path) && (
          <div>
            <p className="text-xs text-gray-500 mb-2">X-ray Image</p>
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

        {/* Multi-label checkboxes */}
        <div className="rounded-lg border border-gray-800 bg-gray-900/40 p-3">
          <p className="text-xs text-gray-500 mb-3">Select final labels</p>

          <div className="space-y-3">
            {/* Diseases */}
            <div>
              <p className="text-xs text-gray-600 mb-1.5">🫁 Lung Diseases</p>
              <div className="flex flex-wrap gap-2">
                {["Pneumonia", "COPD_Emphysema", "Fibrosis"].map((name) => (
                  <label
                    key={name}
                    className="flex items-center gap-1.5 text-sm cursor-pointer"
                  >
                    <input
                      type="checkbox"
                      checked={labels[name] || false}
                      onChange={() => toggleLabel(name)}
                      className="accent-blue-500"
                    />
                    <span
                      className="rounded-full px-2 py-0.5 text-xs font-medium"
                      style={{
                        backgroundColor: (ALL_COLORS[name] ?? "#6b7280") + "20",
                        color: ALL_COLORS[name] ?? "#6b7280",
                      }}
                    >
                      {name}
                    </span>
                  </label>
                ))}
              </div>
            </div>

            {/* Acoustic */}
            <div>
              <p className="text-xs text-gray-600 mb-1.5">🔊 Acoustic Signs</p>
              <div className="flex flex-wrap gap-2">
                {["Crackle", "Wheeze"].map((name) => (
                  <label
                    key={name}
                    className="flex items-center gap-1.5 text-sm cursor-pointer"
                  >
                    <input
                      type="checkbox"
                      checked={labels[name] || false}
                      onChange={() => toggleLabel(name)}
                      className="accent-cyan-500"
                    />
                    <span
                      className="rounded-full px-2 py-0.5 text-xs font-medium"
                      style={{
                        backgroundColor: (ALL_COLORS[name] ?? "#6b7280") + "20",
                        color: ALL_COLORS[name] ?? "#6b7280",
                      }}
                    >
                      {name}
                    </span>
                  </label>
                ))}
              </div>
            </div>
          </div>
        </div>

        <button
          onClick={handleApprove}
          disabled={saving}
          className="w-full rounded-lg bg-blue-600 px-3 py-2 text-sm font-semibold text-white hover:bg-blue-500 disabled:opacity-60"
        >
          {saving ? "Saving..." : "Save & Sync"}
        </button>

        {onReject && (
          <button
            onClick={handleReject}
            disabled={deleting}
            className="w-full rounded-lg border border-yellow-600/30 bg-yellow-600/10 px-3 py-2 text-sm font-semibold text-yellow-400 hover:bg-yellow-600/20 disabled:opacity-60"
          >
            {deleting ? "Moving..." : "Move to Trash"}
          </button>
        )}
      </div>
    </div>
  );
}
