import { useState } from "react";
import { Play, Square, Loader2 } from "lucide-react";
import { api } from "../../lib/api";

interface Props {
  trainingActive: boolean;
  onStateChange: () => void;
}

export default function TrainingControls({ trainingActive, onStateChange }: Props) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showForm, setShowForm] = useState(false);
  const [modality, setModality] = useState("image");
  const [totalRounds, setTotalRounds] = useState(5);
  const [totalEpochs, setTotalEpochs] = useState(2);

  const handleStart = async () => {
    setLoading(true);
    setError(null);
    try {
      await api.training.start({
        modality,
        total_rounds: totalRounds,
        total_epochs: totalEpochs,
      });
      setShowForm(false);
      onStateChange();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Lỗi khởi động training");
    } finally {
      setLoading(false);
    }
  };

  const handleStop = async () => {
    setLoading(true);
    setError(null);
    try {
      await api.training.stop();
      onStateChange();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Lỗi dừng training");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card space-y-4">
      <h3 className="text-sm font-semibold text-gray-300">Điều khiển</h3>

      {!trainingActive && !showForm && (
        <button onClick={() => setShowForm(true)} className="btn-primary flex items-center gap-2">
          <Play size={16} />
          Bắt đầu Training
        </button>
      )}

      {!trainingActive && showForm && (
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="text-xs text-gray-400">Modality</label>
              <select
                value={modality}
                onChange={(e) => setModality(e.target.value)}
                className="mt-1 w-full rounded-lg border border-gray-700 bg-gray-900 px-3 py-2 text-sm text-white"
              >
                <option value="image">Image</option>
                <option value="audio">Audio</option>
              </select>
            </div>
            <div>
              <label className="text-xs text-gray-400">Số vòng</label>
              <input
                type="number"
                min={1}
                max={50}
                value={totalRounds}
                onChange={(e) => setTotalRounds(Number(e.target.value))}
                className="mt-1 w-full rounded-lg border border-gray-700 bg-gray-900 px-3 py-2 text-sm text-white"
              />
            </div>
          </div>
          <div>
            <label className="text-xs text-gray-400">Epoch mỗi vòng</label>
            <input
              type="number"
              min={1}
              max={20}
              value={totalEpochs}
              onChange={(e) => setTotalEpochs(Number(e.target.value))}
              className="mt-1 w-full rounded-lg border border-gray-700 bg-gray-900 px-3 py-2 text-sm text-white"
            />
          </div>
          <div className="flex gap-3">
            <button
              onClick={handleStart}
              disabled={loading}
              className="btn-primary flex items-center gap-2"
            >
              {loading && <Loader2 size={14} className="animate-spin" />}
              Xác nhận
            </button>
            <button
              onClick={() => setShowForm(false)}
              className="rounded-lg px-4 py-2 text-sm text-gray-400 hover:text-white transition-colors"
            >
              Hủy
            </button>
          </div>
        </div>
      )}

      {trainingActive && (
        <button onClick={handleStop} disabled={loading} className="btn-danger flex items-center gap-2">
          {loading && <Loader2 size={14} className="animate-spin" />}
          <Square size={14} />
          Dừng Training
        </button>
      )}

      {error && (
        <div className="rounded-lg border border-red-600/30 bg-red-600/10 px-3 py-2 text-xs text-red-400">
          {error}
        </div>
      )}
    </div>
  );
}
