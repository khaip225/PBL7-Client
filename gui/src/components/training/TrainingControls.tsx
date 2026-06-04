import { useState } from "react";
import { Play, Square, Loader2, Users, Radio, Wifi } from "lucide-react";
import { api } from "../../lib/api";
import type { AvailableJob } from "../../lib/types";

interface Props {
  trainingActive: boolean;
  onStateChange: () => void;
  availableJobs: AvailableJob[];
  jobsLoading: boolean;
}

export default function TrainingControls({ trainingActive, onStateChange, availableJobs, jobsLoading }: Props) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showForm, setShowForm] = useState(false);
  const [modality, setModality] = useState("image");
  const [totalRounds, setTotalRounds] = useState(5);
  const [totalEpochs, setTotalEpochs] = useState(2);
  const [joiningId, setJoiningId] = useState<string | null>(null);

  const handleJoin = async (job: AvailableJob) => {
    setLoading(true);
    setJoiningId(job.job_id);
    setError(null);
    try {
      await api.training.start({
        modality: job.task_type,
        total_rounds: job.num_rounds,
        total_epochs: totalEpochs,
        job_id: job.job_id,
      });
      onStateChange();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Lỗi tham gia training");
    } finally {
      setLoading(false);
      setJoiningId(null);
    }
  };

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
      setError(e instanceof Error ? e.message : "Lỗi bắt đầu training");
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

  const taskTypeLabel = (t: string) =>
    t === "audio" ? "Âm thanh" : t === "image" ? "Hình ảnh" : t === "alignment" ? "Đa phương thức" : t;

  return (
    <div className="card space-y-4">
      <h3 className="text-sm font-semibold text-gray-300">Điều khiển</h3>

      {/* Job invitations from VPS */}
      {!trainingActive && (
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs text-gray-500">
              {jobsLoading ? "Đang tìm job..." : `${availableJobs.length} job khả dụng`}
            </span>
            {jobsLoading && <Loader2 size={12} className="animate-spin text-gray-500" />}
          </div>

          {availableJobs.length > 0 && (
            <div className="space-y-2 max-h-80 overflow-y-auto">
              {availableJobs.map((job) => (
                <div
                  key={job.job_id}
                  className={`rounded-lg border p-3 space-y-2 ${
                    job.has_data
                      ? "border-gray-700 bg-gray-800/50"
                      : "border-gray-800 bg-gray-900/30 opacity-50"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium text-white">{job.name}</span>
                    <span className={`px-2 py-0.5 rounded text-xs ${
                      job.task_type === "audio" ? "bg-purple-900/30 text-purple-400"
                        : job.task_type === "alignment" ? "bg-emerald-900/30 text-emerald-400"
                        : "bg-blue-900/30 text-blue-400"
                    }`}>
                      {taskTypeLabel(job.task_type)}
                    </span>
                  </div>

                  <div className="flex items-center gap-3 text-xs text-gray-400">
                    <span className="flex items-center gap-1">
                      <Radio size={12} /> {job.num_rounds} vòng
                    </span>
                    <span className="flex items-center gap-1">
                      <Users size={12} /> {job.joined_clients.length}/{job.min_clients}
                    </span>
                    <span className="flex items-center gap-1">
                      <Wifi size={12} /> port {job.port}
                    </span>
                  </div>

                  <div className="text-xs text-gray-500">
                    Chiến lược: {job.strategy}
                  </div>

                  {job.has_data ? (
                    <button
                      onClick={() => handleJoin(job)}
                      disabled={loading}
                      className="w-full btn-primary flex items-center justify-center gap-2 text-sm py-1.5"
                    >
                      {loading && joiningId === job.job_id ? (
                        <Loader2 size={14} className="animate-spin" />
                      ) : (
                        <Play size={14} />
                      )}
                      Tham gia
                    </button>
                  ) : (
                    <div className="text-center text-xs text-red-400 py-1">
                      Không có dữ liệu {taskTypeLabel(job.task_type)}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}

          {availableJobs.length === 0 && !jobsLoading && (
            <div className="text-center py-3 text-xs text-gray-500">
              Không có job training nào từ server
            </div>
          )}

          {/* Manual start button */}
          {!showForm && (
            <button
              onClick={() => setShowForm(true)}
              className="w-full btn-primary flex items-center justify-center gap-2 text-sm py-1.5 mt-3"
            >
              <Play size={14} />
              Bắt đầu training thủ công
            </button>
          )}

          {/* Manual training form */}
          {showForm && (
            <div className="rounded-lg border border-gray-700 bg-gray-800/50 p-3 space-y-3 mt-3">
              <div>
                <label className="text-xs text-gray-400">Modality</label>
                <select
                  value={modality}
                  onChange={(e) => setModality(e.target.value)}
                  className="mt-1 w-full rounded-lg border border-gray-700 bg-gray-900 px-3 py-2 text-sm text-white"
                >
                  <option value="image">Hình ảnh (X-quang)</option>
                  <option value="audio">Âm thanh (Phổi)</option>
                  <option value="alignment">Đa phương thức (Alignment)</option>
                </select>
              </div>
              <div>
                <label className="text-xs text-gray-400">Số vòng (rounds)</label>
                <input
                  type="number"
                  min={1}
                  max={50}
                  value={totalRounds}
                  onChange={(e) => setTotalRounds(Number(e.target.value))}
                  className="mt-1 w-full rounded-lg border border-gray-700 bg-gray-900 px-3 py-2 text-sm text-white"
                />
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
        </div>
      )}

      {/* Stop button */}
      {trainingActive && (
        <button onClick={handleStop} disabled={loading} className="btn-danger flex items-center gap-2 w-full justify-center">
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
