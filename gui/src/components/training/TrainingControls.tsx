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
  /** null = thủ công hoặc chưa chọn; string = đang join job_id */
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);

  /** Bấm "Tham gia" → mở form xác nhận với thông số từ job */
  const handleSelectJob = (job: AvailableJob) => {
    setModality(job.task_type);
    setTotalRounds(job.num_rounds);
    setTotalEpochs(2);
    setSelectedJobId(job.job_id);
    setShowForm(true);
    setError(null);
  };

  const handleStart = async () => {
    setLoading(true);
    setError(null);
    try {
      await api.training.start({
        modality,
        total_rounds: totalRounds,
        total_epochs: totalEpochs,
        job_id: selectedJobId ?? undefined,
      });
      setShowForm(false);
      setSelectedJobId(null);
      onStateChange();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Training start error");
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
      setError(e instanceof Error ? e.message : "Training stop error");
    } finally {
      setLoading(false);
    }
  };

  const taskTypeLabel = (t: string) =>
    t === "audio" ? "Audio" : t === "image" ? "Image" : t === "alignment" ? "Multimodal" : t;

  return (
    <div className="card space-y-4">
      <h3 className="text-sm font-semibold text-gray-300">Controls</h3>

      {/* Job invitations from VPS */}
      {!trainingActive && (
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs text-gray-500">
              {jobsLoading ? "Finding jobs..." : `${availableJobs.length} job(s) available`}
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
                      <Radio size={12} /> {job.num_rounds} rounds
                    </span>
                    <span className="flex items-center gap-1">
                      <Users size={12} /> {job.joined_clients.length}/{job.min_clients}
                    </span>
                    <span className="flex items-center gap-1">
                      <Wifi size={12} /> port {job.port}
                    </span>
                  </div>

                  <div className="text-xs text-gray-500">
                    Strategy: {job.strategy}
                  </div>

                  {job.has_data ? (
                    <button
                      onClick={() => handleSelectJob(job)}
                      disabled={loading}
                      className="w-full btn-primary flex items-center justify-center gap-2 text-sm py-1.5"
                    >
                      <Play size={14} />
                      Join
                    </button>
                  ) : (
                    <div className="text-center text-xs text-red-400 py-1">
                      No data for {taskTypeLabel(job.task_type)}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}

          {availableJobs.length === 0 && !jobsLoading && (
            <div className="text-center py-3 text-xs text-gray-500">
              No training jobs from server
            </div>
          )}

          {/* Manual start button */}
          {!showForm && (
            <button
              onClick={() => { setSelectedJobId(null); setShowForm(true); }}
              className="w-full btn-primary flex items-center justify-center gap-2 text-sm py-1.5 mt-3"
            >
              <Play size={14} />
              Start manual training
            </button>
          )}

          {/* Training config form (dùng chung cho cả join job và thủ công) */}
          {showForm && (
            <div className="rounded-lg border border-gray-700 bg-gray-800/50 p-3 space-y-3 mt-3">
              {selectedJobId && (
                <div className="text-xs text-blue-400 bg-blue-900/20 rounded px-2 py-1">
                  📡 Join job from VPS (ID: {selectedJobId.slice(0, 8)}...)
                </div>
              )}

              <div>
                <label className="text-xs text-gray-400">Modality</label>
                <select
                  value={modality}
                  onChange={(e) => setModality(e.target.value)}
                  className="mt-1 w-full rounded-lg border border-gray-700 bg-gray-900 px-3 py-2 text-sm text-white"
                >
                  <option value="image">Image (X-ray)</option>
                  <option value="audio">Audio (Lung)</option>
                  <option value="alignment">Multimodal (Alignment)</option>
                </select>
              </div>
              <div>
                <label className="text-xs text-gray-400">Rounds</label>
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
                <label className="text-xs text-gray-400">Epochs per round</label>
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
                  {selectedJobId ? "Join" : "Confirm"}
                </button>
                <button
                  onClick={() => { setShowForm(false); setSelectedJobId(null); }}
                  className="rounded-lg px-4 py-2 text-sm text-gray-400 hover:text-white transition-colors"
                >
                  Cancel
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
          Stop Training
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
