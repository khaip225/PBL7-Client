import { useTrainingState } from "../../hooks/useTrainingState";
import ConnectionStatus from "./ConnectionStatus";
import TrainingProgress from "./TrainingProgress";
import MetricsDisplay from "./MetricsDisplay";
import SystemMetrics from "./SystemMetrics";
import TrainingControls from "./TrainingControls";
import { Loader2 } from "lucide-react";

export default function TrainingPage() {
  const { state, loading, error, refetch } = useTrainingState();

  if (loading && !state) {
    return (
      <div className="flex items-center gap-2 text-gray-400">
        <Loader2 size={18} className="animate-spin" />
        Đang tải trạng thái...
      </div>
    );
  }

  if (error && !state) {
    return (
      <div className="rounded-lg border border-red-600/30 bg-red-600/10 px-4 py-3 text-sm text-red-400">
        {error}
      </div>
    );
  }

  if (!state) return null;

  return (
    <div className="mx-auto max-w-4xl space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-white">FL Training</h2>
          <p className="text-sm text-gray-400 mt-1">
            Client: {state.client_name} | Modality: {state.modality || "—"}
          </p>
        </div>
        <ConnectionStatus
          connectedToServer={state.connected_to_server}
          connectedToFlower={state.connected_to_flower}
        />
      </div>

      {error && (
        <div className="rounded-lg border border-yellow-600/30 bg-yellow-600/10 px-3 py-2 text-xs text-yellow-400">
          {error}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6">
          <TrainingProgress
            currentRound={state.current_round}
            totalRounds={state.total_rounds}
            currentEpoch={state.current_epoch}
            totalEpochs={state.total_epochs}
            status={state.status}
            trainingActive={state.training_active}
          />
          <MetricsDisplay
            trainLoss={state.train_loss}
            trainAccuracy={state.train_accuracy}
            valLoss={state.val_loss}
            valAccuracy={state.val_accuracy}
            precision={state.precision}
            recall={state.recall}
            f1={state.f1}
            auc={state.auc}
          />
        </div>
        <div className="space-y-6">
          <SystemMetrics metrics={state.system} />
          <TrainingControls
            trainingActive={state.training_active}
            onStateChange={refetch}
          />
        </div>
      </div>

      {state.recent_logs.length > 0 && (
        <div className="card">
          <h3 className="text-sm font-semibold text-gray-300 mb-3">Logs</h3>
          <div className="max-h-48 overflow-y-auto space-y-1 font-mono text-xs">
            {state.recent_logs.map((log, i) => (
              <div key={i} className="flex gap-2 text-gray-400">
                <span className="text-gray-600 shrink-0">{log.timestamp}</span>
                <span
                  className={
                    log.level === "ERROR"
                      ? "text-red-400"
                      : log.level === "WARNING"
                      ? "text-yellow-400"
                      : "text-gray-300"
                  }
                >
                  {log.message}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
