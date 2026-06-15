import MetricBar from "../shared/MetricBar";
import type { SystemMetrics as SM } from "../../lib/types";

interface Props {
  metrics: SM;
}

export default function SystemMetrics({ metrics }: Props) {
  return (
    <div className="card space-y-4">
      <h3 className="text-sm font-semibold text-gray-300">System</h3>
      <div className="space-y-3">
        <MetricBar label="CPU" value={metrics.cpu_percent} color="blue" />
        <MetricBar label="RAM" value={metrics.ram_percent} color="yellow" />
        <MetricBar label="GPU" value={metrics.gpu_percent} color="green" />
        <MetricBar label="Disk" value={metrics.disk_percent} color="gray" />
      </div>
      <div className="flex justify-between text-xs text-gray-500 pt-1 border-t border-gray-800">
        <span>GPU Temp: {metrics.gpu_temp != null ? `${metrics.gpu_temp}°C` : "—"}</span>
        <span>Latency: {metrics.latency_ms}ms</span>
      </div>
    </div>
  );
}
