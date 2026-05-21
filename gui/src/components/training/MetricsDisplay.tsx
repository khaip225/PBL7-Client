import { TrendingDown, Target, Gauge, Activity } from "lucide-react";
import StatCard from "../shared/StatCard";

interface Props {
  trainLoss: number | null;
  trainAccuracy: number | null;
  valLoss: number | null;
  valAccuracy: number | null;
  precision: number | null;
  recall: number | null;
  f1: number | null;
  auc: number | null;
}

const formatPct = (value: number | null) =>
  value != null ? `${(value * 100).toFixed(1)}%` : "—";

export default function MetricsDisplay({
  trainLoss,
  trainAccuracy,
  valLoss,
  valAccuracy,
  precision,
  recall,
  f1,
  auc,
}: Props) {
  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
      <StatCard
        label="Train Loss"
        value={trainLoss != null ? trainLoss.toFixed(4) : "—"}
        icon={<TrendingDown size={18} />}
        color="red"
      />
      <StatCard
        label="Train Acc"
        value={formatPct(trainAccuracy)}
        icon={<Target size={18} />}
        color="green"
      />
      <StatCard
        label="Val Loss"
        value={valLoss != null ? valLoss.toFixed(4) : "—"}
        icon={<TrendingDown size={18} />}
        color="orange"
      />
      <StatCard
        label="Val Acc"
        value={formatPct(valAccuracy)}
        icon={<Target size={18} />}
        color="teal"
      />
      <StatCard
        label="Precision"
        value={formatPct(precision)}
        icon={<Gauge size={18} />}
        color="blue"
      />
      <StatCard
        label="Recall"
        value={formatPct(recall)}
        icon={<Gauge size={18} />}
        color="purple"
      />
      <StatCard
        label="F1"
        value={formatPct(f1)}
        icon={<Activity size={18} />}
        color="pink"
      />
      <StatCard
        label="AUC"
        value={auc != null ? auc.toFixed(4) : "—"}
        icon={<Activity size={18} />}
        color="indigo"
      />
    </div>
  );
}
