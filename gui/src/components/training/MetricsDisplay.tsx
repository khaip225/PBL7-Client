import { TrendingDown, Target } from "lucide-react";
import StatCard from "../shared/StatCard";

interface Props {
  loss: number | null;
  accuracy: number | null;
}

export default function MetricsDisplay({ loss, accuracy }: Props) {
  return (
    <div className="grid grid-cols-2 gap-4">
      <StatCard
        label="Loss"
        value={loss != null ? loss.toFixed(4) : "—"}
        icon={<TrendingDown size={18} />}
        color="red"
      />
      <StatCard
        label="Accuracy"
        value={accuracy != null ? `${(accuracy * 100).toFixed(1)}%` : "—"}
        icon={<Target size={18} />}
        color="green"
      />
    </div>
  );
}
