interface Props {
  label: string;
  value: number;
  max?: number;
  color?: "blue" | "green" | "red" | "yellow";
}

const colorMap = {
  blue: "bg-blue-500",
  green: "bg-green-500",
  red: "bg-red-500",
  yellow: "bg-yellow-500",
};

export default function MetricBar({ label, value, max = 100, color = "blue" }: Props) {
  const pct = Math.min((value / max) * 100, 100);

  return (
    <div>
      <div className="flex justify-between text-xs mb-1">
        <span className="text-gray-400">{label}</span>
        <span className="text-gray-300">{pct.toFixed(0)}%</span>
      </div>
      <div className="h-2 rounded-full bg-gray-800 overflow-hidden">
        <div
          className={`h-full rounded-full ${colorMap[color]} transition-all duration-500`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
