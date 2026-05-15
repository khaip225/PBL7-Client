import type { ReactNode } from "react";

interface Props {
  label: string;
  value: string | number;
  unit?: string;
  icon?: ReactNode;
  color?: "blue" | "green" | "red" | "yellow" | "gray";
}

const colorMap = {
  blue: "text-blue-400 bg-blue-600/10",
  green: "text-green-400 bg-green-600/10",
  red: "text-red-400 bg-red-600/10",
  yellow: "text-yellow-400 bg-yellow-600/10",
  gray: "text-gray-400 bg-gray-600/10",
};

export default function StatCard({ label, value, unit, icon, color = "blue" }: Props) {
  return (
    <div className="card flex items-center gap-4">
      {icon && (
        <div className={`flex h-10 w-10 items-center justify-center rounded-lg ${colorMap[color]}`}>
          {icon}
        </div>
      )}
      <div>
        <p className="text-xs text-gray-500 uppercase tracking-wide">{label}</p>
        <p className="text-lg font-semibold text-white">
          {value}
          {unit && <span className="text-sm text-gray-400 ml-0.5">{unit}</span>}
        </p>
      </div>
    </div>
  );
}
