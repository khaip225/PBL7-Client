interface Props {
  currentRound: number;
  totalRounds: number;
  status: string;
  trainingActive: boolean;
}

export default function TrainingProgress({ currentRound, totalRounds, status, trainingActive }: Props) {
  const pct = totalRounds > 0 ? (currentRound / totalRounds) * 100 : 0;

  return (
    <div className="card space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-300">Tiến trình</h3>
        <span
          className={`rounded-full px-2.5 py-0.5 text-xs font-medium ${
            trainingActive
              ? "bg-green-600/15 text-green-400"
              : "bg-gray-700 text-gray-400"
          }`}
        >
          {status}
        </span>
      </div>
      <div className="flex items-end gap-2">
        <span className="text-3xl font-bold text-white">{currentRound}</span>
        <span className="text-lg text-gray-500">/ {totalRounds} vòng</span>
      </div>
      <div className="h-2 rounded-full bg-gray-800 overflow-hidden">
        <div
          className="h-full rounded-full bg-blue-500 transition-all duration-700"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
