import type { DiagnosisResult as DR } from "../../lib/types";

interface Props {
  result: DR;
}

export default function DiagnosisResult({ result }: Props) {
  const isAbnormal = result.result.label !== "normal";
  const { audio_score, image_score, fusion_score } = result.scores;

  const scoreLabel = (s: number | null) =>
    s != null ? `${(s * 100).toFixed(1)}%` : "—";

  return (
    <div className="card space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-300">Kết quả</h3>
        <span className="text-xs text-gray-500">{result.timestamp}</span>
      </div>

      <div
        className={`flex items-center gap-3 rounded-lg px-4 py-3 ${
          isAbnormal
            ? "bg-red-600/10 border border-red-600/30"
            : "bg-green-600/10 border border-green-600/30"
        }`}
      >
        <span
          className={`flex h-3 w-3 rounded-full ${
            isAbnormal ? "bg-red-400" : "bg-green-400"
          }`}
        />
        <span
          className={`text-lg font-bold ${
            isAbnormal ? "text-red-400" : "text-green-400"
          }`}
        >
          {isAbnormal ? "Bất thường" : "Bình thường"}
        </span>
        <span className="text-sm text-gray-400 ml-auto">
          {result.result.label}
        </span>
      </div>

      <div className="grid grid-cols-3 gap-3 text-center">
        {result.mode !== "image" && (
          <div>
            <p className="text-xs text-gray-500">Audio Score</p>
            <p className="text-lg font-semibold text-white">{scoreLabel(audio_score)}</p>
          </div>
        )}
        {result.mode !== "audio" && (
          <div>
            <p className="text-xs text-gray-500">Image Score</p>
            <p className="text-lg font-semibold text-white">{scoreLabel(image_score)}</p>
          </div>
        )}
        {result.mode === "fusion" && (
          <div>
            <p className="text-xs text-gray-500">Fusion Score</p>
            <p className="text-lg font-semibold text-blue-400">{scoreLabel(fusion_score)}</p>
          </div>
        )}
      </div>
    </div>
  );
}
