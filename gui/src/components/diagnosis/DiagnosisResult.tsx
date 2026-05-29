import type { DiagnosisResult as DR, ClassProbabilities } from "../../lib/types";
import { DISEASE_NAMES, DISEASE_COLORS, ACOUSTIC_NAMES, ACOUSTIC_COLORS } from "../../lib/types";

interface Props {
  result: DR;
}

/** Render a horizontal bar for one probability */
function ProbBar({ name, value, color }: { name: string; value: number; color: string }) {
  const pct = Math.min(value * 100, 100);
  return (
    <div className="flex items-center gap-2">
      <span className="w-28 text-xs text-gray-400 shrink-0 truncate" title={name}>
        {name}
      </span>
      <div className="flex-1 h-4 bg-gray-800 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${pct}%`, backgroundColor: color }}
        />
      </div>
      <span className="w-12 text-xs text-right text-gray-300 font-mono">
        {pct.toFixed(1)}%
      </span>
    </div>
  );
}

/** Render a group: title + ProbBar per class */
function ScoreGroup({
  title,
  scores,
  names,
  colors,
}: {
  title: string;
  scores: ClassProbabilities | null;
  names: string[];
  colors: Record<string, string>;
}) {
  if (!scores) {
    return (
      <div className="space-y-1">
        <p className="text-xs text-gray-500 font-semibold uppercase tracking-wide">{title}</p>
        <p className="text-xs text-gray-600 italic">Khong co du lieu</p>
      </div>
    );
  }
  return (
    <div className="space-y-1.5">
      <p className="text-xs text-gray-500 font-semibold uppercase tracking-wide">{title}</p>
      {names.map((name) => (
        <ProbBar
          key={name}
          name={name}
          value={scores[name] ?? 0}
          color={colors[name] ?? "#6b7280"}
        />
      ))}
    </div>
  );
}

export default function DiagnosisResult({ result }: Props) {
  const { label, confidence } = result.result;
  const { audio_scores, image_scores, fusion_scores } = result.scores;

  const isNormal = label === "Normal";
  const barColor = isNormal ? "#22c55e" : "#ef4444";

  return (
    <div className="card space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-300">Ket qua</h3>
        <span className="text-xs text-gray-500">{result.timestamp}</span>
      </div>

      {/* Primary result badge */}
      <div
        className={`flex items-center gap-3 rounded-lg px-4 py-3 ${
          isNormal
            ? "bg-green-600/10 border border-green-600/30"
            : "bg-red-600/10 border border-red-600/30"
        }`}
      >
        <span
          className={`flex h-3 w-3 rounded-full ${isNormal ? "bg-green-400" : "bg-red-400"}`}
        />
        <span className={`text-lg font-bold ${isNormal ? "text-green-400" : "text-red-400"}`}>
          {isNormal ? "Binh thuong" : "Bat thuong"}
        </span>
        <span className="text-sm text-gray-400 ml-auto">{label}</span>
      </div>

      {/* Overall confidence bar */}
      <div className="space-y-1">
        <p className="text-xs text-gray-500 font-semibold uppercase tracking-wide">
          Do tin cay
        </p>
        <ProbBar name="Confidence" value={confidence / 100} color={barColor} />
      </div>

      {/* Disease probabilities (image / fusion mode) */}
      {(result.mode === "image" || result.mode === "fusion") && (
        <ScoreGroup
          title={result.mode === "fusion" ? "🫁 Benh phoi (Fusion Ontology)" : "🫁 Benh phoi"}
          scores={result.mode === "fusion" ? fusion_scores : image_scores}
          names={[...DISEASE_NAMES, ...(result.mode === "fusion" ? ["Normal"] : [])]}
          colors={DISEASE_COLORS}
        />
      )}

      {/* Acoustic attributes (audio / fusion mode) */}
      {(result.mode === "audio" || result.mode === "fusion") && (
        <ScoreGroup
          title="🔊 Thuoc tinh am thanh"
          scores={audio_scores}
          names={ACOUSTIC_NAMES}
          colors={ACOUSTIC_COLORS}
        />
      )}

      {/* Saved files info */}
      <div className="border-t border-gray-700 pt-3 mt-2">
        <p className="text-xs text-gray-500 mb-1">Files da luu</p>
        <div className="text-xs text-gray-400 space-y-0.5">
          {result.saved.audio_dest && (
            <p className="truncate">🎵 {result.saved.audio_dest.split(/[/\\]/).pop()}</p>
          )}
          {result.saved.image_dest && (
            <p className="truncate">📸 {result.saved.image_dest.split(/[/\\]/).pop()}</p>
          )}
        </div>
      </div>
    </div>
  );
}
