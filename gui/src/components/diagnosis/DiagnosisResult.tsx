import type { DiagnosisResult as DR, ClassProbabilities } from "../../lib/types";
import { DISEASE_NAMES, DISEASE_COLORS, ACOUSTIC_NAMES, ACOUSTIC_COLORS, ALL_COLORS, ALL_LABELS } from "../../lib/types";

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
  const { labels, confidence } = result.result;
  const { audio_scores, image_scores, fusion_scores } = result.scores;

  const allNormal = labels.length === 1 && labels[0] === "Normal";

  return (
    <div className="card space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-300">Ket qua</h3>
        <span className="text-xs text-gray-500">{result.timestamp}</span>
      </div>

      {/* Detected labels — multi-label badges */}
      <div className="rounded-lg border border-gray-700 bg-gray-900/40 px-4 py-3">
        <p className="text-xs text-gray-500 mb-2 font-semibold uppercase tracking-wide">
          Phat hien
        </p>
        {allNormal ? (
          <span className="inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-sm font-medium bg-green-600/15 text-green-400 border border-green-600/30">
            <span className="flex h-2 w-2 rounded-full bg-green-400" />
            Binh thuong
          </span>
        ) : (
          <div className="flex flex-wrap gap-2">
            {labels.map((label) => (
              <span
                key={label}
                className="inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-sm font-medium border"
                style={{
                  backgroundColor: (ALL_COLORS[label] ?? "#6b7280") + "20",
                  color: ALL_COLORS[label] ?? "#6b7280",
                  borderColor: (ALL_COLORS[label] ?? "#6b7280") + "40",
                }}
              >
                <span
                  className="flex h-2 w-2 rounded-full"
                  style={{ backgroundColor: ALL_COLORS[label] ?? "#6b7280" }}
                />
                {label}
              </span>
            ))}
          </div>
        )}
      </div>

      {/* Overall confidence bar */}
      <div className="space-y-1">
        <p className="text-xs text-gray-500 font-semibold uppercase tracking-wide">
          Do tin cay
        </p>
        <ProbBar
          name="Confidence"
          value={confidence / 100}
          color={allNormal ? "#22c55e" : "#ef4444"}
        />
      </div>

      {/* Disease probabilities (image / fusion / audio→image inferred) */}
      {(result.mode === "image" || result.mode === "fusion") && (
        <ScoreGroup
          title={result.mode === "fusion" ? "🫁 Bệnh phổi (Fusion Ontology)" : "🫁 Bệnh phổi"}
          scores={result.mode === "fusion" ? fusion_scores : image_scores}
          names={[...DISEASE_NAMES, ...(result.mode === "fusion" ? ["Normal"] : [])]}
          colors={DISEASE_COLORS}
        />
      )}

      {/* Cross-modal: audio → inferred disease (audio-only mode) */}
      {result.mode === "audio" && image_scores && (
        <ScoreGroup
          title="🫁 Bệnh phổi (Suy luận từ âm thanh)"
          scores={image_scores}
          names={DISEASE_NAMES}
          colors={DISEASE_COLORS}
        />
      )}

      {/* Acoustic attributes (audio / fusion / image→acoustic inferred) */}
      {(result.mode === "audio" || result.mode === "fusion") && (
        <ScoreGroup
          title="🔊 Thuộc tính âm thanh"
          scores={audio_scores}
          names={ACOUSTIC_NAMES}
          colors={ACOUSTIC_COLORS}
        />
      )}

      {/* Cross-modal: image → inferred acoustic (image-only mode) */}
      {result.mode === "image" && audio_scores && (
        <ScoreGroup
          title="🔊 Thuộc tính âm thanh (Suy luận từ ảnh)"
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
