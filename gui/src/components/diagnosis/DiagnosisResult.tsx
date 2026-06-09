import type {
  DiagnosisResult as DR,
  CrossModalResult,
  RetrievalItem,
  LateFusionResultDetail,
} from "../../lib/types";
import {
  DISEASE_NAMES,
  DISEASE_COLORS,
  ACOUSTIC_NAMES,
  ACOUSTIC_COLORS,
  ALL_COLORS,
} from "../../lib/types";
import { Lightbulb, Search, Brain, Layers, Maximize2, Volume2 } from "lucide-react";
import { useState } from "react";

interface Props {
  result: DR;
  compact?: boolean;
}

function cn(...c: (string | false | undefined)[]) { return c.filter(Boolean).join(" "); }

function confidenceColor(level: string): string {
  switch (level) {
    case "Rất cao": return "#22c55e";
    case "Cao": return "#3b82f6";
    case "Trung bình": return "#f97316";
    default: return "#ef4444";
  }
}

function MiniBar({ name, value, color }: { name: string; value: number; color: string }) {
  const pct = Math.round(value * 100);
  return (
    <div className="flex items-center gap-1.5 text-[11px]">
      <span className="w-28 text-gray-400 truncate">{name}</span>
      <div className="flex-1 h-1.5 bg-gray-700 rounded-full overflow-hidden">
        <div className="h-full rounded-full transition-all duration-500"
          style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
      <span className="w-9 text-right text-gray-300 font-mono text-[11px]">{pct}%</span>
    </div>
  );
}

/* ── Lightbox toàn màn ──────────────────────────────────────────── */
function Lightbox({ src, onClose }: { src: string; onClose: () => void }) {
  return (
    <div className="fixed inset-0 z-50 bg-black/95 flex items-center justify-center cursor-pointer"
      onClick={onClose}>
      <img src={src} alt="Zoom" className="max-w-[94vw] max-h-[94vh] object-contain rounded" />
    </div>
  );
}

/* ── 5 Section ──────────────────────────────────────────────────── */
export default function DiagnosisResult({ result, compact }: Props) {
  const { labels } = result.result;
  const { audio_scores, image_scores } = result.scores;
  const allNormal = labels.length === 1 && labels[0] === "Normal";
  const isImage = result.mode === "image";
  const isAudio = result.mode === "audio";

  return (
    <div className={cn("flex flex-col gap-3", compact && "h-full")}>
      {/* Tiêu đề nhỏ */}
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-white">
          Kết quả — {isImage ? "Ảnh X-quang" : isAudio ? "Âm thanh" : "Fusion"}
        </h3>
        <span className="text-[10px] text-gray-500">{result.timestamp}</span>
      </div>

      {/* ===== HĐ1: Phân loại + HĐ5: Kết luận (2 cột) ============ */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
        {/* HĐ1 */}
        <div className="rounded-lg border border-blue-500/25 bg-blue-500/5 p-3">
          <div className="flex items-center gap-2 mb-2">
            <span className="w-5 h-5 rounded-full bg-blue-500/20 text-blue-400 text-[10px] font-bold flex items-center justify-center">1</span>
            <span className="text-[11px] font-semibold text-gray-300 flex items-center gap-1"><Brain size={12} /> Phân loại</span>
          </div>

          {/* Tags */}
          <div className="flex flex-wrap gap-1 mb-2">
            {allNormal ? (
              <span className="inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-medium bg-green-600/15 text-green-400 border border-green-600/30">
                <span className="w-1.5 h-1.5 rounded-full bg-green-400" /> Bình thường
              </span>
            ) : (
              labels.map((lbl) => (
                <span key={lbl} className="inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-medium border"
                  style={{ backgroundColor: (ALL_COLORS[lbl] ?? "#6b7280") + "20",
                    color: ALL_COLORS[lbl], borderColor: (ALL_COLORS[lbl] ?? "#6b7280") + "40" }}>
                  <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: ALL_COLORS[lbl] }} />
                  {lbl}
                </span>
              ))
            )}
          </div>

          {/* Bars */}
          {isImage && image_scores && DISEASE_NAMES.map(n =>
            <MiniBar key={n} name={n} value={image_scores[n] ?? 0} color={DISEASE_COLORS[n]} />)}
          {isImage && audio_scores &&
            <div className="text-[10px] text-gray-500 mt-1 mb-0.5">🔊 Âm thanh (suy luận):</div>}
          {isImage && audio_scores && ACOUSTIC_NAMES.map(n =>
            <MiniBar key={n} name={n} value={audio_scores[n] ?? 0} color={ACOUSTIC_COLORS[n]} />)}

          {isAudio && audio_scores && ACOUSTIC_NAMES.map(n =>
            <MiniBar key={n} name={n} value={audio_scores[n] ?? 0} color={ACOUSTIC_COLORS[n]} />)}
          {isAudio && image_scores &&
            <div className="text-[10px] text-gray-500 mt-1 mb-0.5">🫁 Bệnh (suy luận):</div>}
          {isAudio && image_scores && DISEASE_NAMES.map(n =>
            <MiniBar key={n} name={n} value={image_scores[n] ?? 0} color={DISEASE_COLORS[n]} />)}
        </div>

        {/* HĐ5 */}
        {result.late_fusion && (
          <div className={cn("rounded-lg border p-3",
            result.late_fusion.is_normal
              ? "border-green-500/25 bg-green-500/5"
              : "border-orange-500/25 bg-orange-500/5")}>
            <div className="flex items-center gap-2 mb-2">
              <span className="w-5 h-5 rounded-full bg-orange-500/20 text-orange-400 text-[10px] font-bold flex items-center justify-center">5</span>
              <span className="text-[11px] font-semibold text-gray-300 flex items-center gap-1"><Layers size={12} /> Kết luận Late Fusion</span>
            </div>
            <LateFusionBox data={result.late_fusion} />
          </div>
        )}
      </div>

      {/* ===== HĐ2: Zero-shot ===================================== */}
      {result.cross_modal && (
        <div className="rounded-lg border border-cyan-500/25 bg-cyan-500/5 p-3">
          <div className="flex items-center gap-2 mb-2">
            <span className="w-5 h-5 rounded-full bg-cyan-500/20 text-cyan-400 text-[10px] font-bold flex items-center justify-center">2</span>
            <span className="text-[11px] font-semibold text-gray-300 flex items-center gap-1"><Lightbulb size={12} /> Zero-shot Liên Modal</span>
          </div>
          <div className="rounded bg-cyan-500/10 border border-cyan-500/20 p-2.5">
            <p className="text-[11px] text-cyan-300 leading-relaxed">{result.cross_modal.message}</p>
            <div className="mt-1.5 flex flex-wrap gap-x-3 gap-y-0.5">
              {Object.entries(result.cross_modal.scores).map(([name, val]) =>
                <span key={name} className="text-[11px] text-cyan-200 font-mono">{name}: {Math.round(val * 100)}%</span>)}
            </div>
          </div>
        </div>
      )}

      {/* ===== HĐ3: Retrieval ===================================== */}
      {result.retrieval && result.retrieval.length > 0 && (
        <div className="rounded-lg border border-purple-500/25 bg-purple-500/5 p-3">
          <div className="flex items-center gap-2 mb-2">
            <span className="w-5 h-5 rounded-full bg-purple-500/20 text-purple-400 text-[10px] font-bold flex items-center justify-center">3</span>
            <span className="text-[11px] font-semibold text-gray-300 flex items-center gap-1"><Search size={12} /> Truy xuất bằng chứng</span>
          </div>
          {isImage ? (
            <div className="space-y-1.5">
              {result.retrieval.map((item, i) => <RetrievalAudioRow key={i} item={item} />)}
            </div>
          ) : (
            <div className="flex gap-2 overflow-x-auto pb-1">
              {result.retrieval.map((item, i) => <RetrievalImageThumb key={i} item={item} />)}
            </div>
          )}
        </div>
      )}

      {/* ===== HĐ4: XAI ========================================== */}
      {(result.heatmap_path || result.attention_map_path) && (
        <div className="rounded-lg border border-orange-500/25 bg-orange-500/5 p-3">
          <div className="flex items-center gap-2 mb-2">
            <span className="w-5 h-5 rounded-full bg-orange-500/20 text-orange-400 text-[10px] font-bold flex items-center justify-center">4</span>
            <span className="text-[11px] font-semibold text-gray-300 flex items-center gap-1"><Search size={12} /> Giải thích (XAI)</span>
          </div>
          <div className="grid grid-cols-2 gap-2">
            {result.heatmap_path &&
              <XaiThumb label="Grad-CAM"
                url={`/api/heatmap?path=${encodeURIComponent(result.heatmap_path)}`} />}
            {result.attention_map_path &&
              <XaiThumb label="Audio Attention"
                url={`/api/heatmap?path=${encodeURIComponent(result.attention_map_path)}`} />}
          </div>
        </div>
      )}
    </div>
  );
}

/* ── LateFusion Box ──────────────────────────────────────────────── */
function LateFusionBox({ data }: { data: LateFusionResultDetail }) {
  const cc = confidenceColor(data.confidence_level);
  return (
    <div className="flex items-start justify-between gap-3">
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-semibold"
            style={{ backgroundColor: cc + "20", color: cc }}>
            {data.is_normal ? "✅ Bình thường" : "⚠️ " + data.primary_diagnosis}
          </span>
          <span className="text-[10px] text-gray-500">{data.agreement}</span>
        </div>
        {data.fusion_scores && (
          <div className="mt-1.5 flex flex-wrap gap-x-2 gap-y-0.5">
            {Object.entries(data.fusion_scores).map(([name, val]) =>
              <span key={name} className="text-[10px] text-gray-400">
                {name}: <span className="text-gray-200 font-mono">{Math.round(val * 100)}%</span>
              </span>)}
          </div>
        )}
      </div>
      <div className="text-right shrink-0">
        <p className="text-xl font-bold" style={{ color: cc }}>{Math.round(data.confidence * 100)}%</p>
        <span className="inline-block text-[10px] rounded-full px-2 py-0.5 font-medium"
          style={{ backgroundColor: cc + "20", color: cc, border: `1px solid ${cc}40` }}>
          {data.confidence_level}
        </span>
      </div>
    </div>
  );
}

/* ── Retrieval Audio Row ────────────────────────────────────────── */
function RetrievalAudioRow({ item }: { item: RetrievalItem }) {
  const fileUrl = `/api/retrieval/file?path=${encodeURIComponent(item.file_path)}`;
  const lbl = item.disease_label || item.acoustic_label || "";
  return (
    <div className="flex items-center gap-2 rounded bg-gray-800/60 border border-gray-700/50 p-2">
      <div className="w-8 h-8 rounded-full bg-cyan-500/15 flex items-center justify-center shrink-0">
        <Volume2 size={14} className="text-cyan-400" />
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-[11px] text-gray-300 truncate" title={item.file_name}>{item.file_name}</p>
        <div className="flex items-center gap-1.5">
          {lbl && <span className="text-[10px] rounded px-1 py-0.5 font-medium"
            style={{ backgroundColor: (ALL_COLORS[lbl] ?? "#6b7280") + "20", color: ALL_COLORS[lbl] }}>{lbl}</span>}
          <span className="text-[10px] text-gray-500">{Math.round(item.similarity * 100)}% match</span>
        </div>
      </div>
      <audio controls className="h-6 w-[100px] shrink-0">
        <source src={fileUrl} />
      </audio>
    </div>
  );
}

/* ── Retrieval Image Thumb ──────────────────────────────────────── */
function RetrievalImageThumb({ item }: { item: RetrievalItem }) {
  const [lb, setLb] = useState(false);
  const fileUrl = `/api/retrieval/file?path=${encodeURIComponent(item.file_path)}`;
  return (
    <>
      <div className="shrink-0 w-[100px] group cursor-pointer" onClick={() => setLb(true)}>
        <div className="w-full h-[75px] rounded-lg overflow-hidden bg-black border border-gray-600">
          <img src={fileUrl} alt={item.file_name}
            className="w-full h-full object-cover hover:scale-105 transition-transform" />
        </div>
        <p className="text-[10px] text-gray-500 text-center mt-0.5 truncate" title={item.file_name}>
          {item.file_name.length > 16 ? item.file_name.slice(0, 14) + "…" : item.file_name}
        </p>
        <p className="text-[10px] text-purple-400 text-center font-mono">{Math.round(item.similarity * 100)}%</p>
      </div>
      {lb && <Lightbox src={fileUrl} onClose={() => setLb(false)} />}
    </>
  );
}

/* ── XAI Thumb ──────────────────────────────────────────────────── */
function XaiThumb({ label, url }: { label: string; url: string }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <div className="rounded-lg overflow-hidden border border-gray-600 bg-black cursor-pointer group relative"
        onClick={() => setOpen(true)}>
        <img src={url} alt={label} className="w-full h-28 object-contain" />
        <div className="absolute top-1 right-1 bg-black/60 rounded p-0.5 opacity-0 group-hover:opacity-100 transition">
          <Maximize2 size={12} className="text-white" />
        </div>
        <p className="absolute bottom-0 inset-x-0 bg-black/60 text-center text-[10px] text-gray-300 py-0.5">{label}</p>
      </div>
      {open && <Lightbox src={url} onClose={() => setOpen(false)} />}
    </>
  );
}
