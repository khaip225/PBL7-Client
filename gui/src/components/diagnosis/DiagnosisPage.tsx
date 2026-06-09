import { useState, useEffect } from "react";
import { useDiagnosis } from "../../hooks/useDiagnosis";
import DiagnosisResult from "./DiagnosisResult";
import { Loader2, Maximize2, X, Upload, FileSearch, Image, Mic } from "lucide-react";

type SubTab = "upload" | "result";

function heatmapUrl(path: string | null | undefined): string | null {
  if (!path) return null;
  return `/api/heatmap?path=${encodeURIComponent(path)}`;
}

export default function DiagnosisPage() {
  const d = useDiagnosis();
  const [subTab, setSubTab] = useState<SubTab>("upload");
  const [lightboxSrc, setLightboxSrc] = useState<string | null>(null);
  const [imgPreview, setImgPreview] = useState<string | null>(null);
  const [audioPreview, setAudioPreview] = useState<string | null>(null);

  useEffect(() => {
    if (d.result) setSubTab("result");
  }, [d.result]);

  useEffect(() => {
    if (d.imageFile) {
      const url = URL.createObjectURL(d.imageFile);
      setImgPreview(url);
      return () => URL.revokeObjectURL(url);
    }
    setImgPreview(null);
  }, [d.imageFile]);

  useEffect(() => {
    if (d.audioFile) {
      const url = URL.createObjectURL(d.audioFile);
      setAudioPreview(url);
      return () => URL.revokeObjectURL(url);
    }
    setAudioPreview(null);
  }, [d.audioFile]);

  const hmUrl = d.result ? heatmapUrl(d.result.heatmap_path) : null;
  const attUrl = d.result ? heatmapUrl(d.result.attention_map_path) : null;
  const isImage = d.mode === "image";
  const isAudio = d.mode === "audio";
  const isFusion = d.mode === "fusion";

  return (
    <div className="space-y-4">
      {/* ── Header ───────────────────────────────────────────────── */}
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div>
          <h2 className="text-xl font-bold text-white">Chẩn đoán Bệnh Lý Phổi</h2>
          <p className="text-sm text-gray-400 mt-0.5">
            Upload ảnh X-quang hoặc audio để phân tích đa phương thức
          </p>
        </div>

        <div className="flex gap-1 bg-gray-800 rounded-lg p-1">
          <button
            onClick={() => setSubTab("upload")}
            className={`flex items-center gap-1.5 px-4 py-1.5 rounded-md text-sm font-medium transition ${
              subTab === "upload" ? "bg-gray-700 text-white shadow" : "text-gray-400 hover:text-gray-200"
            }`}
          >
            <Upload size={14} /> Upload
          </button>
          <button
            onClick={() => setSubTab("result")}
            disabled={!d.result}
            className={`flex items-center gap-1.5 px-4 py-1.5 rounded-md text-sm font-medium transition ${
              !d.result ? "text-gray-600 cursor-not-allowed"
                : subTab === "result" ? "bg-gray-700 text-white shadow" : "text-gray-400 hover:text-gray-200"
            }`}
          >
            <FileSearch size={14} /> Kết quả chi tiết
          </button>
        </div>
      </div>

      {d.error && (
        <div className="rounded-lg border border-red-600/30 bg-red-600/10 px-4 py-3 text-sm text-red-400">
          {d.error}
        </div>
      )}

      {/* ═══ TAB: Upload (căn giữa, max-width) ═════════════════════ */}
      {subTab === "upload" && (
        <div className="max-w-lg flex flex-col gap-4">
          {/* Mode selector */}
          <div className="flex gap-1 bg-gray-800/80 rounded-lg p-1 w-fit">
            {(["image", "audio", "fusion"] as const).map((m) => (
              <button
                key={m}
                onClick={() => d.setMode(m)}
                disabled={d.loading}
                className={`flex items-center gap-1.5 px-4 py-1.5 rounded-md text-sm font-medium transition ${
                  d.mode === m ? "bg-blue-600 text-white shadow" : "text-gray-400 hover:text-gray-200"
                }`}
              >
                {m === "image" && <><Image size={14} /> Ảnh X-quang</>}
                {m === "audio" && <><Mic size={14} /> Âm thanh</>}
                {m === "fusion" && <><Image size={14} />+<Mic size={14} /> Fusion</>}
              </button>
            ))}
          </div>

          {/* Upload area */}
          <div className={`grid gap-3 ${isFusion ? "grid-cols-2" : "grid-cols-1"}`}>
            {(isImage || isFusion) && (
              <UploadBox
                icon={<Image size={22} />}
                label="Ảnh X-quang"
                accept="image/*"
                file={d.imageFile}
                onChange={d.setImageFile}
                preview={imgPreview}
                disabled={d.loading}
              />
            )}
            {(isAudio || isFusion) && (
              <UploadBox
                icon={<Mic size={22} />}
                label="File Audio (.wav)"
                accept="audio/*"
                file={d.audioFile}
                onChange={d.setAudioFile}
                preview={audioPreview}
                disabled={d.loading}
                isAudio
              />
            )}
          </div>

          {/* Buttons */}
          <div className="flex gap-3">
            <button onClick={d.run} disabled={!d.canSubmit}
              className="btn-primary flex items-center gap-2">
              {d.loading && <Loader2 size={16} className="animate-spin" />}
              {d.loading ? "Đang chẩn đoán..." : "Chẩn đoán"}
            </button>
            {d.result && <button onClick={d.reset} className="btn-danger">Đặt lại</button>}
          </div>
        </div>
      )}

      {/* ═══ TAB: Kết quả — FULL WIDTH, 3 CỘT ══════════════════════ */}
      {subTab === "result" && d.result && (
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
          {/* Cột 1: Ảnh gốc + Heatmap + Attention */}
          <div className="space-y-3">
            {d.imageUrl && (
              <ImageCard title="Ảnh X-quang gốc" src={d.imageUrl!} onZoom={() => setLightboxSrc(d.imageUrl!)} />
            )}
            {hmUrl && (
              <ImageCard title="Grad-CAM Heatmap" src={hmUrl!} onZoom={() => setLightboxSrc(hmUrl!)} borderColor="border-blue-500/40" />
            )}
            {attUrl && (
              <ImageCard title="Audio Attention Map" src={attUrl!} onZoom={() => setLightboxSrc(attUrl!)} height="h-44" borderColor="border-purple-500/40" />
            )}
            {d.audioUrl && (
              <div className="rounded-lg border border-gray-700 bg-gray-900/30 p-3">
                <p className="text-[11px] text-gray-500 uppercase tracking-wide mb-2">Audio đã upload</p>
                <audio src={d.audioUrl} controls className="w-full h-8" />
              </div>
            )}
          </div>

          {/* Cột 2 + 3: 5 hành động pipeline */}
          <div className="xl:col-span-2">
            <DiagnosisResult result={d.result} compact />
          </div>
        </div>
      )}

      {/* Lightbox */}
      {lightboxSrc && (
        <div className="fixed inset-0 z-50 bg-black/90 flex items-center justify-center cursor-pointer"
          onClick={() => setLightboxSrc(null)}>
          <button className="absolute top-4 right-4 text-white/80 hover:text-white"
            onClick={() => setLightboxSrc(null)}>
            <X size={28} />
          </button>
          <img src={lightboxSrc} alt="Zoom" className="max-w-[92vw] max-h-[92vh] object-contain rounded" />
        </div>
      )}
    </div>
  );
}

/* ── Upload Box ──────────────────────────────────────────────────── */
function UploadBox({ icon, label, accept, file, onChange, preview, disabled, isAudio }:
  { icon: React.ReactNode; label: string; accept: string; file: File | null;
    onChange: (f: File | null) => void; preview: string | null; disabled: boolean; isAudio?: boolean }) {
  return (
    <div
      className={`rounded-xl border-2 border-dashed p-4 flex flex-col items-center gap-2 transition cursor-pointer ${
        file ? "border-green-500/50 bg-green-500/5" : "border-gray-600 hover:border-gray-500 bg-gray-900/30"
      }`}
      onClick={() => !disabled && document.getElementById(`file-${label}`)?.click()}
    >
      <input id={`file-${label}`} type="file" accept={accept} disabled={disabled} className="hidden"
        onChange={(e) => { const f = e.target.files?.[0]; if (f) onChange(f); }} />
      {file ? (
        isAudio ? (
          <div className="w-full">
            <p className="text-xs text-green-400 text-center mb-2 truncate">{file.name}</p>
            {preview && <audio src={preview} controls className="w-full h-8" />}
          </div>
        ) : (
          <>
            {preview && <img src={preview} alt="preview" className="max-h-40 rounded-lg object-contain" />}
            <p className="text-xs text-green-400 mt-1">{file.name}</p>
          </>
        )
      ) : (
        <>
          <div className="w-10 h-10 rounded-full bg-gray-700 flex items-center justify-center text-gray-400">{icon}</div>
          <p className="text-sm text-gray-400">{label}</p>
          <p className="text-xs text-gray-600">Click để chọn file</p>
        </>
      )}
    </div>
  );
}

/* ── Image Card ──────────────────────────────────────────────────── */
function ImageCard({ title, src, onZoom, height = "h-48", borderColor = "border-gray-600" }:
  { title: string; src: string; onZoom: () => void; height?: string; borderColor?: string }) {
  return (
    <div className="space-y-1">
      <p className="text-[11px] text-gray-500 uppercase tracking-wide">{title}</p>
      <div className={`rounded-xl overflow-hidden border ${borderColor} bg-black cursor-pointer group relative`}
        onClick={onZoom}>
        <img src={src} alt={title} className={`w-full ${height} object-contain`} />
        <div className="absolute top-2 right-2 bg-black/60 rounded p-1 opacity-0 group-hover:opacity-100 transition">
          <Maximize2 size={14} className="text-white" />
        </div>
      </div>
    </div>
  );
}
