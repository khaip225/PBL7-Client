import { useState } from "react";
import { useDiagnosis } from "../../hooks/useDiagnosis";
import ModeSelector from "./ModeSelector";
import FileUploader from "./FileUploader";
import ImagePreview from "./ImagePreview";
import AudioPlayer from "./AudioPlayer";
import ConfidenceGauge from "./ConfidenceGauge";
import DiagnosisResult from "./DiagnosisResult";
import { Loader2, Maximize2, X } from "lucide-react";

function heatmapUrl(path: string | null): string | null {
  if (!path) return null;
  return `/api/heatmap?path=${encodeURIComponent(path)}`;
}

export default function DiagnosisPage() {
  const d = useDiagnosis();
  const [lightboxSrc, setLightboxSrc] = useState<string | null>(null);

  const hmUrl = d.result ? heatmapUrl(d.result.heatmap_path) : null;

  return (
    <div className="mx-auto max-w-3xl space-y-6">
      <div>
        <h2 className="text-xl font-bold text-white">Chẩn đoán Bệnh Lý Phổi</h2>
        <p className="text-sm text-gray-400 mt-1">
          Tải lên ảnh X-quang, file audio hoặc cả hai để chẩn đoán.
        </p>
      </div>

      <ModeSelector value={d.mode} onChange={d.setMode} disabled={d.loading} />

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {(d.mode === "fusion" || d.mode === "image") && (
          <div className="space-y-3">
            <FileUploader
              file={d.imageFile}
              onChange={d.setImageFile}
              accept="image/*"
              label="Ảnh X-quang"
              icon="image"
            />
            {/* Neu chua co ket qua: preview bt. Neu co roi: hien anh goc + heatmap canh nhau */}
            {!d.result && <ImagePreview url={d.imageUrl} />}
          </div>
        )}
        {(d.mode === "fusion" || d.mode === "audio") && (
          <div className="space-y-3">
            <FileUploader
              file={d.audioFile}
              onChange={d.setAudioFile}
              accept="audio/*"
              label="File Audio (.wav)"
              icon="audio"
            />
            <AudioPlayer url={d.audioUrl} />
          </div>
        )}
      </div>

      <div className="flex gap-3">
        <button
          onClick={d.run}
          disabled={!d.canSubmit}
          className="btn-primary flex items-center gap-2"
        >
          {d.loading && <Loader2 size={16} className="animate-spin" />}
          {d.loading ? "Đang chẩn đoán..." : "Chẩn đoán"}
        </button>
        {d.result && (
          <button onClick={d.reset} className="btn-danger">
            Đặt lại
          </button>
        )}
      </div>

      {d.error && (
        <div className="rounded-lg border border-red-600/30 bg-red-600/10 px-4 py-3 text-sm text-red-400">
          {d.error}
        </div>
      )}

      {/* So sanh anh goc vs heatmap — hien ngay khi co ket qua */}
      {d.result && d.imageUrl && hmUrl && (
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-500 uppercase tracking-wide">
              Anh X-quang goc
            </span>
            <span className="text-gray-600">|</span>
            <span className="text-xs text-blue-400 uppercase tracking-wide">
              Grad-CAM Heatmap
            </span>
          </div>
          <div className="grid grid-cols-2 gap-3">
            {/* Anh goc */}
            <div
              className="rounded-xl overflow-hidden border border-gray-600 bg-black cursor-pointer group relative"
              onClick={() => setLightboxSrc(d.imageUrl!)}
            >
              <img src={d.imageUrl!} alt="X-quang" className="w-full h-64 object-contain" />
              <div className="absolute top-2 right-2 bg-black/60 rounded p-1.5 opacity-0 group-hover:opacity-100 transition-opacity">
                <Maximize2 size={16} className="text-white" />
              </div>
            </div>
            {/* Heatmap */}
            <div
              className="rounded-xl overflow-hidden border border-blue-500/40 bg-black cursor-pointer group relative"
              onClick={() => setLightboxSrc(hmUrl!)}
            >
              <img src={hmUrl!} alt="Heatmap" className="w-full h-64 object-contain" />
              <div className="absolute top-2 right-2 bg-black/60 rounded p-1.5 opacity-0 group-hover:opacity-100 transition-opacity">
                <Maximize2 size={16} className="text-white" />
              </div>
            </div>
          </div>
          <p className="text-xs text-gray-500 text-center">
            Click vao anh de phong to
          </p>
        </div>
      )}

      {d.result && (
        <div className="space-y-6">
          <ConfidenceGauge
            value={d.result.result.confidence}
            threshold={d.result.result.threshold}
          />
          <DiagnosisResult result={d.result} />
        </div>
      )}

      {/* Lightbox */}
      {lightboxSrc && (
        <div
          className="fixed inset-0 z-50 bg-black/90 flex items-center justify-center cursor-pointer"
          onClick={() => setLightboxSrc(null)}
        >
          <button
            className="absolute top-4 right-4 text-white/80 hover:text-white"
            onClick={() => setLightboxSrc(null)}
          >
            <X size={28} />
          </button>
          <img
            src={lightboxSrc}
            alt="Phong to"
            className="max-w-[90vw] max-h-[90vh] object-contain"
          />
        </div>
      )}
    </div>
  );
}
