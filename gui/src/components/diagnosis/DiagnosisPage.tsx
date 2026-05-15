import { useDiagnosis } from "../../hooks/useDiagnosis";
import ModeSelector from "./ModeSelector";
import FileUploader from "./FileUploader";
import ImagePreview from "./ImagePreview";
import AudioPlayer from "./AudioPlayer";
import ConfidenceGauge from "./ConfidenceGauge";
import DiagnosisResult from "./DiagnosisResult";
import { Loader2 } from "lucide-react";

export default function DiagnosisPage() {
  const d = useDiagnosis();

  return (
    <div className="mx-auto max-w-3xl space-y-6">
      <div>
        <h2 className="text-xl font-bold text-white">Chẩn đoán Viêm Phổi</h2>
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
            <ImagePreview url={d.imageUrl} />
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

      {d.result && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <ConfidenceGauge
            value={d.result.result.confidence}
            threshold={d.result.result.threshold}
          />
          <DiagnosisResult result={d.result} />
        </div>
      )}
    </div>
  );
}
