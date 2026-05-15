import { useCallback, type DragEvent } from "react";
import { Upload, X, FileImage, FileAudio } from "lucide-react";

interface Props {
  file: File | null;
  onChange: (f: File | null) => void;
  accept: string;
  label: string;
  icon: "image" | "audio";
}

const acceptLabels: Record<string, string> = {
  "image/*": "PNG, JPG",
  "audio/*": "WAV",
};

export default function FileUploader({ file, onChange, accept, label, icon }: Props) {
  const handleDrop = useCallback(
    (e: DragEvent) => {
      e.preventDefault();
      const f = e.dataTransfer.files[0];
      if (f) onChange(f);
    },
    [onChange]
  );

  const handleDragOver = (e: DragEvent) => {
    e.preventDefault();
  };

  const Icon = icon === "image" ? FileImage : FileAudio;

  return (
    <div>
      <p className="text-sm text-gray-400 mb-2">{label}</p>
      {file ? (
        <div className="flex items-center justify-between rounded-lg border border-gray-700 bg-gray-900 p-3">
          <div className="flex items-center gap-2 text-sm text-gray-300">
            <Icon size={18} className="text-blue-400" />
            <span className="truncate max-w-[200px]">{file.name}</span>
            <span className="text-gray-600">
              ({(file.size / 1024).toFixed(0)} KB)
            </span>
          </div>
          <button
            onClick={() => onChange(null)}
            className="text-gray-500 hover:text-red-400"
          >
            <X size={16} />
          </button>
        </div>
      ) : (
        <label
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          className="flex flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed border-gray-700 bg-gray-900/50 p-6 cursor-pointer hover:border-gray-500 hover:bg-gray-900 transition-colors"
        >
          <Upload size={24} className="text-gray-500" />
          <span className="text-sm text-gray-400">
            Kéo thả hoặc <span className="text-blue-400">chọn file</span>
          </span>
          <span className="text-xs text-gray-600">{acceptLabels[accept] ?? accept}</span>
          <input
            type="file"
            accept={accept}
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) onChange(f);
            }}
            className="hidden"
          />
        </label>
      )}
    </div>
  );
}
