import { Layers, Image, Mic } from "lucide-react";
import type { DiagnosisMode } from "../../hooks/useDiagnosis";

const modes: { id: DiagnosisMode; label: string; desc: string; icon: typeof Layers }[] = [
  { id: "fusion", label: "Fusion", desc: "Combined X-ray & Audio", icon: Layers },
  { id: "image", label: "X-ray", desc: "X-ray image only", icon: Image },
  { id: "audio", label: "Audio", desc: "Audio file only", icon: Mic },
];

interface Props {
  value: DiagnosisMode;
  onChange: (m: DiagnosisMode) => void;
  disabled?: boolean;
}

export default function ModeSelector({ value, onChange, disabled }: Props) {
  return (
    <div className="flex gap-3">
      {modes.map(({ id, label, desc, icon: Icon }) => (
        <button
          key={id}
          disabled={disabled}
          onClick={() => onChange(id)}
          className={`flex-1 flex items-center gap-3 rounded-lg border p-4 text-left transition-colors ${
            value === id
              ? "border-blue-500 bg-blue-600/10 text-blue-400"
              : "border-gray-700 bg-gray-900 text-gray-400 hover:border-gray-600"
          } disabled:opacity-50`}
        >
          <Icon size={22} />
          <div>
            <p className="text-sm font-medium text-white">{label}</p>
            <p className="text-xs text-gray-500">{desc}</p>
          </div>
        </button>
      ))}
    </div>
  );
}
