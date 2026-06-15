import { useRef, useState } from "react";
import { Play, Pause } from "lucide-react";

interface Props {
  url: string | null;
}

export default function AudioPlayer({ url }: Props) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [playing, setPlaying] = useState(false);

  if (!url) return null;

  const toggle = () => {
    const a = audioRef.current;
    if (!a) return;
    if (playing) {
      a.pause();
    } else {
      a.play().catch(() => {});
    }
    setPlaying(!playing);
  };

  return (
    <div className="flex items-center gap-3 rounded-lg border border-gray-700 bg-gray-900 p-3">
      <audio
        ref={audioRef}
        src={url}
        onEnded={() => setPlaying(false)}
        className="hidden"
      />
      <button
        onClick={toggle}
        className="flex h-10 w-10 items-center justify-center rounded-full bg-blue-600 text-white hover:bg-blue-500 transition-colors"
      >
        {playing ? <Pause size={18} /> : <Play size={18} className="ml-0.5" />}
      </button>
      <div>
        <p className="text-sm text-gray-300">Audio file uploaded</p>
        <p className="text-xs text-gray-500">{playing ? "Playing..." : "Ready"}</p>
      </div>
    </div>
  );
}
