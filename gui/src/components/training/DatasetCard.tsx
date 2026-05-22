import { Database, Music, Image as ImageIcon } from "lucide-react";
import type { DatasetInfo } from "../../lib/types";

interface Props {
  dataset?: DatasetInfo | null;
}

const safe = (d?: DatasetInfo | null): DatasetInfo => ({
  total_samples: d?.total_samples ?? 0,
  audio_samples: d?.audio_samples ?? 0,
  image_samples: d?.image_samples ?? 0,
  has_audio: d?.has_audio ?? false,
  has_image: d?.has_image ?? false,
});

export default function DatasetCard({ dataset }: Props) {
  const ds = safe(dataset);

  return (
    <div className="card space-y-3">
      <h3 className="text-sm font-semibold text-gray-300 flex items-center gap-2">
        <Database size={14} />
        Dữ liệu huấn luyện
      </h3>

      <div className="space-y-2">
        {ds.has_image && (
          <div className="flex items-center justify-between text-sm">
            <span className="flex items-center gap-1.5 text-gray-400">
              <ImageIcon size={14} className="text-blue-400" />
              Hình ảnh
            </span>
            <span className="font-mono text-white">{ds.image_samples.toLocaleString()}</span>
          </div>
        )}

        {ds.has_audio && (
          <div className="flex items-center justify-between text-sm">
            <span className="flex items-center gap-1.5 text-gray-400">
              <Music size={14} className="text-purple-400" />
              Âm thanh
            </span>
            <span className="font-mono text-white">{ds.audio_samples.toLocaleString()}</span>
          </div>
        )}

        <div className="flex items-center justify-between text-sm pt-2 border-t border-gray-800">
          <span className="text-gray-500">Tổng</span>
          <span className="font-mono text-white font-semibold">{ds.total_samples.toLocaleString()} samples</span>
        </div>

        {!ds.has_audio && !ds.has_image && (
          <div className="text-center py-2 text-xs text-yellow-400">
            Chưa có dữ liệu huấn luyện
          </div>
        )}
      </div>
    </div>
  );
}
