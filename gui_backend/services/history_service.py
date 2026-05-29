import csv
import os
import re
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOCAL_DATA = os.path.join(PROJECT_ROOT, "Local_Data", "pending")

FILENAME_RE = re.compile(r"^(.+)_(\d{8}_\d{6})\.(\w+)$")
ALLOWED_AUDIO_EXT = {".wav"}
ALLOWED_IMAGE_EXT = {".png", ".jpg", ".jpeg"}


class HistoryService:
    def list_records(self, page: int = 1, page_size: int = 20) -> dict:
        records = []
        seen = {}  # key: (timestamp, label) -> record

        metadata = self._load_metadata()

        image_dir = os.path.join(LOCAL_DATA, "image_files")
        audio_dir = os.path.join(LOCAL_DATA, "audio_files")

        for dir_path, file_type in [(image_dir, "image"), (audio_dir, "audio")]:
            if not os.path.isdir(dir_path):
                continue
            for fname in sorted(os.listdir(dir_path), reverse=True):
                m = FILENAME_RE.match(fname)
                if not m:
                    continue
                label, ts_str, ext = m.group(1), m.group(2), m.group(3)
                ext_dot = f".{ext.lower()}"
                if file_type == "audio" and ext_dot not in ALLOWED_AUDIO_EXT:
                    continue
                if file_type == "image" and ext_dot not in ALLOWED_IMAGE_EXT:
                    continue
                try:
                    ts = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
                except ValueError:
                    continue

                key = (ts_str, label)
                if key not in seen:
                    seen[key] = {
                        "id": f"{label}_{ts_str}",
                        "timestamp": ts.isoformat(),
                        "label": label,
                        "mode": "unknown",
                        "audio_file": None,
                        "image_file": None,
                        "audio_path": None,
                        "image_path": None,
                        "confidence": None,
                    }

                record = seen[key]
                if file_type == "audio":
                    record["audio_file"] = fname
                    record["audio_path"] = os.path.join(audio_dir, fname)
                else:
                    record["image_file"] = fname
                    record["image_path"] = os.path.join(image_dir, fname)

                # Determine mode
                if record["audio_file"] and record["image_file"]:
                    record["mode"] = "fusion"
                elif record["audio_file"]:
                    record["mode"] = "audio"
                else:
                    record["mode"] = "image"

                meta = metadata.get(record["id"])
                if meta and meta.get("confidence") is not None:
                    record["confidence"] = meta.get("confidence")

        records = sorted(seen.values(), key=lambda r: r["timestamp"], reverse=True)
        total = len(records)

        start = (page - 1) * page_size
        items = records[start:start + page_size]

        return {"items": items, "total": total, "page": page, "page_size": page_size}

    def get_image_path(self, record_id: str) -> str | None:
        record = self._find_record(record_id)
        return record.get("image_path") if record else None

    def get_audio_path(self, record_id: str) -> str | None:
        record = self._find_record(record_id)
        return record.get("audio_path") if record else None

    def get_record(self, record_id: str) -> dict | None:
        return self._find_record(record_id)

    def _find_record(self, record_id: str) -> dict | None:
        result = self.list_records(page=1, page_size=1000)
        for item in result["items"]:
            if item["id"] == record_id:
                return item
        return None

    def _load_metadata(self) -> dict:
        meta_path = os.path.join(LOCAL_DATA, "metadata", "diagnosis_log.csv")
        if not os.path.exists(meta_path):
            return {}

        data = {}
        with open(meta_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                record_id = row.get("record_id")
                if not record_id:
                    continue
                confidence = row.get("confidence")
                try:
                    confidence_val = float(confidence) if confidence not in (None, "") else None
                except ValueError:
                    confidence_val = None
                data[record_id] = {
                    "confidence": confidence_val,
                }
        return data
