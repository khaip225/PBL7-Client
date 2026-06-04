import csv
import json
import os
import re
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOCAL_DATA = os.path.join(PROJECT_ROOT, "Local_Data", "pending")
TRASH_DIR = os.path.join(PROJECT_ROOT, "Local_Data", "trash")

FILENAME_RE = re.compile(r"^(.+)_(\d{8}_\d{6})\.(\w+)$")
ALLOWED_AUDIO_EXT = {".wav"}
ALLOWED_IMAGE_EXT = {".png", ".jpg", ".jpeg"}

KNOWN_DISEASES = {"Pneumonia", "COPD_Emphysema", "Fibrosis"}
KNOWN_ACOUSTIC = {"Crackle", "Wheeze"}
KNOWN_LABELS = KNOWN_DISEASES | KNOWN_ACOUSTIC | {"Normal"}


class HistoryService:
    @staticmethod
    def _parse_label_str(label_str: str) -> list[str]:
        """Parse multi-label filename prefix back to individual labels.

        Example: "Pneumonia_Crackle" → ["Pneumonia", "Crackle"]
                 "Normal" → ["Normal"]
        """
        if not label_str:
            return ["Normal"]
        # Try known labels in order of length (longest first to match COPD_Emphysema)
        known_sorted = sorted(KNOWN_LABELS, key=len, reverse=True)
        # Greedy: iterate through known labels and collect matches
        result = []
        remaining = label_str
        for known in known_sorted:
            # Check at word boundaries (underscore or start/end)
            idx = remaining.find(known)
            while idx != -1:
                # Check left boundary
                left_ok = idx == 0 or remaining[idx - 1] == "_"
                # Check right boundary
                right_end = idx + len(known)
                right_ok = right_end == len(remaining) or remaining[right_end] == "_"
                if left_ok and right_ok:
                    result.append(known)
                    # Remove this match from remaining
                    before = remaining[:idx]
                    after = remaining[right_end:]
                    remaining = (before + after).replace("__", "_").strip("_")
                    break
                idx = remaining.find(known, idx + 1)
        if not result:
            # Fallback: simple split
            result = [s for s in label_str.split("_") if s]
        if not result:
            return ["Normal"]
        return result

    def list_records(self, page: int = 1, page_size: int = 20) -> dict:
        records = []
        seen = {}  # key: (timestamp, label_str) -> record

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
                label_str, ts_str, ext = m.group(1), m.group(2), m.group(3)
                ext_dot = f".{ext.lower()}"
                if file_type == "audio" and ext_dot not in ALLOWED_AUDIO_EXT:
                    continue
                if file_type == "image" and ext_dot not in ALLOWED_IMAGE_EXT:
                    continue
                try:
                    ts = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
                except ValueError:
                    continue

                # Parse multi-label from filename prefix
                labels = self._parse_label_str(label_str)

                key = (ts_str, label_str)
                if key not in seen:
                    seen[key] = {
                        "id": f"{label_str}_{ts_str}",
                        "timestamp": ts.isoformat(),
                        "labels": labels,
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
                if meta:
                    if meta.get("labels") is not None:
                        record["labels"] = meta["labels"]
                    if meta.get("confidence") is not None:
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

    def delete_record(self, record_id: str) -> dict:
        """Chuyển record vào thùng rác thay vì xóa thẳng."""
        record = self._find_record(record_id)
        if not record:
            raise ValueError(f"Record not found: {record_id}")

        os.makedirs(os.path.join(TRASH_DIR, "image_files"), exist_ok=True)
        os.makedirs(os.path.join(TRASH_DIR, "audio_files"), exist_ok=True)
        os.makedirs(os.path.join(TRASH_DIR, "metadata"), exist_ok=True)
        os.makedirs(os.path.join(TRASH_DIR, "heatmaps"), exist_ok=True)

        moved_files = []

        # Chuyển ảnh vào trash
        img_path = record.get("image_path")
        img_file = record.get("image_file")
        if img_path and img_file and os.path.isfile(img_path):
            dest = os.path.join(TRASH_DIR, "image_files", img_file)
            os.rename(img_path, dest)
            moved_files.append(img_file)

        # Chuyển audio vào trash
        aud_path = record.get("audio_path")
        aud_file = record.get("audio_file")
        if aud_path and aud_file and os.path.isfile(aud_path):
            dest = os.path.join(TRASH_DIR, "audio_files", aud_file)
            os.rename(aud_path, dest)
            moved_files.append(aud_file)

        # Chuyển meta vào trash
        self._move_metadata_to_trash(record_id, record)

        return {
            "record_id": record_id,
            "moved_files": moved_files,
        }

    # ------------------------------------------------------------------
    # Thùng rác
    # ------------------------------------------------------------------

    def list_trash(self) -> list[dict]:
        """Liệt kê các record trong thùng rác."""
        trash_meta = os.path.join(TRASH_DIR, "metadata", "trash_log.csv")
        if not os.path.exists(trash_meta):
            return []

        records = []
        with open(trash_meta, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels_str = row.get("detected_labels", "[]")
                try:
                    labels = json.loads(labels_str)
                except (json.JSONDecodeError, TypeError):
                    labels = self._parse_label_str(row.get("label", ""))

                confidence = row.get("confidence")
                try:
                    confidence = float(confidence) if confidence not in (None, "") else None
                except ValueError:
                    confidence = None

                records.append({
                    "id": row.get("record_id", ""),
                    "timestamp": row.get("timestamp", ""),
                    "labels": labels,
                    "mode": row.get("mode", "unknown"),
                    "audio_file": row.get("audio_file") or None,
                    "image_file": row.get("image_file") or None,
                    "audio_path": os.path.join(TRASH_DIR, "audio_files", row.get("audio_file", "")) if row.get("audio_file") else None,
                    "image_path": os.path.join(TRASH_DIR, "image_files", row.get("image_file", "")) if row.get("image_file") else None,
                    "confidence": confidence,
                    "trashed_at": row.get("trashed_at", ""),
                })

        return sorted(records, key=lambda r: r["trashed_at"], reverse=True)

    def restore_from_trash(self, record_id: str) -> dict:
        """Khôi phục record từ thùng rác về pending."""
        record = self._find_trash_record(record_id)
        if not record:
            raise ValueError(f"Record not found in trash: {record_id}")

        restored = []

        # Chuyển ảnh về
        if record.get("image_file"):
            src = os.path.join(TRASH_DIR, "image_files", record["image_file"])
            dst = os.path.join(LOCAL_DATA, "image_files", record["image_file"])
            if os.path.isfile(src):
                os.rename(src, dst)
                restored.append(record["image_file"])

        # Chuyển audio về
        if record.get("audio_file"):
            src = os.path.join(TRASH_DIR, "audio_files", record["audio_file"])
            dst = os.path.join(LOCAL_DATA, "audio_files", record["audio_file"])
            if os.path.isfile(src):
                os.rename(src, dst)
                restored.append(record["audio_file"])

        # Khôi phục metadata về diagnosis_log.csv
        self._restore_metadata_from_trash(record_id, record)

        return {"record_id": record_id, "restored_files": restored}

    def delete_permanently(self, record_id: str) -> dict:
        """Xóa vĩnh viễn record khỏi thùng rác."""
        record = self._find_trash_record(record_id)
        if not record:
            raise ValueError(f"Record not found in trash: {record_id}")

        deleted = []

        img_file = record.get("image_file")
        if img_file:
            path = os.path.join(TRASH_DIR, "image_files", img_file)
            if os.path.isfile(path):
                os.remove(path)
                deleted.append(img_file)

        aud_file = record.get("audio_file")
        if aud_file:
            path = os.path.join(TRASH_DIR, "audio_files", aud_file)
            if os.path.isfile(path):
                os.remove(path)
                deleted.append(aud_file)

        # Xóa dòng trong trash_log.csv
        self._remove_from_trash_log(record_id)

        return {"record_id": record_id, "deleted_files": deleted}

    def _move_metadata_to_trash(self, record_id: str, record: dict):
        """Chuyển dòng metadata từ diagnosis_log sang trash_log."""
        # Đọc dòng cần chuyển từ diagnosis_log
        meta_path = os.path.join(LOCAL_DATA, "metadata", "diagnosis_log.csv")
        target_row = None
        remaining_rows = []
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames
                for row in reader:
                    if row.get("record_id") == record_id:
                        target_row = dict(row)
                    else:
                        remaining_rows.append(row)
            with open(meta_path, "w", encoding="utf-8", newline="") as f:
                if fieldnames:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(remaining_rows)

        # Ghi vào trash_log
        trash_meta = os.path.join(TRASH_DIR, "metadata", "trash_log.csv")
        trash_fields = [
            "record_id", "timestamp", "mode", "label", "detected_labels",
            "confidence", "audio_file", "image_file", "trashed_at",
        ]
        file_exists = os.path.exists(trash_meta)
        with open(trash_meta, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=trash_fields)
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                "record_id": record_id,
                "timestamp": record.get("timestamp", ""),
                "mode": record.get("mode", ""),
                "label": "_".join(record.get("labels", [])),
                "detected_labels": json.dumps(record.get("labels", []), ensure_ascii=False),
                "confidence": record.get("confidence"),
                "audio_file": record.get("audio_file") or "",
                "image_file": record.get("image_file") or "",
                "trashed_at": datetime.now().isoformat(),
            })

    def _restore_metadata_from_trash(self, record_id: str, record: dict):
        """Khôi phục metadata từ trash_log về diagnosis_log."""
        # Đọc dòng từ trash_log
        trash_meta = os.path.join(TRASH_DIR, "metadata", "trash_log.csv")
        target_row = None
        remaining = []
        if os.path.exists(trash_meta):
            with open(trash_meta, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames
                for row in reader:
                    if row.get("record_id") == record_id:
                        target_row = dict(row)
                    else:
                        remaining.append(row)
            with open(trash_meta, "w", encoding="utf-8", newline="") as f:
                if fieldnames:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(remaining)

        # Ghi lại vào diagnosis_log
        if target_row:
            diag_path = os.path.join(LOCAL_DATA, "metadata", "diagnosis_log.csv")
            diag_fields = [
                "record_id", "timestamp", "mode", "label", "detected_labels",
                "confidence", "audio_file", "image_file", "audio_path",
                "image_path", "audio_score", "image_score", "fusion_score",
            ]
            file_exists = os.path.exists(diag_path)
            with open(diag_path, "a", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=diag_fields)
                if not file_exists:
                    writer.writeheader()
                writer.writerow({
                    "record_id": record_id,
                    "timestamp": target_row.get("timestamp", ""),
                    "mode": target_row.get("mode", ""),
                    "label": target_row.get("label", ""),
                    "detected_labels": target_row.get("detected_labels", "[]"),
                    "confidence": target_row.get("confidence", ""),
                    "audio_file": target_row.get("audio_file", ""),
                    "image_file": target_row.get("image_file", ""),
                    "audio_path": os.path.join(LOCAL_DATA, "audio_files", target_row.get("audio_file", "")),
                    "image_path": os.path.join(LOCAL_DATA, "image_files", target_row.get("image_file", "")),
                    "audio_score": "",
                    "image_score": "",
                    "fusion_score": "",
                })

    def _remove_from_trash_log(self, record_id: str):
        """Xóa dòng trong trash_log.csv."""
        trash_meta = os.path.join(TRASH_DIR, "metadata", "trash_log.csv")
        if not os.path.exists(trash_meta):
            return
        rows = []
        with open(trash_meta, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            for row in reader:
                if row.get("record_id") != record_id:
                    rows.append(row)
        with open(trash_meta, "w", encoding="utf-8", newline="") as f:
            if fieldnames:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

    def _find_trash_record(self, record_id: str) -> dict | None:
        """Tìm record trong thùng rác."""
        trash_list = self.list_trash()
        for r in trash_list:
            if r["id"] == record_id:
                return r
        return None

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
                # Parse detected_labels JSON from metadata
                labels = None
                detected_str = row.get("detected_labels", "")
                if detected_str:
                    try:
                        labels = json.loads(detected_str)
                    except (json.JSONDecodeError, TypeError):
                        # Fallback: parse from label column
                        label_col = row.get("label", "")
                        if label_col:
                            labels = self._parse_label_str(label_col)
                data[record_id] = {
                    "confidence": confidence_val,
                    "labels": labels,
                }
        return data
