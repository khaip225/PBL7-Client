import csv
import json
import os
import shutil
import soundfile as sf
from dataclasses import dataclass
from pathlib import Path

from config import config


@dataclass
class SyncResult:
    audio_dest: str | None
    image_dest: str | None
    batch_dir: str
    csv_path: str | None


class DataSyncManager:
    def __init__(
        self,
        *,
        client_id: str | None = None,
        base_dir: str | None = None,
        state_file: str | None = None,
    ):
        project_root = Path(base_dir or Path(__file__).resolve().parent.parent)
        self.project_root = project_root
        self.client_id = client_id or config.CLIENT_ID
        self.client_tag = self._normalize_client_tag(client_id)

        self.local_data_dir = project_root / "Local_Data"
        self.pending_dir = self.local_data_dir / "pending"

        self.pending_audio_dir = self.pending_dir / "audio_files"
        self.pending_image_dir = self.pending_dir / "image_files"

        self.state_file = Path(state_file or (project_root / "local_managers" / "fl_state.json"))

    def load_state(self) -> dict:
        if not self.state_file.exists():
            default_state = {"current_batch": 1, "client_id": self.client_tag, "threshold": 300}
            self._write_state(default_state)
            return default_state

        with self.state_file.open("r", encoding="utf-8") as f:
            return json.load(f)

    def save_state(self, state: dict) -> None:
        self._write_state(state)

    def advance_batch(self) -> dict:
        state = self.load_state()
        state["current_batch"] = int(state.get("current_batch", 1)) + 1
        self._write_state(state)
        return state

    def sync_record(self, audio_path: str | None, image_path: str | None, label: str) -> SyncResult:
        state = self.load_state()
        batch_dir = self._ensure_batch_dir(int(state.get("current_batch", 1)))

        audio_dest = None
        image_dest = None
        csv_path = None

        if audio_path:
            audio_segments, csv_path = self._sync_audio_segments(audio_path, batch_dir, label)
            audio_dest = audio_segments[0] if audio_segments else None

        if image_path:
            class_dir = "PNEUMONIA" if label == "Abnormal" else "NORMAL"
            image_dest_dir = batch_dir / "fl_image" / class_dir
            image_dest = self._move_to_batch(image_path, image_dest_dir)

        return SyncResult(audio_dest=audio_dest, image_dest=image_dest, batch_dir=str(batch_dir), csv_path=csv_path)

    def _move_to_batch(self, source_path: str, dest_dir: Path) -> str:
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / os.path.basename(source_path)

        source_abs = os.path.abspath(source_path)
        dest_abs = os.path.abspath(dest_path)
        if source_abs != dest_abs:
            shutil.move(source_path, dest_path)

        return str(dest_path)

    def _append_audio_ledger(self, batch_dir: Path, audio_path: str, label: str) -> str:
        label_code = 1 if label == "Abnormal" else 0
        csv_dir = batch_dir / "metadata" / "audio_fl"
        csv_dir.mkdir(parents=True, exist_ok=True)
        csv_path = csv_dir / f"{self.client_tag}_train.csv"

        if not csv_path.exists():
            with csv_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["path", "label"])

        with csv_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([os.path.basename(audio_path), label_code])

        return str(csv_path)

    def _sync_audio_segments(self, audio_path: str, batch_dir: Path, label: str) -> tuple[list[str], str | None]:
        dest_dir = batch_dir / "fl_audio"
        dest_dir.mkdir(parents=True, exist_ok=True)

        label = "Abnormal" if label == "Abnormal" else "Normal"
        meta = self._load_segment_metadata(audio_path)

        audio_data, sr = sf.read(audio_path)
        total_samples = audio_data.shape[0]
        duration = total_samples / float(sr) if sr > 0 else 0.0

        segments = meta["segments"]
        if not segments:
            segments = self._generate_segments(duration, meta["chunk_duration"], meta["overlap"])

        selected = []
        if label == "Normal":
            selected = segments
        else:
            threshold = meta["threshold"]
            selected = [seg for seg in segments if seg.get("score", 0.0) >= threshold]
            if not selected and segments:
                selected = [max(segments, key=lambda s: s.get("score", 0.0))]

        segment_paths = []
        csv_path = None
        txt_lines = []
        stem = Path(audio_path).stem
        for idx, seg in enumerate(selected, start=1):
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", 0.0))
            start_idx = int(start * sr)
            end_idx = min(int(end * sr), total_samples)
            if end_idx <= start_idx:
                continue

            segment_audio = audio_data[start_idx:end_idx]
            segment_name = f"{stem}_seg{idx}.wav"
            segment_path = dest_dir / segment_name
            sf.write(segment_path, segment_audio, sr)

            segment_paths.append(str(segment_path))
            csv_path = self._append_audio_ledger(batch_dir, str(segment_path), label)
            txt_lines.append(f"{start:.3f}\t{end:.3f}\t{1 if label == 'Abnormal' else 0}")

        if txt_lines:
            txt_path = dest_dir / f"{stem}.txt"
            with txt_path.open("w", encoding="utf-8") as f:
                f.write("\n".join(txt_lines) + "\n")

        json_path = self._pending_json_path(audio_path)
        if os.path.exists(json_path):
            os.remove(json_path)

        if os.path.exists(audio_path):
            os.remove(audio_path)

        return segment_paths, csv_path if segment_paths else None

    def _load_segment_metadata(self, audio_path: str) -> dict:
        json_path = self._pending_json_path(audio_path)
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return {
                    "segments": data.get("segments", []),
                    "chunk_duration": float(data.get("chunk_duration", config.AUDIO_CHUNK_DURATION)),
                    "overlap": float(data.get("overlap", config.AUDIO_CHUNK_OVERLAP)),
                    "threshold": float(data.get("threshold", config.AUDIO_SEGMENT_THRESHOLD)),
                }
            except (OSError, json.JSONDecodeError, ValueError):
                pass

        return {
            "segments": [],
            "chunk_duration": config.AUDIO_CHUNK_DURATION,
            "overlap": config.AUDIO_CHUNK_OVERLAP,
            "threshold": config.AUDIO_SEGMENT_THRESHOLD,
        }

    @staticmethod
    def _generate_segments(duration: float, chunk_duration: float, overlap: float) -> list[dict]:
        segments = []
        hop = chunk_duration * (1 - overlap)
        if hop <= 0:
            hop = chunk_duration

        start = 0.0
        idx = 0
        while start < duration:
            end = min(start + chunk_duration, duration)
            segments.append({"index": idx, "start": start, "end": end, "score": 0.0})
            idx += 1
            start += hop

        return segments

    def _ensure_batch_dir(self, batch_idx: int) -> Path:
        base = Path(config.FL_DATA_DIR).resolve()
        parent = base.parent if base.name.startswith("fl_data") else base
        batch_dir = parent / f"fl_data_{batch_idx}"

        (batch_dir / "fl_audio").mkdir(parents=True, exist_ok=True)
        (batch_dir / "fl_image").mkdir(parents=True, exist_ok=True)
        (batch_dir / "metadata" / "audio_fl").mkdir(parents=True, exist_ok=True)

        return batch_dir

    def _write_state(self, state: dict) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with self.state_file.open("w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=True, indent=2)

    @staticmethod
    def _normalize_client_tag(client_id) -> str:
        if isinstance(client_id, str):
            value = client_id.strip()
            if value.startswith("client_"):
                return value
            if value.isdigit():
                return f"client_{value}"
        return f"client_{client_id}"

    @staticmethod
    def _pending_json_path(audio_path: str) -> str:
        return os.path.splitext(audio_path)[0] + ".json"
