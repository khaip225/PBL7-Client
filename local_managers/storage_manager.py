import csv
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

from config import config


class StorageManager:
    def __init__(self, base_dir=None, client_id=1):
        self.client_id = client_id
        self.client_tag = self._normalize_client_tag(client_id)
        self.pending_dir = base_dir or os.path.join(".", "Local_Data", "pending")
        self.client_dir = self.pending_dir  # backward compat for diagnosis_service
        self.audio_dir = os.path.join(self.pending_dir, "audio_files")
        self.image_dir = os.path.join(self.pending_dir, "image_files")
        self.meta_dir = os.path.join(self.pending_dir, "metadata")
        self.meta_file = os.path.join(self.meta_dir, "diagnosis_log.csv")

        os.makedirs(self.audio_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.meta_dir, exist_ok=True)

        self.fl_data_dir = config.FL_DATA_DIR
        self.fl_image_dir = os.path.join(self.fl_data_dir, "fl_image")

    def save_files(self, audio_source, image_source, labels, *, mode=None, confidence=None, scores=None):
        """Save diagnosis files with multi-label support.

        Args:
            labels: list[str] — detected disease/acoustic classes (e.g. ["Pneumonia", "Crackle"])
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Join labels for filename prefix
        label_str = "_".join(labels) if labels else "Normal"
        audio_dest, image_dest = None, None

        if audio_source is not None:
            audio_ext = os.path.splitext(audio_source)[1]
            new_audio_name = f"{label_str}_{timestamp}{audio_ext}"
            audio_dest = os.path.join(self.audio_dir, new_audio_name)
            shutil.copy2(audio_source, audio_dest)

        if image_source is not None:
            image_ext = os.path.splitext(image_source)[1]
            new_image_name = f"{label_str}_{timestamp}{image_ext}"
            image_dest = os.path.join(self.image_dir, new_image_name)
            shutil.copy2(image_source, image_dest)

        if mode is None:
            if audio_source and image_source:
                mode = "fusion"
            elif audio_source:
                mode = "audio"
            else:
                mode = "image"

        self._write_metadata(
            timestamp=timestamp,
            mode=mode,
            labels=labels,
            confidence=confidence,
            audio_path=audio_dest,
            image_path=image_dest,
            scores=scores or {},
        )

        if getattr(config, 'FL_SYNC_ENABLED', False):
            # Sync to FL training directories only when explicitly enabled
            self._sync_to_fl_data(audio_source, image_source, labels, timestamp)

        return audio_dest, image_dest

    def _write_metadata(self, *, timestamp, mode, labels, confidence, audio_path, image_path, scores):
        label_str = "_".join(labels) if labels else "Normal"
        record_id = f"{label_str}_{timestamp}"
        audio_file = os.path.basename(audio_path) if audio_path else None
        image_file = os.path.basename(image_path) if image_path else None

        fieldnames = [
            "record_id",
            "timestamp",
            "mode",
            "label",
            "detected_labels",
            "confidence",
            "audio_file",
            "image_file",
            "audio_path",
            "image_path",
            "audio_score",
            "image_score",
            "fusion_score",
        ]

        file_exists = os.path.exists(self.meta_file)
        with open(self.meta_file, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                "record_id": record_id,
                "timestamp": timestamp,
                "mode": mode,
                "label": label_str,
                "detected_labels": json.dumps(labels, ensure_ascii=False),
                "confidence": confidence,
                "audio_file": audio_file,
                "image_file": image_file,
                "audio_path": audio_path,
                "image_path": image_path,
                "audio_score": scores.get("audio_score"),
                "image_score": scores.get("image_score"),
                "fusion_score": scores.get("fusion_score"),
            })

    def _ensure_next_fl_data_dir(self) -> str:
        base = Path(self.fl_data_dir).resolve()
        if base.name.startswith("fl_data"):
            parent = base.parent
        else:
            parent = base

        max_idx = -1
        for item in parent.iterdir():
            if not item.is_dir():
                continue
            name = item.name
            if name == "fl_data":
                max_idx = max(max_idx, 0)
            elif name.startswith("fl_data_"):
                try:
                    idx = int(name.split("_")[-1])
                    max_idx = max(max_idx, idx)
                except ValueError:
                    continue

        next_idx = max_idx + 1
        next_name = "fl_data" if next_idx == 0 else f"fl_data_{next_idx}"
        next_dir = parent / next_name

        (next_dir / "fl_audio").mkdir(parents=True, exist_ok=True)
        (next_dir / "fl_image").mkdir(parents=True, exist_ok=True)
        (next_dir / "metadata" / "audio_fl").mkdir(parents=True, exist_ok=True)

        return str(next_dir)

    def _sync_to_fl_data(self, audio_source, image_source, labels, timestamp):
        """Sync files to FL data directories with multi-label support."""
        label_str = "_".join(labels) if labels else "Normal"
        # Compute per-class indicators from labels list
        pneu = 1 if "Pneumonia" in labels else 0
        copd = 1 if "COPD_Emphysema" in labels else 0
        fibr = 1 if "Fibrosis" in labels else 0
        crackle = 1 if "Crackle" in labels else 0
        wheeze = 1 if "Wheeze" in labels else 0

        # Image -> fl_image/ with multi-label CSV
        if image_source:
            dest_dir = self.fl_image_dir
            os.makedirs(dest_dir, exist_ok=True)
            ext = os.path.splitext(image_source)[1]
            dest_name = f"{label_str}_{timestamp}{ext}"
            dest = os.path.join(dest_dir, dest_name)
            shutil.copy2(image_source, dest)

            csv_dir = os.path.join(self.fl_data_dir, "metadata", "image_fl")
            os.makedirs(csv_dir, exist_ok=True)
            csv_path = os.path.join(csv_dir, f"client_{self.client_id}_train.csv")

            if not os.path.exists(csv_path):
                with open(csv_path, "w", encoding="utf-8") as f:
                    f.write("path,Pneumonia,COPD_Emphysema,Fibrosis\n")

            with open(csv_path, "a", encoding="utf-8") as f:
                f.write(f"{dest_name},{pneu},{copd},{fibr}\n")

        # Audio -> fl_audio/ + multi-label CSV
        if audio_source:
            audio_dir = os.path.join(self.fl_data_dir, "fl_audio")
            os.makedirs(audio_dir, exist_ok=True)
            ext = os.path.splitext(audio_source)[1]
            new_name = f"{label_str}_{timestamp}{ext}"
            dest = os.path.join(audio_dir, new_name)
            shutil.copy2(audio_source, dest)

            csv_dir = os.path.join(self.fl_data_dir, "metadata", "audio_fl")
            os.makedirs(csv_dir, exist_ok=True)
            csv_path = os.path.join(csv_dir, f"{self.client_tag}_train.csv")

            if not os.path.exists(csv_path):
                with open(csv_path, "w", encoding="utf-8") as f:
                    f.write("path,crackle,wheeze\n")

            with open(csv_path, "a", encoding="utf-8") as f:
                f.write(f"{new_name},{crackle},{wheeze}\n")

    @staticmethod
    def _normalize_client_tag(client_id) -> str:
        if isinstance(client_id, str):
            value = client_id.strip()
            if value.startswith("client_"):
                return value
            if value.isdigit():
                return f"client_{value}"
        return f"client_{client_id}"
