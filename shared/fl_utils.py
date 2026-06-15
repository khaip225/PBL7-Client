"""Shared FL utilities — resolve batch data directory, check data availability."""

import json
from pathlib import Path


def resolve_fl_data_dir(base_dir: str | None = None, state_file: str | None = None) -> Path:
    """Resolve the current batch data directory from fl_state.json.

    All 3 callers (dataset_loader, monitor, training_service) use the same logic.
    """
    base = Path(base_dir or "Local_Data/fl_data").resolve()
    if base.name.startswith("fl_data_"):
        return base

    parent = base.parent if base.name.startswith("fl_data") else base
    if state_file is None:
        repo_root = Path(__file__).resolve().parent.parent
        state_path = repo_root / "local_managers" / "fl_state.json"
    else:
        state_path = Path(state_file)

    try:
        if state_path.exists():
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            batch = int(state.get("current_batch", 0))
            if batch > 0:
                resolved = parent / f"fl_data_{batch}"
                if resolved.exists():
                    return resolved
    except (OSError, json.JSONDecodeError, ValueError):
        pass

    return base


def check_data_for_modality(base: Path, modality: str) -> bool:
    """Check if a modality has training data in the given base directory."""
    if modality == "audio":
        csv_path = base / "metadata" / "audio_fl" / "client_1_train.csv"
        return csv_path.exists()
    elif modality == "image":
        img_dir = base / "fl_image"
        return img_dir.exists() and any(img_dir.iterdir())
    elif modality == "alignment":
        has_img = (base / "fl_image").exists() and any((base / "fl_image").iterdir())
        has_aud = (base / "metadata" / "audio_fl" / "client_1_train.csv").exists()
        return has_img and has_aud
    return False
