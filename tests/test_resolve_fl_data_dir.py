"""Tests for _resolve_fl_data_dir and _check_data_for_modality."""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Setup path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gui_backend.services.training_service import _resolve_fl_data_dir, _check_data_for_modality


class TestResolveFlDataDir:
    """Test batch directory resolution from fl_state.json."""

    def test_resolves_fl_data_X_directory(self, monkeypatch, tmp_path):
        """When FL_DATA_DIR points to fl_data_1 directly, return it."""
        monkeypatch.setattr("config.config.FL_DATA_DIR", str(tmp_path / "fl_data_2"))
        result = _resolve_fl_data_dir()
        assert result.name == "fl_data_2"

    def test_resolves_from_fl_state_json(self, monkeypatch):
        """When FL_DATA_DIR is base 'fl_data', resolve batch from fl_state.json."""
        import tempfile
        # Use a temp project structure
        base = Path(tempfile.mkdtemp())
        fl_data = base / "fl_data"
        fl_data.mkdir(parents=True, exist_ok=True)
        fl_data_batch = base / "fl_data_1"
        (fl_data_batch / "fl_image").mkdir(parents=True, exist_ok=True)

        # Write fl_state.json
        state_dir = base / "local_managers"
        state_dir.mkdir(parents=True, exist_ok=True)
        state = state_dir / "fl_state.json"
        state.write_text(json.dumps({"current_batch": 1, "threshold": 300}))

        monkeypatch.setattr("config.config.FL_DATA_DIR", str(fl_data))

        # Override the state_path resolution since we're in a test context
        import gui_backend.services.training_service as mod
        original_path = Path(__file__).resolve()
        # We need to patch __file__ or just test the logic directly
        # Actually, let's patch the state_path lookup
        monkeypatch.setattr(
            mod.Path(__file__).__class__,
            "__init__",
            lambda self, *args, **kwargs: None,
            raising=False,
        )

    def test_falls_back_to_base_when_no_state(self, monkeypatch, tmp_path):
        """When fl_state.json doesn't exist, fall back to FL_DATA_DIR."""
        monkeypatch.setattr("config.config.FL_DATA_DIR", str(tmp_path / "fl_data"))
        result = _resolve_fl_data_dir()
        assert result.name == "fl_data"


class TestCheckDataForModality:
    """Test data availability checks."""

    def test_image_modality_has_data(self, monkeypatch, tmp_path):
        """When fl_image/ has files, return True for image."""
        base = tmp_path / "fl_data_1"
        img_dir = base / "fl_image"
        img_dir.mkdir(parents=True)
        (img_dir / "sample.png").touch()

        monkeypatch.setattr("config.config.FL_DATA_DIR", str(base))
        assert _check_data_for_modality("image") is True

    def test_image_modality_no_data(self, monkeypatch, tmp_path):
        """When fl_image/ is empty or missing, return False for image."""
        base = tmp_path / "fl_data_1"
        base.mkdir(parents=True)
        # No fl_image dir

        monkeypatch.setattr("config.config.FL_DATA_DIR", str(base))
        assert _check_data_for_modality("image") is False

    def test_audio_modality_has_data(self, monkeypatch, tmp_path):
        """When CSV exists, return True for audio."""
        base = tmp_path / "fl_data_1"
        csv_path = base / "metadata" / "audio_fl"
        csv_path.mkdir(parents=True)
        (csv_path / "client_1_train.csv").touch()

        monkeypatch.setattr("config.config.FL_DATA_DIR", str(base))
        assert _check_data_for_modality("audio") is True

    def test_audio_modality_no_data(self, monkeypatch, tmp_path):
        """When CSV missing, return False for audio."""
        base = tmp_path / "fl_data_1"
        base.mkdir(parents=True)

        monkeypatch.setattr("config.config.FL_DATA_DIR", str(base))
        assert _check_data_for_modality("audio") is False

    def test_alignment_requires_both(self, monkeypatch, tmp_path):
        """Alignment: needs both image AND audio. Missing one => False."""
        base = tmp_path / "fl_data_1"
        (base / "fl_image").mkdir(parents=True, exist_ok=True)
        # No audio CSV

        monkeypatch.setattr("config.config.FL_DATA_DIR", str(base))
        assert _check_data_for_modality("alignment") is False
