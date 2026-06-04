"""Dataset loaders for Prototype FL encoder training.

Datasets return:
  - Image: (tensor[3,224,224], labels[4])  → [Normal, Pneumonia, COPD, Fibrosis]
  - Audio: (tensor[3,224,224], labels[3])  → [normal, crackle, wheeze]

The image dataset also returns a mel-spectrogram as 3-channel 224x224 for ViT input.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import config


# ---------------------------------------------------------------------------
# FL state helper
# ---------------------------------------------------------------------------

def _load_fl_state(state_file: Path) -> dict:
    if not state_file.exists():
        return {"current_batch": 0}
    try:
        with state_file.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"current_batch": 0}


def _resolve_fl_base_dir(base_dir: str | None = None, state_file: str | None = None) -> str:
    base = Path(base_dir or config.FL_DATA_DIR).resolve()
    if base.name.startswith("fl_data_"):
        return str(base)

    parent = base.parent if base.name.startswith("fl_data") else base
    state_path = Path(state_file or (Path(__file__).resolve().parent.parent / "local_managers" / "fl_state.json"))
    state = _load_fl_state(state_path)
    batch = int(state.get("current_batch", 0))

    if batch <= 0:
        return str(parent / "fl_data")
    return str(parent / f"fl_data_{batch}")


# ---------------------------------------------------------------------------
# Image Dataset (4-class multi-label) — NIH / local
# ---------------------------------------------------------------------------

# ImageNet stats for DenseNet121
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

IMG_CLASSES = ["Normal", "Pneumonia", "COPD_Emphysema", "Fibrosis"]
IMG_CLASS_COLS = ["Normal", "Pneumonia", "COPD_Emphysema", "Fibrosis"]


class FLImageEncoderDataset(Dataset):
    """Multi-label image dataset for encoder training.

    Expects CSV columns: path, Normal, Pneumonia, COPD_Emphysema, Fibrosis
    """

    def __init__(self, data_dir: str, csv_file: str | None = None,
                 is_train: bool = True, img_size: int = 224):
        self.data_dir = data_dir
        self.is_train = is_train
        self.img_size = img_size
        self.class_cols = IMG_CLASS_COLS

        # CSV is required for multi-label
        if csv_file and os.path.exists(csv_file):
            self.df = pd.read_csv(csv_file)
        else:
            raise FileNotFoundError(
                f"Multi-label image CSV not found: {csv_file}. "
                f"Run diagnosis + review first to generate labeled data."
            )

        # Validate columns
        for col in self.class_cols:
            if col not in self.df.columns:
                self.df[col] = 0.0

        # Transforms
        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomRotation(10),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        try:
            row = self.df.iloc[idx]
            img_path = str(row["path"])
            labels = torch.tensor(
                [float(row.get(c, 0)) for c in self.class_cols],
                dtype=torch.float32,
            )

            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
            return image, labels

        except Exception:
            # Return dummy on file error
            dummy = torch.zeros(3, self.img_size, self.img_size)
            return dummy, torch.zeros(len(self.class_cols), dtype=torch.float32)


def load_image_data(client_id: int, batch_size: int = 32,
                    base_dir: str | None = None, state_file: str | None = None):
    """Load image train + val DataLoaders."""
    base = _resolve_fl_base_dir(base_dir, state_file)
    client_dir = os.path.join(base, "fl_image", f"client_{client_id}")
    if not os.path.exists(client_dir):
        client_dir = os.path.join(base, "fl_image")

    # Find training data
    train_dir = os.path.join(client_dir, "train")
    if not os.path.exists(train_dir):
        train_dir = client_dir

    val_dir = os.path.join(client_dir, "test")
    if not os.path.exists(val_dir):
        val_dir = os.path.join(client_dir, "val")
    if not os.path.exists(val_dir):
        val_dir = client_dir  # fallback: same as train

    # Try CSV
    train_csv = os.path.join(client_dir, "train_labels.csv")
    if not os.path.exists(train_csv):
        # Also try metadata path
        train_csv = os.path.join(base, "metadata", "image_fl", f"client_{client_id}_train.csv")
    val_csv = os.path.join(client_dir, "val_labels.csv")
    if not os.path.exists(val_csv):
        val_csv = os.path.join(base, "metadata", "image_fl", f"client_{client_id}_val.csv")
    # Fallback: use train CSV for val if val doesn't exist
    if not os.path.exists(val_csv):
        print(f"[Image Data] Val CSV not found, using train CSV for validation")
        val_csv = train_csv

    train_ds = FLImageEncoderDataset(
        train_dir,
        csv_file=train_csv if os.path.exists(train_csv) else None,
        is_train=True,
    )
    val_ds = FLImageEncoderDataset(
        val_dir,
        csv_file=val_csv if os.path.exists(val_csv) else None,
        is_train=False,
    )

    # Weighted sampler for class imbalance
    train_loader = _build_loader(train_ds, batch_size, is_train=True)
    val_loader   = _build_loader(val_ds, batch_size, is_train=False)

    print(f"[Image Data] Client {client_id}: train={len(train_ds)}, val={len(val_ds)}")
    if os.path.exists(train_csv):
        print(f"  Train CSV: {train_csv}")
    print(f"  Train dir: {train_dir}")
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Audio Dataset (3-class multi-label) — ICBHI / local
# ---------------------------------------------------------------------------

AUD_CLASSES = ["normal", "crackle", "wheeze"]
AUD_CLASS_COLS = ["normal", "crackle", "wheeze"]

# Label mapping from ICBHI
AUD_LABEL_MAP = {
    "normal":  [1.0, 0.0, 0.0],
    "crackle": [0.0, 1.0, 0.0],
    "wheeze":  [0.0, 0.0, 1.0],
    "both":    [0.0, 1.0, 1.0],
}


class FLAudioEncoderDataset(Dataset):
    """Multi-label audio dataset for encoder training.

    Loads mel-spectrograms from .wav files, converts to 3-channel 224x224 for ViT input.

    Expects CSV columns: path, normal, crackle, wheeze   (or path, label)
    If using 'label' column (old format), converts via AUD_LABEL_MAP.
    """

    def __init__(self, csv_file: str, audio_dir: str, is_train: bool = True,
                 target_sr: int = 16000, n_mels: int = 128, max_length: int = 15,
                 img_size: int = 224):
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"Audio CSV not found: {csv_file}")

        self.df = pd.read_csv(csv_file)
        self.audio_dir = audio_dir
        self.is_train = is_train
        self.target_sr = target_sr
        self.max_length = max_length
        self.img_size = img_size

        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sr,
            n_fft=1024,
            hop_length=512,
            n_mels=n_mels,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB()

        # Audio augmentations
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=10)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=20)

        # Determine label mode
        if "crackle" in self.df.columns and "wheeze" in self.df.columns:
            self._label_mode = "multi"
        elif "normal" in self.df.columns and "crackle" in self.df.columns:
            self._label_mode = "multi"
        elif "label" in self.df.columns:
            self._label_mode = "single"  # old binary format
        else:
            self._label_mode = "none"
            print("[WARNING] Audio CSV has no recognized label columns")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # --- Load audio ---
        wav_path = str(row["path"])
        audio_path = wav_path
        if not os.path.exists(audio_path):
            # Try audio_dir + basename
            fname = os.path.basename(wav_path.replace("\\", "/"))
            audio_path = os.path.join(self.audio_dir, fname)
        if not os.path.exists(audio_path):
            # Return dummy
            dummy = torch.zeros(3, self.img_size, self.img_size)
            return dummy, torch.zeros(3, dtype=torch.float32)

        try:
            import soundfile as sf
            waveform_np, sr = sf.read(audio_path)
            waveform = torch.from_numpy(waveform_np).float()
        except Exception:
            waveform, sr = torchaudio.load(audio_path)

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.ndim == 2:
            waveform = waveform.transpose(0, 1)

        # Mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            waveform = resampler(waveform)

        # Pad/truncate
        target_len = self.target_sr * self.max_length
        cur_len = waveform.shape[1]
        if cur_len < target_len:
            waveform = F.pad(waveform, (0, target_len - cur_len))
        else:
            waveform = waveform[:, :target_len]

        # Mel spectrogram
        spec = self.mel_spec(waveform)       # (1, n_mels, time)
        spec_db = self.to_db(spec)

        # Normalize
        mean = spec_db.mean()
        std = spec_db.std()
        spec_db = (spec_db - mean) / (std + 1e-6)

        # Augmentation
        if self.is_train:
            spec_db = self.freq_mask(spec_db)
            spec_db = self.time_mask(spec_db)

        # Convert to 3-channel 224x224 for ViT
        spec_3ch = spec_db.repeat(3, 1, 1)                    # (3, 128, T)
        spec_3ch = F.interpolate(
            spec_3ch.unsqueeze(0),
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)  # (3, 224, 224)

        # Normalize to ImageNet-like range
        spec_3ch = (spec_3ch - spec_3ch.mean()) / (spec_3ch.std() + 1e-6)

        # --- Labels ---
        if self._label_mode == "multi":
            normal_val  = float(row.get("normal", 0))
            crackle_val = float(row.get("crackle", 0))
            wheeze_val  = float(row.get("wheeze", 0))
            # Ensure normal = 1 when neither crackle nor wheeze
            if normal_val == 0 and crackle_val == 0 and wheeze_val == 0:
                normal_val = 1.0
            labels = torch.tensor([normal_val, crackle_val, wheeze_val], dtype=torch.float32)
        elif self._label_mode == "single":
            lbl = str(row["label"]).lower()
            labels = torch.tensor(AUD_LABEL_MAP.get(lbl, [1.0, 0.0, 0.0]), dtype=torch.float32)
        else:
            labels = torch.zeros(3, dtype=torch.float32)

        return spec_3ch, labels


def load_audio_data(client_id: int, batch_size: int = 16,
                    base_dir: str | None = None, state_file: str | None = None):
    """Load audio train + val DataLoaders."""
    base = _resolve_fl_base_dir(base_dir, state_file)
    audio_dir = os.path.join(base, "fl_audio")

    # Check metadata path first, then fallback to fl_audio root
    train_csv = os.path.join(base, "metadata", "audio_fl", f"client_{client_id}_train.csv")
    if not os.path.exists(train_csv):
        train_csv = os.path.join(audio_dir, "train_labels.csv")
    val_csv = os.path.join(base, "metadata", "audio_fl", f"client_{client_id}_val.csv")
    if not os.path.exists(val_csv):
        val_csv = os.path.join(audio_dir, "val_labels.csv")
    # Fallback: use train CSV for val if val doesn't exist
    if not os.path.exists(val_csv):
        print(f"[Audio Data] Val CSV not found, using train CSV for validation")
        val_csv = train_csv

    train_ds = FLAudioEncoderDataset(train_csv, audio_dir, is_train=True)
    val_ds = FLAudioEncoderDataset(val_csv, audio_dir, is_train=False)

    train_loader = _build_loader(train_ds, batch_size, is_train=True)
    val_loader   = _build_loader(val_ds, batch_size, is_train=False)

    print(f"[Audio Data] Client {client_id}: train={len(train_ds)}, val={len(val_ds)}")
    print(f"  Train CSV: {train_csv}")
    print(f"  Audio dir: {audio_dir}")
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Multimodal dataset (for alignment/multimodal clients)
# ---------------------------------------------------------------------------

def load_multimodal_data(client_id: int, batch_size_img: int = 32, batch_size_aud: int = 16,
                         base_dir: str | None = None, state_file: str | None = None):
    """Load both image and audio loaders."""
    img_train, img_val = load_image_data(client_id, batch_size_img, base_dir, state_file)
    aud_train, aud_val = load_audio_data(client_id, batch_size_aud, base_dir, state_file)
    return (img_train, img_val), (aud_train, aud_val)


# ---------------------------------------------------------------------------
# Helper: weighted sampler + DataLoader
# ---------------------------------------------------------------------------

def _build_loader(dataset: Dataset, batch_size: int, is_train: bool) -> DataLoader:
    """Build DataLoader with optional WeightedRandomSampler for class imbalance."""
    if is_train and len(dataset) > 0:
        # Compute sample weights from label distribution
        try:
            sample_weights = _compute_sample_weights(dataset)
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
            return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                              num_workers=0, pin_memory=True, drop_last=True)
        except Exception:
            pass
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train,
                      num_workers=0, pin_memory=True, drop_last=is_train)


def _compute_sample_weights(dataset: Dataset) -> list[float]:
    """Compute per-sample weights based on inverse class frequency."""
    # Image dataset (FLImageEncoderDataset) — always has df and class_cols
    if hasattr(dataset, 'class_cols') and hasattr(dataset, 'df') and dataset.df is not None:
        df = dataset.df
        class_cols = dataset.class_cols
        freqs = np.array([float(df[c].sum()) for c in class_cols])
    elif hasattr(dataset, 'df') and dataset.df is not None:
        # Audio dataset (FLAudioEncoderDataset)
        df = dataset.df
        if hasattr(dataset, '_label_mode') and dataset._label_mode == "multi":
            freqs = np.array([float(df.get(c, pd.Series([0] * len(df))).sum())
                              for c in AUD_CLASS_COLS])
        else:
            freqs = np.array([len(df)] * 3)
    else:
        return [1.0] * len(dataset)

    freqs = np.maximum(freqs, 1.0)
    class_weights = 1.0 / freqs

    # Image dataset weighting
    if hasattr(dataset, 'class_cols') and hasattr(dataset, 'df') and dataset.df is not None:
        weights = []
        for _, row in dataset.df.iterrows():
            ws = [class_weights[i] for i, c in enumerate(dataset.class_cols) if float(row.get(c, 0)) > 0.5]
            weights.append(np.mean(ws) if ws else 1.0)
        return weights

    # Audio dataset weighting
    if hasattr(dataset, 'df') and dataset.df is not None:
        weights = []
        for _, row in dataset.df.iterrows():
            if hasattr(dataset, '_label_mode') and dataset._label_mode == "multi":
                ws = [class_weights[i] for i, c in enumerate(AUD_CLASS_COLS) if float(row.get(c, 0)) > 0.5]
                weights.append(np.mean(ws) if ws else 1.0)
            else:
                weights.append(1.0)
        return weights

    return [1.0] * len(dataset)
