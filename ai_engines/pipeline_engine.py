"""PipelineEngine — Điều phối toàn bộ 5 hành động chẩn đoán theo kế hoạch demo.

Models khớp chính xác với kiến trúc từ notebook Stage 4 (pbl7-fl.ipynb).

Kịch bản A (Upload Ảnh X-Quang):
  HĐ1 - Phân loại trực tiếp (Classification)
  HĐ2 - Suy luận Zero-shot chéo (Cross-modal Prediction)
  HĐ3 - Truy xuất bằng chứng (Retrieval)
  HĐ4 - Trí tuệ giải thích được (XAI: Grad-CAM)
  HĐ5 - Chốt kết luận (Late Fusion)

Kịch bản B (Upload File Âm thanh):
  HĐ1 - Phân loại trực tiếp (Classification)
  HĐ2 - Suy luận cấu trúc phổi Zero-shot
  HĐ3 - Truy xuất X-quang (Retrieval)
  HĐ4 - Trí tuệ giải thích được (XAI: Audio Attention)
  HĐ5 - Chốt kết luận (Late Fusion)
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import densenet121, DenseNet121_Weights

# ── Path setup ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

IMAGE_CLASS_NAMES = ["Normal", "Pneumonia", "COPD_Emphysema", "Fibrosis"]
IMAGE_DISEASE_NAMES = ["Pneumonia", "COPD_Emphysema", "Fibrosis"]
AUDIO_CLASS_NAMES = ["Normal", "Crackle", "Wheeze"]
AUDIO_ATTR_NAMES = ["Crackle", "Wheeze"]

# Prototype name to display name mapping
PROTO_DISPLAY = {
    "p_normal_img": "Phổi bình thường (Ảnh)",
    "p_pneumonia": "Viêm phổi",
    "p_emphysema": "COPD/Khí phế thũng",
    "p_fibrosis": "Xơ phổi",
    "p_normal_aud": "Âm thanh bình thường",
    "p_crackle": "Ran nổ (Crackle)",
    "p_wheeze": "Ran rít (Wheeze)",
}

# Prototype name mapping: checkpoint → internal
# Notebook uses p_emphysema, not p_copd
PROTO_NAME_MAP = {
    "p_normal_img": "p_normal_img",
    "p_normal_aud": "p_normal_aud",
    "p_pneumonia": "p_pneumonia",
    "p_emphysema": "p_emphysema",
    "p_fibrosis": "p_fibrosis",
    "p_crackle": "p_crackle",
    "p_wheeze": "p_wheeze",
}

# Audio prototypes output order: [normal_aud, crackle, wheeze]
# Image prototypes output order: [normal_img, pneumonia, emphysema, fibrosis]


# ══════════════════════════════════════════════════════════════════════════════
# Model kiến trúc khớp với notebook Stage 4
# ══════════════════════════════════════════════════════════════════════════════

class DenseNet121EncoderNB(nn.Module):
    """Image encoder khớp chính xác với notebook Stage 4.

    DenseNet121 backbone → AdaptiveAvgPool2d(1,1) → Projection(1024→512→256)
    Projection: Linear → LayerNorm → GELU → Dropout → Linear → LayerNorm
    """

    def __init__(self, embedding_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        backbone = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        proj = self.projection(x)
        return F.normalize(proj, dim=1)


class ASTEncoderNB(nn.Module):
    """Audio encoder khớp chính xác với notebook Stage 4.

    ASTModel (MIT/ast-finetuned-audioset-10-10-0.4593) với interpolated position
    embeddings cho mel-spectrogram (128 mel bins, max_frames=384 → 12×37 patches).
    Projection: Linear → LayerNorm → GELU → Dropout → Linear → LayerNorm
    """

    def __init__(self, embedding_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        from transformers import ASTModel

        self.backbone = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

        # Interpolate position embeddings: (12, 101) → (12, 37)
        old_pos_embed = self.backbone.embeddings.position_embeddings  # (1, N_old, 768)
        hidden_size = self.backbone.config.hidden_size

        # Split: first 2 are special (CLS + distillation), rest are patches
        cls_dist_embed = old_pos_embed[:, :2, :]
        patch_embed = old_pos_embed[:, 2:, :]

        # Reshape to spatial grid: (1, 12, 101, 768)
        patch_embed = patch_embed.transpose(1, 2).reshape(1, hidden_size, 12, 101)

        # Bilinear interpolate to target: 12 × 37
        new_patch_embed = F.interpolate(
            patch_embed, size=(12, 37), mode="bilinear", align_corners=False
        ).flatten(2).transpose(1, 2)

        self.backbone.embeddings.position_embeddings = nn.Parameter(
            torch.cat([cls_dist_embed, new_patch_embed], dim=1)
        )
        self.backbone.embeddings.position_embeddings.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, T, 128) — mel-spectrogram tensor
            attention_mask: (B, N_patches) — optional attention mask
        """
        if attention_mask is not None:
            if attention_mask.ndim == 3:
                attention_mask = attention_mask.squeeze(1)
            elif attention_mask.ndim == 1:
                attention_mask = attention_mask.unsqueeze(0)
            outputs = self.backbone(x, attention_mask=attention_mask, output_hidden_states=True)
        else:
            outputs = self.backbone(x, output_hidden_states=True)
        hidden = outputs.hidden_states

        # Multi-layer CLS pooling (last 3 layers)
        cls_stack = torch.stack([h[:, 0] for h in hidden[-3:]], dim=1)  # (B, 3, 768)
        cls_mean = cls_stack.mean(dim=1)                                  # (B, 768)

        # Patch pooling (after CLS + distillation tokens)
        patch_tokens = outputs.last_hidden_state[:, 2:, :]  # Skip 2 special tokens
        patch = 0.75 * patch_tokens.mean(dim=1) + 0.25 * patch_tokens.max(dim=1).values  # (B, 768)

        feat = 0.7 * cls_mean + 0.3 * patch  # (B, 768)
        proj = self.projection(feat)
        return F.normalize(proj, dim=1)

    def forward_attention(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """Forward với output_attentions=True để lấy attention maps."""
        if attention_mask is not None:
            if attention_mask.ndim == 3:
                attention_mask = attention_mask.squeeze(1)
            elif attention_mask.ndim == 1:
                attention_mask = attention_mask.unsqueeze(0)

        return self.backbone(
            x, attention_mask=attention_mask,
            output_hidden_states=True, output_attentions=True,
        )


class MomentumPrototypeModuleNB(nn.Module):
    """Prototype module khớp chính xác với notebook Stage 4.

    7 prototypes với momentum EMA (m=0.90).
    """

    _NAMES = [
        "p_normal_img", "p_normal_aud",
        "p_pneumonia", "p_emphysema", "p_fibrosis",
        "p_crackle", "p_wheeze",
    ]

    def __init__(self, dim: int = 256, momentum: float = 0.90):
        super().__init__()
        self.dim = dim
        self.momentum = momentum

        for name in self._NAMES:
            setattr(self, name, nn.Parameter(torch.randn(dim)))

        for name in self._NAMES:
            target = F.normalize(getattr(self, name).data.clone(), dim=0)
            self.register_buffer(f"target_{name}", target)

    @torch.no_grad()
    def normalize_and_update(self):
        for name in self._NAMES:
            online = getattr(self, name)
            online.copy_(F.normalize(online.data, dim=0))
            target = getattr(self, f"target_{name}")
            target.mul_(self.momentum).add_(online.data * (1.0 - self.momentum))
            target.copy_(F.normalize(target, dim=0))

    def get_img_protos(self) -> torch.Tensor:
        """Return (4, dim): normal_img, pneumonia, emphysema, fibrosis."""
        return torch.stack([
            self.target_p_normal_img,
            self.target_p_pneumonia,
            self.target_p_emphysema,
            self.target_p_fibrosis,
        ])

    def get_aud_protos(self) -> torch.Tensor:
        """Return (3, dim): normal_aud, crackle, wheeze."""
        return torch.stack([
            self.target_p_normal_aud,
            self.target_p_crackle,
            self.target_p_wheeze,
        ])

    def get_img_class_protos(self) -> torch.Tensor:
        """Return (3, dim): pneumonia, emphysema, fibrosis (no normal)."""
        return torch.stack([
            self.target_p_pneumonia,
            self.target_p_emphysema,
            self.target_p_fibrosis,
        ])

    def get_aud_class_protos(self) -> torch.Tensor:
        """Return (2, dim): crackle, wheeze (no normal)."""
        return torch.stack([
            self.target_p_crackle,
            self.target_p_wheeze,
        ])


# ══════════════════════════════════════════════════════════════════════════════
# Dataclass kết quả
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RetrievalItem:
    file_path: str
    file_name: str
    similarity: float
    case_id: str = ""
    disease_label: str = ""
    acoustic_label: str = ""


@dataclass
class LateFusionResult:
    primary_diagnosis: str
    confidence: float
    confidence_level: str    # "Rất cao" / "Cao" / "Trung bình" / "Thấp"
    agreement: str
    fusion_scores: dict
    is_normal: bool


@dataclass
class ImagePipelineResult:
    disease_probs: dict
    cross_modal_acoustic: dict
    cross_modal_message: str
    retrieved_audio: list[RetrievalItem] = field(default_factory=list)
    heatmap_path: str = ""
    gradcam_enabled: bool = True
    late_fusion: Optional[LateFusionResult] = None
    embedding: Optional[np.ndarray] = None
    timestamp: str = ""
    mode: str = "image"


@dataclass
class AudioPipelineResult:
    acoustic_probs: dict
    cross_modal_disease: dict
    cross_modal_message: str
    retrieved_images: list[RetrievalItem] = field(default_factory=list)
    attention_map_path: str = ""
    attention_enabled: bool = True
    late_fusion: Optional[LateFusionResult] = None
    embedding: Optional[np.ndarray] = None
    timestamp: str = ""
    mode: str = "audio"


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline Engine
# ══════════════════════════════════════════════════════════════════════════════

class PipelineEngine:
    """Engine điều phối toàn bộ pipeline chẩn đoán 5 hành động."""

    def __init__(
        self,
        stage4_path: str,
        image_head_path: str,
        audio_head_path: str,
        audio_db_path: Optional[str] = None,
        image_db_path: Optional[str] = None,
        device: Optional[str] = None,
        threshold: float = 0.5,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold

        # ── Load models ──────────────────────────────────────────────
        self._load_stage4(stage4_path)
        self._load_classifier_heads(image_head_path, audio_head_path)
        self._load_databases(audio_db_path, image_db_path)

        # ── Image transform ──────────────────────────────────────────
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        # ── Audio preprocessing params ───────────────────────────────
        self.target_sr = 16000
        self.max_duration = 15

        print(f"[PipelineEngine] Đã khởi tạo trên {self.device}")
        print(f"[PipelineEngine] Prototypes: {len(self.prototypes._NAMES)}")
        print(f"[PipelineEngine] Image head: {self.num_image_classes} classes")
        print(f"[PipelineEngine] Audio head: {self.num_audio_classes} classes")

    # ═══════════════════════════════════════════════════════════════════
    # Loading
    # ═══════════════════════════════════════════════════════════════════

    def _load_stage4(self, stage4_path: str):
        """Load stage4_best_model.pth: Encoders + Prototypes (notebook architecture)."""
        print(f"[PipelineEngine] Đang load stage4 từ: {stage4_path}")
        checkpoint = torch.load(stage4_path, map_location=self.device, weights_only=False)

        # ── Image Encoder (DenseNet121EncoderNB) ──────────────────
        self.image_encoder = DenseNet121EncoderNB(embedding_dim=256, dropout=0.2)
        img_enc_state = checkpoint["image_encoder"]
        # Load với strict=True vì model và checkpoint có cùng kiến trúc
        missing, unexpected = self.image_encoder.load_state_dict(img_enc_state, strict=False)
        if missing:
            print(f"[PipelineEngine] Image encoder missing: {len(missing)} keys")
        if unexpected:
            print(f"[PipelineEngine] Image encoder unexpected: {len(unexpected)} keys")
        self.image_encoder.to(self.device)
        self.image_encoder.eval()

        # ── Audio Encoder (ASTEncoderNB) ──────────────────────────
        self.audio_encoder = ASTEncoderNB(embedding_dim=256, dropout=0.2)
        aud_enc_state = checkpoint["audio_encoder"]
        missing, unexpected = self.audio_encoder.load_state_dict(aud_enc_state, strict=False)
        if missing:
            print(f"[PipelineEngine] Audio encoder missing: {len(missing)} keys")
        if unexpected:
            print(f"[PipelineEngine] Audio encoder unexpected: {len(unexpected)} keys")
        self.audio_encoder.to(self.device)
        self.audio_encoder.eval()

        # ── Prototype Module ─────────────────────────────────────
        self.prototypes = MomentumPrototypeModuleNB(dim=256, momentum=0.90)
        proto_state = checkpoint["prototype_module"]
        self.prototypes.load_state_dict(proto_state, strict=False)
        self.prototypes.to(self.device)
        self.prototypes.eval()
        self.prototypes.normalize_and_update()

        self.stage4_round = checkpoint.get("round", "unknown")
        print(f"[PipelineEngine] Stage4 round {self.stage4_round} loaded")

    def _load_classifier_heads(self, image_head_path: str, audio_head_path: str):
        """Load Global_Image_best.pth và Global_Audio_best.pth — chỉ lấy head."""
        # ── Image Classifier Head ───────────────────────────────────
        print(f"[PipelineEngine] Loading image head: {image_head_path}")
        img_ckpt = torch.load(image_head_path, map_location=self.device, weights_only=True)

        # Check if checkpoint uses "encoder." prefix (from Global_Image_best.pth)
        head_prefix = "head."
        img_head_keys = [k for k in img_ckpt.keys() if k.startswith(head_prefix)]
        if not img_head_keys:
            print("[PipelineEngine] WARNING: No head.* keys in image checkpoint!")

        # Build classifier head matching notebook architecture
        # head.0 = Linear(256, 128), head.3 = Linear(128, num_classes)
        # The numbers (0, 3) are from nn.Sequential: 0→Linear, 1→ReLU, 2→Dropout, 3→Linear
        self.num_image_classes = img_ckpt["head.3.weight"].shape[0]  # 4

        self.image_classifier_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, self.num_image_classes),
        )
        head_state = {}
        for k, v in img_ckpt.items():
            if k.startswith("head."):
                head_state[k.replace("head.", "")] = v
        self.image_classifier_head.load_state_dict(head_state, strict=True)
        self.image_classifier_head.to(self.device)
        self.image_classifier_head.eval()
        print(f"[PipelineEngine] Image head: {self.num_image_classes} classes OK")

        # ── Audio Classifier Head ───────────────────────────────────
        print(f"[PipelineEngine] Loading audio head: {audio_head_path}")
        aud_ckpt = torch.load(audio_head_path, map_location=self.device, weights_only=True)
        self.num_audio_classes = aud_ckpt["head.3.weight"].shape[0]  # 3

        self.audio_classifier_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, self.num_audio_classes),
        )
        aud_head_state = {}
        for k, v in aud_ckpt.items():
            if k.startswith("head."):
                aud_head_state[k.replace("head.", "")] = v
        self.audio_classifier_head.load_state_dict(aud_head_state, strict=True)
        self.audio_classifier_head.to(self.device)
        self.audio_classifier_head.eval()
        print(f"[PipelineEngine] Audio head: {self.num_audio_classes} classes OK")

    def _load_databases(self, audio_db_path: Optional[str], image_db_path: Optional[str]):
        self.audio_database = None
        self.image_database = None

        if audio_db_path and os.path.exists(audio_db_path):
            self.audio_database = np.load(audio_db_path, allow_pickle=True).item()
            print(f"[PipelineEngine] Audio DB: {len(self.audio_database['embeddings'])} files")

        if image_db_path and os.path.exists(image_db_path):
            self.image_database = np.load(image_db_path, allow_pickle=True).item()
            print(f"[PipelineEngine] Image DB: {len(self.image_database['embeddings'])} files")

    # ═══════════════════════════════════════════════════════════════════
    # Preprocessing
    # ═══════════════════════════════════════════════════════════════════

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        return self.image_transform(image).unsqueeze(0).to(self.device)

    def preprocess_audio(self, audio_path: str) -> torch.Tensor:
        """Audio → mel-spectrogram tensor (1, 128, 384) khớp notebook."""
        import torchaudio
        import soundfile as sf

        try:
            waveform, sr = torchaudio.load(audio_path)
        except Exception:
            waveform_np, sr = sf.read(audio_path)
            waveform = torch.from_numpy(waveform_np).float()
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            elif waveform.ndim == 2:
                waveform = waveform.transpose(0, 1)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
            waveform = resampler(waveform)

        target_samples = self.target_sr * self.max_duration
        current = waveform.shape[1]
        if current < target_samples:
            waveform = F.pad(waveform, (0, target_samples - current))
        else:
            waveform = waveform[:, :target_samples]

        # Mel-spectrogram
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.target_sr, n_fft=1024, hop_length=512, n_mels=128,
        )
        mel_spec = mel_transform(waveform)  # (1, 128, T)

        # Z-score normalize (khớp notebook)
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-6)

        # Pad/trim time dimension to MAX_FRAMES=384
        if mel_spec.shape[-1] < 384:
            mel_spec = F.pad(mel_spec, (0, 384 - mel_spec.shape[-1]))
        else:
            mel_spec = mel_spec[:, :, :384]

        return mel_spec.float().to(self.device)  # (1, 128, 384)

    # ═══════════════════════════════════════════════════════════════════
    # Hành động 1: Phân loại trực tiếp
    # ═══════════════════════════════════════════════════════════════════

    def classify_image(self, image_tensor: torch.Tensor) -> dict:
        with torch.no_grad():
            embedding = self.image_encoder(image_tensor)
            logits = self.image_classifier_head(embedding)
            probs = torch.sigmoid(logits).squeeze(0)
        return {IMAGE_CLASS_NAMES[i]: round(probs[i].item(), 4)
                for i in range(self.num_image_classes)}

    def classify_audio(self, audio_tensor: torch.Tensor) -> dict:
        with torch.no_grad():
            embedding = self.audio_encoder(audio_tensor)
            logits = self.audio_classifier_head(embedding)
            probs = torch.sigmoid(logits).squeeze(0)
        return {AUDIO_CLASS_NAMES[i]: round(probs[i].item(), 4)
                for i in range(self.num_audio_classes)}

    def get_image_embedding(self, image_tensor: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            emb = self.image_encoder(image_tensor)
        return emb.squeeze(0).cpu().numpy()

    def get_audio_embedding(self, audio_tensor: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            emb = self.audio_encoder(audio_tensor)
        return emb.squeeze(0).cpu().numpy()

    # ═══════════════════════════════════════════════════════════════════
    # Hành động 2: Cross-modal Zero-shot
    # ═══════════════════════════════════════════════════════════════════

    def cross_modal_image_to_acoustic(self, image_embedding: np.ndarray) -> dict:
        """Vector ảnh → Cosine similarity với Audio Prototypes → Acoustic prediction."""
        emb_t = F.normalize(torch.from_numpy(image_embedding).float().unsqueeze(0).to(self.device), p=2, dim=-1)
        aud_protos = self.prototypes.get_aud_protos()  # (3, 256): normal_aud, crackle, wheeze
        sims = F.cosine_similarity(emb_t, aud_protos.unsqueeze(0), dim=-1).squeeze(0).cpu().numpy()
        sims_clamped = np.clip(sims, -1, 1)
        return {
            "Crackle": round(float((sims_clamped[1] + 1) / 2), 4),   # crackle proto
            "Wheeze": round(float((sims_clamped[2] + 1) / 2), 4),    # wheeze proto
        }

    def cross_modal_audio_to_disease(self, audio_embedding: np.ndarray) -> dict:
        """Vector âm thanh → Cosine similarity với Image Prototypes → Disease prediction."""
        emb_t = F.normalize(torch.from_numpy(audio_embedding).float().unsqueeze(0).to(self.device), p=2, dim=-1)
        img_protos = self.prototypes.get_img_protos()  # (4, 256): normal, pneumonia, emphysema, fibrosis
        sims = F.cosine_similarity(emb_t, img_protos.unsqueeze(0), dim=-1).squeeze(0).cpu().numpy()
        sims_clamped = np.clip(sims, -1, 1)
        return {
            "Normal": round(float((sims_clamped[0] + 1) / 2), 4),
            "Pneumonia": round(float((sims_clamped[1] + 1) / 2), 4),
            "COPD_Emphysema": round(float((sims_clamped[2] + 1) / 2), 4),
            "Fibrosis": round(float((sims_clamped[3] + 1) / 2), 4),
        }

    def _build_cross_modal_message(self, scores: dict, direction: str = "image_to_acoustic") -> str:
        if direction == "image_to_acoustic":
            crackle = scores.get("Crackle", 0)
            wheeze = scores.get("Wheeze", 0)
            if crackle > wheeze and crackle > 0.3:
                return (f"🔊 Hệ thống dự đoán nếu nghe bằng ống nghe, bệnh nhân này "
                        f"có khả năng phát ra tiếng Ran nổ (Crackle) rất cao ({(crackle*100):.0f}%).")
            elif wheeze > crackle and wheeze > 0.3:
                return (f"🔊 Hệ thống dự đoán nếu nghe bằng ống nghe, bệnh nhân này "
                        f"có khả năng phát ra tiếng Ran rít (Wheeze) rất cao ({(wheeze*100):.0f}%).")
            else:
                return "🔊 Dựa trên ảnh X-quang, chưa có dấu hiệu rõ ràng về bất thường âm thanh phổi."
        else:
            normal = scores.get("Normal", 0)
            pne = scores.get("Pneumonia", 0)
            copd = scores.get("COPD_Emphysema", 0)
            fibr = scores.get("Fibrosis", 0)
            best = max(pne, copd, fibr)
            if normal > best:
                return "🫁 Dựa trên âm thanh phổi, hệ thống dự đoán cấu trúc phổi có vẻ bình thường."
            elif pne >= best:
                return (f"🫁 Dựa trên âm thanh này, hệ thống dự đoán nếu chụp X-quang, "
                        f"hình ảnh cấu trúc phổi sẽ có dấu hiệu của Viêm phổi (Pneumonia) ({(pne*100):.0f}%).")
            elif copd >= best:
                return (f"🫁 Dựa trên tiếng rít (Wheeze) này, hệ thống dự đoán nếu chụp X-quang, "
                        f"hình ảnh cấu trúc phổi sẽ có dấu hiệu của Khí phế thũng (Emphysema) ({(copd*100):.0f}%).")
            else:
                return (f"🫁 Dựa trên âm thanh này, hệ thống dự đoán nếu chụp X-quang, "
                        f"hình ảnh cấu trúc phổi sẽ có dấu hiệu của Xơ phổi (Fibrosis) ({(fibr*100):.0f}%).")

    # ═══════════════════════════════════════════════════════════════════
    # Hành động 3: Retrieval
    # ═══════════════════════════════════════════════════════════════════

    def retrieve_audio(self, image_embedding: np.ndarray, top_k: int = 3) -> list[RetrievalItem]:
        return self._retrieve(image_embedding, self.audio_database, top_k)

    def retrieve_images(self, audio_embedding: np.ndarray, top_k: int = 3) -> list[RetrievalItem]:
        return self._retrieve(audio_embedding, self.image_database, top_k)

    def _retrieve(self, query: np.ndarray, database: Optional[dict], top_k: int = 3) -> list[RetrievalItem]:
        if database is None or "embeddings" not in database:
            return []

        db_emb = database["embeddings"]
        db_files = database.get("files", [])
        db_labels = database.get("labels", [])
        db_cases = database.get("case_ids", [])

        query_norm = query / (np.linalg.norm(query) + 1e-8)
        db_norm = db_emb / (np.linalg.norm(db_emb, axis=1, keepdims=True) + 1e-8)
        similarities = np.dot(db_norm, query_norm)

        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        for idx in top_indices:
            sim = float(similarities[idx])
            if sim < 0.01:
                continue
            fp = db_files[idx] if idx < len(db_files) else ""
            fn = os.path.basename(fp)
            cid = db_cases[idx] if idx < len(db_cases) else f"#{idx+1}"
            lb = db_labels[idx] if idx < len(db_labels) else {}
            dis = lb.get("disease", "") if isinstance(lb, dict) else ""
            aco = lb.get("acoustic", "") if isinstance(lb, dict) else ""
            results.append(RetrievalItem(fp, fn, round(sim, 4), cid, dis, aco))
        return results

    # ═══════════════════════════════════════════════════════════════════
    # Hành động 4: XAI
    # ═══════════════════════════════════════════════════════════════════

    def generate_gradcam(self, image_path: str, image_tensor: torch.Tensor,
                         save_dir: Optional[str] = None) -> str:
        """Grad-CAM cho ảnh X-quang, dùng model encoder + head."""
        wrapper = _GradCAMWrapper(self.image_encoder, self.image_classifier_head)
        wrapper.to(self.device).eval()

        target_layer = self.image_encoder.features.denseblock4.denselayer16.conv2
        gradcam = _GradCAM(wrapper, target_layer)

        with torch.no_grad():
            logits = wrapper(image_tensor)
            probs = torch.sigmoid(logits).squeeze(0)
        best_idx = int(probs.argmax().item())
        best_class = IMAGE_CLASS_NAMES[best_idx]

        heatmap = gradcam.generate(image_tensor, class_idx=best_idx)
        original = Image.open(image_path).convert("RGB")
        overlay_img = gradcam.overlay(original, heatmap)

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            ts = int(time.time() * 1000)
            path = os.path.join(save_dir, f"heatmap_{best_class}_{ts}.png")
            overlay_img.save(path)
            gradcam.remove()
            return path

        gradcam.remove()
        return ""

    def generate_audio_attention(self, audio_path: str, audio_tensor: torch.Tensor,
                                 save_dir: Optional[str] = None) -> str:
        """Audio attention map overlay trên mel-spectrogram."""
        import cv2

        # Forward với attention outputs
        with torch.no_grad():
            outputs = self.audio_encoder.backbone(
                audio_tensor, output_hidden_states=True, output_attentions=True
            )

        # Lấy attention từ layer cuối
        attentions = outputs.attentions  # tuple of (B, num_heads, N, N)
        if attentions:
            last_attn = attentions[-1]  # (1, num_heads, N, N)
            # Trung bình trên tất cả heads
            mean_attn = last_attn.mean(dim=1).squeeze(0)  # (N, N)
            cls_attn = mean_attn[0, 2:]  # CLS → patches, skip CLS+dist
            n_patches = cls_attn.shape[0]
            grid_h = 12  # 128 / ~10.67
            grid_w = n_patches // grid_h if n_patches % grid_h == 0 else 37
            if grid_h * grid_w != n_patches:
                grid_w = n_patches // grid_h
                if grid_h * grid_w < n_patches:
                    grid_w += 1
                    # Truncate
                    cls_attn = cls_attn[:grid_h * grid_w]

            cls_attn = cls_attn[:grid_h * grid_w]
            attn_map = cls_attn.reshape(grid_h, grid_w).cpu().numpy()
        else:
            attn_map = np.ones((12, 37))

        # Normalize
        vmin, vmax = attn_map.min(), attn_map.max()
        if vmax > vmin:
            attn_map = (attn_map - vmin) / (vmax - vmin)

        # Overlay lên mel-spectrogram
        mel_np = audio_tensor.squeeze(0).cpu().numpy()  # (128, 384)

        # Resize attention to match mel shape
        attn_resized = cv2.resize(attn_map, (mel_np.shape[1], mel_np.shape[0]))
        attn_color = cv2.applyColorMap(np.uint8(255 * attn_resized), cv2.COLORMAP_JET)
        attn_color = cv2.cvtColor(attn_color, cv2.COLOR_BGR2RGB)

        mel_vis = ((mel_np - mel_np.min()) / (mel_np.max() - mel_np.min() + 1e-8) * 255).astype(np.uint8)
        mel_rgb = cv2.cvtColor(mel_vis, cv2.COLOR_GRAY2RGB)

        overlay = (mel_rgb * 0.5 + attn_color * 0.5).astype(np.uint8)
        overlay_img = Image.fromarray(overlay)

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            ts = int(time.time() * 1000)
            path = os.path.join(save_dir, f"audio_attention_{ts}.png")
            overlay_img.save(path)
            return path
        return ""

    # ═══════════════════════════════════════════════════════════════════
    # Hành động 5: Late Fusion
    # ═══════════════════════════════════════════════════════════════════

    def late_fusion(self, disease_probs: dict, acoustic_probs: dict) -> LateFusionResult:
        pne_cls = disease_probs.get("Pneumonia", 0)
        copd_cls = disease_probs.get("COPD_Emphysema", 0)
        fibr_cls = disease_probs.get("Fibrosis", 0)
        normal_cls = disease_probs.get("Normal", 0)

        crackle = acoustic_probs.get("Crackle", 0)
        wheeze = acoustic_probs.get("Wheeze", 0)

        # Cross-modal boost (ontology)
        pne_boost = crackle
        copd_boost = wheeze
        fibr_boost = crackle * 0.5

        # Weighted fusion: 70% classifier + 30% cross-modal
        w_cls, w_cm = 0.7, 0.3
        p_pne = w_cls * pne_cls + w_cm * pne_boost
        p_copd = w_cls * copd_cls + w_cm * copd_boost
        p_fib = w_cls * fibr_cls + w_cm * fibr_boost
        p_normal = max(0.0, min(1.0, 1.0 - max(p_pne, p_copd, p_fib)))

        fusion_scores = {
            "Pneumonia": round(p_pne, 4),
            "COPD_Emphysema": round(p_copd, 4),
            "Fibrosis": round(p_fib, 4),
            "Normal": round(p_normal, 4),
        }

        disease_scores = {
            "Viêm phổi (Pneumonia)": p_pne,
            "Khí phế thũng (COPD/Emphysema)": p_copd,
            "Xơ phổi (Fibrosis)": p_fib,
        }
        best_disease, best_score = max(disease_scores.items(), key=lambda x: x[1])

        cls_top = max(pne_cls, copd_cls, fibr_cls)
        cm_top = max(pne_boost, copd_boost, fibr_boost)

        if cls_top >= self.threshold and cm_top >= 0.3:
            agreement = "Có sự đồng thuận từ đặc trưng âm thanh"
            agreement_level = 1.0
        elif cls_top >= self.threshold:
            agreement = "Dựa chủ yếu vào đặc trưng hình ảnh"
            agreement_level = 0.7
        elif cm_top >= 0.3:
            agreement = "Dựa chủ yếu vào suy luận chéo từ âm thanh"
            agreement_level = 0.5
        else:
            agreement = "Độ đồng thuận thấp giữa các modality"
            agreement_level = 0.3

        confidence = best_score * (0.7 + 0.3 * agreement_level)
        if confidence >= 0.85:
            level = "Rất cao"
        elif confidence >= 0.7:
            level = "Cao"
        elif confidence >= 0.5:
            level = "Trung bình"
        else:
            level = "Thấp"

        is_normal = p_normal > max(p_pne, p_copd, p_fib)

        if is_normal:
            primary = "Bình thường (Normal)"
            confidence = p_normal
        else:
            primary = best_disease

        return LateFusionResult(
            primary_diagnosis=primary,
            confidence=round(confidence, 4),
            confidence_level=level,
            agreement=agreement,
            fusion_scores=fusion_scores,
            is_normal=is_normal,
        )

    # ═══════════════════════════════════════════════════════════════════
    # Pipeline Runners
    # ═══════════════════════════════════════════════════════════════════

    def run_image_pipeline(self, image_path: str,
                           save_heatmap_dir: Optional[str] = None) -> ImagePipelineResult:
        """Kịch bản A: Upload Ảnh X-Quang → 5 hành động."""
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        image_tensor = self.preprocess_image(image_path)

        # HĐ1
        disease_probs = self.classify_image(image_tensor)
        print(f"[Pipeline] HĐ1 - Image Classification: {disease_probs}")

        # HĐ2
        image_emb = self.get_image_embedding(image_tensor)
        cross_modal_acoustic = self.cross_modal_image_to_acoustic(image_emb)
        cross_modal_msg = self._build_cross_modal_message(cross_modal_acoustic, "image_to_acoustic")
        print(f"[Pipeline] HĐ2 - Cross-modal: {cross_modal_acoustic}")

        # HĐ3
        retrieved_audio = self.retrieve_audio(image_emb, top_k=3)
        print(f"[Pipeline] HĐ3 - Retrieval: {len(retrieved_audio)} audio files")

        # HĐ4
        heatmap_path = ""
        gradcam_enabled = True
        try:
            heatmap_path = self.generate_gradcam(image_path, image_tensor, save_heatmap_dir)
            print(f"[Pipeline] HĐ4 - GradCAM: {heatmap_path}")
        except Exception as e:
            print(f"[Pipeline] HĐ4 - GradCAM error: {e}")
            gradcam_enabled = False

        # HĐ5
        late_fusion = self.late_fusion(disease_probs, cross_modal_acoustic)
        print(f"[Pipeline] HĐ5 - Late Fusion: {late_fusion.primary_diagnosis} ({late_fusion.confidence_level})")

        return ImagePipelineResult(
            disease_probs=disease_probs,
            cross_modal_acoustic=cross_modal_acoustic,
            cross_modal_message=cross_modal_msg,
            retrieved_audio=retrieved_audio,
            heatmap_path=heatmap_path,
            gradcam_enabled=gradcam_enabled,
            late_fusion=late_fusion,
            embedding=image_emb,
            timestamp=ts,
            mode="image",
        )

    def run_audio_pipeline(self, audio_path: str,
                           save_attention_dir: Optional[str] = None) -> AudioPipelineResult:
        """Kịch bản B: Upload File Âm thanh → 5 hành động."""
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        audio_tensor = self.preprocess_audio(audio_path)

        # HĐ1
        acoustic_probs = self.classify_audio(audio_tensor)
        print(f"[Pipeline] HĐ1 - Audio Classification: {acoustic_probs}")

        # HĐ2
        audio_emb = self.get_audio_embedding(audio_tensor)
        cross_modal_disease = self.cross_modal_audio_to_disease(audio_emb)
        cross_modal_msg = self._build_cross_modal_message(cross_modal_disease, "audio_to_disease")
        print(f"[Pipeline] HĐ2 - Cross-modal: {cross_modal_disease}")

        # HĐ3
        retrieved_images = self.retrieve_images(audio_emb, top_k=3)
        print(f"[Pipeline] HĐ3 - Retrieval: {len(retrieved_images)} images")

        # HĐ4
        attention_path = ""
        attention_enabled = True
        try:
            attention_path = self.generate_audio_attention(audio_path, audio_tensor, save_attention_dir)
            print(f"[Pipeline] HĐ4 - Audio Attention: {attention_path}")
        except Exception as e:
            print(f"[Pipeline] HĐ4 - Audio Attention error: {e}")
            attention_enabled = False

        # HĐ5
        late_fusion = self.late_fusion(cross_modal_disease, acoustic_probs)
        print(f"[Pipeline] HĐ5 - Late Fusion: {late_fusion.primary_diagnosis} ({late_fusion.confidence_level})")

        return AudioPipelineResult(
            acoustic_probs=acoustic_probs,
            cross_modal_disease=cross_modal_disease,
            cross_modal_message=cross_modal_msg,
            retrieved_images=retrieved_images,
            attention_map_path=attention_path,
            attention_enabled=attention_enabled,
            late_fusion=late_fusion,
            embedding=audio_emb,
            timestamp=ts,
            mode="audio",
        )


# ══════════════════════════════════════════════════════════════════════════════
# Internal Grad-CAM helpers
# ══════════════════════════════════════════════════════════════════════════════

class _GradCAMWrapper(nn.Module):
    def __init__(self, encoder, classifier_head):
        super().__init__()
        self.encoder = encoder
        self.features = encoder.features
        self.classifier_head = classifier_head

    def forward(self, x):
        emb = self.encoder(x)
        return self.classifier_head(emb)


class _GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = None
        self.gradients = None
        self._fh = self.target_layer.register_forward_hook(
            lambda m, inp, out: setattr(self, 'feature_maps', out))
        self._bh = self.target_layer.register_full_backward_hook(
            lambda m, gin, gout: setattr(self, 'gradients', gout[0]))

    def remove(self):
        self._fh.remove()
        self._bh.remove()

    def generate(self, input_tensor, class_idx=0):
        import cv2
        self.model.zero_grad()
        output = self.model(input_tensor)
        if output.dim() == 1:
            output = output.unsqueeze(0)
        output[0, class_idx].backward()
        pooled = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = (pooled * self.feature_maps).sum(dim=1).squeeze(0)
        cam = F.relu(cam).detach().cpu().numpy()
        if cam.max() > 0:
            cam /= cam.max()
        return cam

    def overlay(self, pil_image, heatmap, alpha=0.4):
        import cv2
        img = np.array(pil_image.convert("RGB"))
        h, w = img.shape[:2]
        hm = cv2.resize(heatmap, (w, h))
        hm_color = cv2.applyColorMap(np.uint8(255 * hm), cv2.COLORMAP_JET)
        hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)
        return Image.fromarray((img * (1 - alpha) + hm_color * alpha).astype(np.uint8))
