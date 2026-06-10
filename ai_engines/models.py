"""NN model classes matching notebook Stage 4 architecture."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet121, DenseNet121_Weights


# ---------------------------------------------------------------------------
# Image Encoder (Stage 4)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Audio Encoder (Stage 4)
# ---------------------------------------------------------------------------

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
        # transformers >=4.49: ASTModel.forward() chỉ nhận input_values, không có attention_mask
        outputs = self.backbone(input_values=x, output_hidden_states=True)
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
        return self.backbone(
            input_values=x,
            output_hidden_states=True, output_attentions=True,
        )


# ---------------------------------------------------------------------------
# Prototype Module (Stage 4)
# ---------------------------------------------------------------------------

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
