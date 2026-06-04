"""Encoder-only models for prototype FL — no classification head.

Port from notebook pbl7-fl.ipynb:
  - DenseNet121Encoder: Image encoder (1024→512→256), L2-normalized
  - ASTEncoder: Audio encoder (768→512→256), L2-normalized

Only encoder + projection — no classifier. The projection head is what gets
shared in FL, while the backbone (features/transformer) stays local.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet121, DenseNet121_Weights


# ---------------------------------------------------------------------------
# Image Encoder
# ---------------------------------------------------------------------------

class DenseNet121Encoder(nn.Module):
    """DenseNet121 encoder producing 256-d L2-normalized embeddings."""

    def __init__(self, embedding_dim: int = 256, dropout: float = 0.2, pretrained: bool = True):
        super().__init__()
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = densenet121(weights=weights)
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
        """Return L2-normalized embedding (B, embedding_dim)."""
        feat = self.pool(F.relu(self.features(x), inplace=True))
        feat = torch.flatten(feat, 1)
        return F.normalize(self.projection(feat), p=2, dim=-1)

    def freeze_backbone(self, keep_last_block: bool = True):
        """Freeze all feature blocks except the last denseblock."""
        for name, param in self.features.named_parameters():
            if keep_last_block and "denseblock4" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        return self


# ---------------------------------------------------------------------------
# Audio Encoder
# ---------------------------------------------------------------------------

class ASTEncoder(nn.Module):
    """AST encoder producing 256-d L2-normalized embeddings.

    Uses ViT backbone (deit-base-distilled-patch16-224) with interpolated
    position embeddings for mel-spectrogram input (128 mel bins, max_frames=384).
    """

    def __init__(self, embedding_dim: int = 256, dropout: float = 0.2,
                 n_mels: int = 128, max_frames: int = 384, pretrained: bool = True):
        super().__init__()
        if not pretrained:
            raise ValueError("AST encoder requires pretrained ViT backbone")

        from transformers import ViTModel  # lazy import

        self.backbone = ViTModel.from_pretrained("facebook/deit-base-distilled-patch16-224")

        # Input is mel-spectrogram resized to 224x224, so 14x14 patches = 196 + CLS = 197 tokens
        # Position embeddings from pretrained model are already correct for this size
        # No interpolation needed since input is 224x224 (same as pretrained)
        self.backbone.embeddings.position_embeddings.requires_grad = False
        self.num_special = 1  # CLS token (ViTModel from deit has 1 special token)

        self.h_patches = 14  # 224 / 16
        self.w_patches = 14  # 224 / 16
        self.n_mels = n_mels
        self.max_frames = max_frames

        self.projection = nn.Sequential(
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return L2-normalized embedding (B, embedding_dim).

        Args:
            pixel_values: (B, 3, 224, 224) — mel-spectrogram as RGB image
        """
        outputs = self.backbone(pixel_values=pixel_values, output_hidden_states=True)

        # Multi-layer mean pooling (last 3 CLS tokens + average all patches)
        hidden_states = outputs.hidden_states
        cls_stack = torch.stack([h[:, 0] for h in hidden_states[-3:]], dim=1)  # (B, 3, 768)
        cls_mean = cls_stack.mean(dim=1)                                         # (B, 768)
        patch_mean = outputs.last_hidden_state[:, self.num_special:, :].mean(dim=1)   # (B, 768)
        pooled = 0.7 * cls_mean + 0.3 * patch_mean                               # (B, 768)

        return F.normalize(self.projection(pooled), p=2, dim=-1)

    def freeze_backbone(self, num_freeze: int = 8):
        """Freeze early transformer blocks."""
        for i, layer in enumerate(self.backbone.encoder.layer):
            if i < num_freeze:
                for param in layer.parameters():
                    param.requires_grad = False
        for param in self.backbone.embeddings.parameters():
            param.requires_grad = False
        return self
