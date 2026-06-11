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

    Uses ASTModel (MIT/ast-finetuned-audioset-10-10-0.4593) with interpolated
    position embeddings for mel-spectrogram input (128 mel bins, max_frames=384).
    Patch grid: 16×16 patch, stride 10 → (128−16)/10+1=12 freq patches,
    (384−16)/10+1=37 time patches → 444 patches + 2 special = 446 tokens.
    """

    def __init__(self, embedding_dim: int = 256, dropout: float = 0.2,
                 n_mels: int = 128, max_frames: int = 384, pretrained: bool = True):
        super().__init__()
        if not pretrained:
            raise ValueError("AST encoder requires pretrained AST backbone")

        from transformers import ASTModel  # lazy import

        self.backbone = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        hidden_size = self.backbone.config.hidden_size  # 768

        # Interpolate position embeddings: (12, 101) → (12, 37)
        # AST pretrained on AudioSet: 128 mel bins × 1024 frames → 12×101 patch grid
        # Our input: 128 mel bins × 384 frames → 12×37 patch grid
        old_pos = self.backbone.embeddings.position_embeddings  # (1, N_old, hidden)
        cls_dist = old_pos[:, :2, :]                             # (1, 2, hidden)  CLS + distillation
        patch = old_pos[:, 2:, :]                                # (1, N_patch_old, hidden)

        # Reshape to spatial grid: (1, hidden, 12, 101)
        patch = patch.transpose(1, 2).reshape(1, hidden_size, 12, 101)

        # Bilinear interpolate to target grid: (1, hidden, 12, 37)
        n_freq_patches = (n_mels - 16) // 10 + 1   # 12 for n_mels=128
        n_time_patches = (max_frames - 16) // 10 + 1  # 37 for max_frames=384
        new_patch = F.interpolate(
            patch, size=(n_freq_patches, n_time_patches),
            mode="bilinear", align_corners=False,
        ).flatten(2).transpose(1, 2)  # (1, N_patch_new, hidden)

        self.backbone.embeddings.position_embeddings = nn.Parameter(
            torch.cat([cls_dist, new_patch], dim=1)
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
        self.num_special = 2  # CLS + distillation tokens

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        """Return L2-normalized embedding (B, embedding_dim).

        Args:
            input_values: (B, T, n_mels) — log-mel spectrogram
        """
        outputs = self.backbone(input_values=input_values, output_hidden_states=True)

        # Multi-layer CLS pooling (last 3 hidden states)
        hidden_states = outputs.hidden_states
        cls_stack = torch.stack([h[:, 0] for h in hidden_states[-3:]], dim=1)  # (B, 3, hidden)
        cls_mean = cls_stack.mean(dim=1)                                         # (B, hidden)

        # Patch pooling (skip CLS + distillation = 2 special tokens)
        patch_tokens = outputs.last_hidden_state[:, self.num_special:, :]        # (B, N_patch, hidden)
        patch = 0.75 * patch_tokens.mean(dim=1) + 0.25 * patch_tokens.max(dim=1).values  # (B, hidden)

        feat = 0.7 * cls_mean + 0.3 * patch  # (B, hidden)
        return F.normalize(self.projection(feat), p=2, dim=-1)

    def freeze_backbone(self, num_freeze: int = 8):
        """Freeze early transformer blocks."""
        for i, layer in enumerate(self.backbone.encoder.layer):
            if i < num_freeze:
                for param in layer.parameters():
                    param.requires_grad = False
        for param in self.backbone.embeddings.parameters():
            param.requires_grad = False
        return self
