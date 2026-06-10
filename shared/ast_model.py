"""AST (Audio Spectrogram Transformer) Multi-label model for lung sound classification.

Matches the architecture from stage5 Kaggle training:
- encoder.backbone: ASTModel (MIT/ast-finetuned-audioset-10-10-0.4593)
- encoder.projection: 768->512->256 with LayerNorm+GELU+Dropout(0.2)
- head: 256->128->num_classes with GELU+Dropout(0.2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ASTMultiLabel(nn.Module):
    def __init__(self, num_classes=3, embedding_dim=256, pretrained=True):
        super().__init__()

        from transformers import ASTModel

        self.encoder = nn.Module()
        if pretrained:
            self.encoder.backbone = ASTModel.from_pretrained(
                "MIT/ast-finetuned-audioset-10-10-0.4593"
            )
        else:
            raise ValueError("AST requires pretrained AST backbone")

        hidden_size = self.encoder.backbone.config.hidden_size

        # Interpolate position embeddings: (12, 101) → (12, 37)
        old_pos = self.encoder.backbone.embeddings.position_embeddings
        cls_dist = old_pos[:, :2, :]
        patch = old_pos[:, 2:, :]
        patch = patch.transpose(1, 2).reshape(1, hidden_size, 12, 101)
        new_patch = F.interpolate(
            patch, size=(12, 37), mode="bilinear", align_corners=False
        ).flatten(2).transpose(1, 2)
        self.encoder.backbone.embeddings.position_embeddings = nn.Parameter(
            torch.cat([cls_dist, new_patch], dim=1)
        )
        self.encoder.backbone.embeddings.position_embeddings.requires_grad = False

        self.encoder.projection = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

        self.head = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x, attention_mask=None):
        if attention_mask is not None:
            if attention_mask.ndim == 3:
                attention_mask = attention_mask.squeeze(1)
            elif attention_mask.ndim == 1:
                attention_mask = attention_mask.unsqueeze(0)

        try:
            out = self.encoder.backbone(
                x, attention_mask=attention_mask, output_hidden_states=True
            )
        except TypeError:
            out = self.encoder.backbone(x, output_hidden_states=True)

        h = out.hidden_states
        cls_feat = torch.stack([h[-1][:, 0], h[-2][:, 0], h[-3][:, 0]], dim=1).mean(dim=1)

        patch_tokens = out.last_hidden_state[:, 2:, :]
        patch = 0.75 * patch_tokens.mean(dim=1) + 0.25 * patch_tokens.max(dim=1).values

        feat = 0.7 * cls_feat + 0.3 * patch
        embedding = self.encoder.projection(feat)
        embedding = F.normalize(embedding, p=2, dim=-1)
        logits = self.head(embedding)
        return logits

    def get_embedding(self, x, attention_mask=None):
        if attention_mask is not None:
            if attention_mask.ndim == 3:
                attention_mask = attention_mask.squeeze(1)
            elif attention_mask.ndim == 1:
                attention_mask = attention_mask.unsqueeze(0)

        try:
            out = self.encoder.backbone(
                x, attention_mask=attention_mask, output_hidden_states=True
            )
        except TypeError:
            out = self.encoder.backbone(x, output_hidden_states=True)

        h = out.hidden_states
        cls_feat = torch.stack([h[-1][:, 0], h[-2][:, 0], h[-3][:, 0]], dim=1).mean(dim=1)

        patch_tokens = out.last_hidden_state[:, 2:, :]
        patch = 0.75 * patch_tokens.mean(dim=1) + 0.25 * patch_tokens.max(dim=1).values

        feat = 0.7 * cls_feat + 0.3 * patch
        embedding = self.encoder.projection(feat)
        return F.normalize(embedding, p=2, dim=-1)


def freeze_early_blocks(model: ASTMultiLabel, num_freeze: int = 8):
    """Freeze early transformer layers for partial fine-tuning."""
    for i, layer in enumerate(model.encoder.backbone.encoder.layer):
        if i < num_freeze:
            for param in layer.parameters():
                param.requires_grad = False
    for param in model.encoder.backbone.embeddings.parameters():
        param.requires_grad = False
    return model
