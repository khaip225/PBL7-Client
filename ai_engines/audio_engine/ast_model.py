"""AST (Audio Spectrogram Transformer) Multi-label model for lung sound classification (Client).

Matches the architecture from stage5 Kaggle training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ASTMultiLabel(nn.Module):
    def __init__(self, num_classes=2, embedding_dim=256, pretrained=True):
        super().__init__()

        self.encoder = nn.Module()
        if pretrained:
            try:
                from transformers import ViTModel
                self.encoder.backbone = ViTModel.from_pretrained(
                    "facebook/deit-base-distilled-patch16-224"
                )
            except ImportError:
                raise ImportError(
                    "transformers library required. Install: pip install transformers"
                )
        else:
            raise ValueError("AST requires pretrained ViT backbone")

        self.encoder.projection = nn.Sequential(
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

        self.head = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        outputs = self.encoder.backbone(pixel_values=x)
        cls_token = outputs.last_hidden_state[:, 0, :]
        embedding = self.encoder.projection(cls_token)
        embedding = F.normalize(embedding, p=2, dim=-1)
        logits = self.head(embedding)
        return logits

    def get_embedding(self, x):
        outputs = self.encoder.backbone(pixel_values=x)
        cls_token = outputs.last_hidden_state[:, 0, :]
        embedding = self.encoder.projection(cls_token)
        return F.normalize(embedding, p=2, dim=-1)


def freeze_early_blocks(model: ASTMultiLabel, num_freeze: int = 8):
    for i, layer in enumerate(model.encoder.backbone.encoder.layer):
        if i < num_freeze:
            for param in layer.parameters():
                param.requires_grad = False
    for param in model.encoder.backbone.embeddings.parameters():
        param.requires_grad = False
    return model
