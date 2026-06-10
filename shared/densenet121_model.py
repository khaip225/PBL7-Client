"""DenseNet121 Multi-label model for chest X-ray disease classification.

Matches the architecture from stage5 Kaggle training:
- encoder.features: DenseNet121 backbone (pretrained)
- encoder.projection: 1024->512->256 with LayerNorm+GELU+Dropout(0.2)
- head: 256->128->num_classes with GELU+Dropout(0.2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet121, DenseNet121_Weights


class DenseNet121MultiLabel(nn.Module):
    def __init__(self, num_classes=4, embedding_dim=256, pretrained=True):
        super().__init__()

        densenet = densenet121(weights=DenseNet121_Weights.DEFAULT if pretrained else None)
        self.encoder = nn.Module()
        self.encoder.features = densenet.features
        # NO extra norm5 — notebook does NOT have this

        self.encoder.projection = nn.Sequential(
            nn.Linear(1024, 512),
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

    def forward(self, x):
        features = self.encoder.features(x)
        features = F.relu(features, inplace=True)
        features = F.adaptive_avg_pool2d(features, (1, 1))
        features = torch.flatten(features, 1)
        embedding = self.encoder.projection(features)
        embedding = F.normalize(embedding, p=2, dim=-1)
        logits = self.head(embedding)
        return logits

    def get_embedding(self, x):
        features = self.encoder.features(x)
        features = F.relu(features, inplace=True)
        features = F.adaptive_avg_pool2d(features, (1, 1))
        features = torch.flatten(features, 1)
        embedding = self.encoder.projection(features)
        return F.normalize(embedding, p=2, dim=-1)


def freeze_backbone(model: DenseNet121MultiLabel):
    """Freeze early DenseNet blocks for transfer learning."""
    frozen = 0
    for name, param in model.encoder.features.named_parameters():
        if 'denseblock4' not in name:
            param.requires_grad = False
            frozen += 1
    return model
