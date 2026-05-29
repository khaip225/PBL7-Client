"""FL Prototype Model wrapper.

Combines image encoder + audio encoder + prototypes + projection heads.
Provides methods to extract only shareable parameters for FL aggregation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
# Ensure project root is on path (PBL7-Client/)
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from shared.densenet121_model import DenseNet121MultiLabel, freeze_backbone
from shared.ast_model import ASTMultiLabel, freeze_early_blocks
from shared.prototypes import DiseasePrototypes, AcousticPrototypes


class FLPrototypeModel(nn.Module):
    """Wrapper model for prototype-based federated learning.

    Shareable parameters (synced across clients):
      - disease_protos (3 x 256)
      - acoustic_protos (2 x 256)
      - img_projection (1024 -> 256)
      - aud_projection (768 -> 256)

    Local parameters (not shared):
      - image_encoder backbone (DenseNet121 features)
      - audio_encoder backbone (ViT transformer)
    """

    def __init__(self, image_pretrained_path=None, audio_pretrained_path=None):
        super().__init__()

        self.image_encoder = DenseNet121MultiLabel(num_classes=3, pretrained=True)
        self.audio_encoder = ASTMultiLabel(num_classes=2, pretrained=True)

        if image_pretrained_path and os.path.exists(image_pretrained_path):
            state = torch.load(image_pretrained_path, map_location="cpu")
            self.image_encoder.load_state_dict(state, strict=False)

        if audio_pretrained_path and os.path.exists(audio_pretrained_path):
            state = torch.load(audio_pretrained_path, map_location="cpu")
            self.audio_encoder.load_state_dict(state, strict=False)

        freeze_backbone(self.image_encoder)
        freeze_early_blocks(self.audio_encoder)

        self.disease_protos = DiseasePrototypes(embedding_dim=256)
        self.acoustic_protos = AcousticPrototypes(embedding_dim=256)

    def shareable_parameters(self):
        """Return only parameters that should be aggregated in FL."""
        params = []
        params.extend(list(self.disease_protos.parameters()))
        params.extend(list(self.acoustic_protos.parameters()))
        # Projection heads
        for name, param in self.image_encoder.encoder.projection.named_parameters():
            params.append(param)
        for name, param in self.audio_encoder.encoder.projection.named_parameters():
            params.append(param)
        return params

    def shareable_state_dict(self):
        """Return state_dict with only shareable parameters."""
        sd = {}
        for prefix, module in [
            ("disease_protos", self.disease_protos),
            ("acoustic_protos", self.acoustic_protos),
            ("img_projection", self.image_encoder.encoder.projection),
            ("aud_projection", self.audio_encoder.encoder.projection),
        ]:
            for k, v in module.state_dict().items():
                sd[f"{prefix}.{k}"] = v
        return sd

    def load_shareable_state_dict(self, sd):
        """Load only shareable parameters from aggregated state_dict."""
        for prefix, module in [
            ("disease_protos", self.disease_protos),
            ("acoustic_protos", self.acoustic_protos),
            ("img_projection", self.image_encoder.encoder.projection),
            ("aud_projection", self.audio_encoder.encoder.projection),
        ]:
            sub_sd = {}
            prefix_len = len(prefix) + 1
            for k, v in sd.items():
                if k.startswith(prefix + "."):
                    sub_sd[k[prefix_len:]] = v
            if sub_sd:
                module.load_state_dict(sub_sd, strict=False)

    def get_embeddings(self, image_batch, audio_batch):
        """Get L2-normalized embeddings from both modalities."""
        img_emb = self.image_encoder.get_embedding(image_batch)
        aud_emb = self.audio_encoder.get_embedding(audio_batch)
        return img_emb, aud_emb
