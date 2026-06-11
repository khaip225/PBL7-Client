"""FL Prototype Model — Encoder + Prototypes + Projection heads.

Ported from notebook pbl7-fl.ipynb (class FederatedClient).

Key design:
  - Image encoder (DenseNet121Encoder): backbone frozen, projection trainable
  - Audio encoder (ASTEncoder): backbone frozen, projection trainable
  - MomentumPrototypeModule: learnable prototypes with EMA targets
  - Memory banks: cross-batch hard negative mining

Shareable (FL sync): prototypes + projection heads
Local (not shared):  encoder backbones
"""

from __future__ import annotations

import copy
import math
import os
import sys
from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Path setup
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from shared.encoder_models import DenseNet121Encoder, ASTEncoder
from shared.momentum_prototype import MomentumPrototypeModule, MemoryBank


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FL_CONFIG = {
    # Architecture
    "embed_dim":        256,
    "dropout":          0.2,
    "n_mels":           128,
    "max_frames":       384,
    "img_size":         224,
    # Training hyperparams (khớp notebook federate_learning.ipynb)
    "temperature":      0.1,
    "memory_bank_size": 1024,
    "lr_backbone":      2e-5,
    "lr_proj":          1e-4,
    "weight_decay":     5e-4,
    "accum_steps":      1,
    # Loss weights
    "contrastive_weight":     1.0,
    "prototype_weight":       1.0,
    "ontology_weight":        1.0,
    "normal_bridge_weight":   0.5,
    "triplet_weight":         0.5,
    "intra_sep_weight":       0.5,
    "inter_sep_weight":       0.5,
    # FedProx
    "use_fedprox":            True,
    "mu":                     0.01,
    # Evaluation
    "patience":               5,
    # Data config
    "img_classes": 4,   # Normal, Pneumonia, COPD/Emphysema, Fibrosis
    "aud_classes": 3,   # Normal, Crackle, Wheeze
}


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def contrastive_loss(embeddings: torch.Tensor, labels: torch.Tensor,
                     mem_feats: torch.Tensor | None, mem_labels: torch.Tensor | None,
                     temperature: float = 0.15) -> torch.Tensor:
    """Supervised contrastive loss with memory bank support.

    Args:
        embeddings: (B, D) L2-normalized
        labels:     (B, num_classes) multi-label float [0,1]
        mem_feats:  (M, D) or None
        mem_labels: (M, num_classes) or None
        temperature: softmax temperature
    """
    if mem_feats is None or mem_labels is None:
        return torch.tensor(0.0, device=embeddings.device)

    all_feats  = torch.cat([embeddings, mem_feats], dim=0)  # (B+M, D)
    all_labels = torch.cat([labels, mem_labels], dim=0)     # (B+M, C)

    sim = torch.clamp(
        torch.matmul(embeddings, all_feats.T) / temperature,
        -50.0, 50.0,
    )  # (B, B+M)

    # Multi-label positive mask: at least one shared label
    mask = (torch.matmul(labels, all_labels.T) > 0).float()  # (B, B+M)
    # Remove self
    eye = torch.eye(embeddings.shape[0], all_feats.shape[0], device=embeddings.device)
    mask = mask * (1.0 - eye)

    # InfoNCE: -log( sum_pos(exp(sim)) / sum_all(exp(sim)) )
    logits  = sim - sim.max(dim=1, keepdim=True)[0].detach()  # stability
    log_prob = logits - torch.log(torch.exp(logits).sum(dim=1, keepdim=True) + 1e-8)

    pos_per_sample = mask.sum(dim=1).clamp(min=1)
    loss = -(mask * log_prob).sum(dim=1) / pos_per_sample
    return loss.mean()


def proto_consistency_loss(embeddings: torch.Tensor, prototypes: torch.Tensor,
                           labels: torch.Tensor, target_sim: float = 0.7) -> torch.Tensor:
    """Pull embeddings toward their positive prototypes."""
    sim = torch.matmul(embeddings, prototypes.T)   # (B, num_protos)
    # Penalize positive pairs below target similarity
    loss_per_class = F.relu(target_sim - sim) * labels  # (B, num_protos)
    class_mask = labels.sum(dim=1).clamp(min=1)          # (B,)
    return (loss_per_class.sum(dim=1) / class_mask).mean()


def ontology_range_loss(proto: MomentumPrototypeModule) -> torch.Tensor:
    """Soft range regularization for ontology-related prototype similarities.

    Ported from notebook ontology_range().
    """
    def r(p1: torch.Tensor, p2: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
        sim = F.cosine_similarity(p1.unsqueeze(0), p2.unsqueeze(0))
        return F.relu(lo - sim) + F.relu(sim - hi)

    return (
        r(proto.p_crackle, proto.p_pneumonia, 0.35, 0.65) +
        r(proto.p_crackle, proto.p_fibrosis,  0.35, 0.65) +
        r(proto.p_wheeze,  proto.p_copd,       0.55, 0.75) +
        r(proto.p_pneumonia, proto.p_fibrosis, -0.1, 0.35) +
        r(proto.p_crackle, proto.p_wheeze,     -0.1, 0.20) +
        r(proto.p_copd, proto.p_pneumonia,     -0.1, 0.25) +
        r(proto.p_copd, proto.p_fibrosis,      -0.1, 0.25)
    ).squeeze()


def normal_bridge_loss(proto: MomentumPrototypeModule, target: float = 0.7) -> torch.Tensor:
    """Pull normal_img and normal_aud prototypes together."""
    sim = F.cosine_similarity(
        proto.p_normal_img.unsqueeze(0),
        proto.p_normal_aud.unsqueeze(0),
    )
    return F.relu(target - sim).squeeze()


def triplet_loss(proto: MomentumPrototypeModule, margin: float = 0.15) -> torch.Tensor:
    """Triplet loss enforcing semantic ordering of prototypes."""
    triplets: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = [
        (proto.p_crackle,    proto.p_pneumonia,  proto.p_wheeze),
        (proto.p_crackle,    proto.p_fibrosis,   proto.p_wheeze),
        (proto.p_wheeze,     proto.p_copd,       proto.p_crackle),
        (proto.p_pneumonia,  proto.p_crackle,    proto.p_copd),
        (proto.p_fibrosis,   proto.p_crackle,    proto.p_copd),
        (proto.p_pneumonia,  proto.p_crackle,    proto.p_fibrosis),
        (proto.p_fibrosis,   proto.p_crackle,    proto.p_pneumonia),
        (proto.p_copd,       proto.p_wheeze,     proto.p_crackle),
    ]

    total = torch.tensor(0.0, device=proto.p_crackle.device)
    for anchor, positive, negative in triplets:
        d_pos = F.cosine_similarity(
            anchor.unsqueeze(0), positive.unsqueeze(0)
        ).squeeze()
        d_neg = F.cosine_similarity(
            anchor.unsqueeze(0), negative.unsqueeze(0)
        ).squeeze()
        total = total + F.relu(d_neg - d_pos + margin)

    return total.squeeze() / len(triplets)


def cross_modal_loss(img_emb: torch.Tensor, aud_emb: torch.Tensor,
                     img_labels: torch.Tensor, aud_labels: torch.Tensor,
                     temperature: float = 0.15) -> torch.Tensor:
    """Cross-modal contrastive loss via ontology bridge.

    img_labels: (B, 4)  =  [Normal, Pneumonia, COPD, Fibrosis]
    aud_labels: (B, 3)  =  [Normal, Crackle, Wheeze]

    Positive mask built via ontology: crackle↔pneumonia+fibrosis, wheeze↔copd, normal↔normal
    """
    # Build cross-modal mask via ontology
    w_normal = torch.matmul(img_labels[:, 0:1], aud_labels[:, 0:1].T)   # Normal↔Normal
    w_crackle = torch.matmul(
        (img_labels[:, 1:2] + img_labels[:, 3:4]).clamp(0, 1),         # Pneumonia+Fibrosis ↔ Crackle
        aud_labels[:, 1:2].T,
    )
    w_wheeze = torch.matmul(img_labels[:, 2:3], aud_labels[:, 2:3].T)   # COPD ↔ Wheeze

    mask = (w_normal + w_crackle + w_wheeze).clamp(0.0, 1.0)  # (B_img, B_aud)
    if mask.sum() < 1e-6:
        return torch.tensor(0.0, device=img_emb.device)

    sim = torch.clamp(
        torch.matmul(img_emb, aud_emb.T) / temperature,
        -50.0, 50.0,
    )  # (B_img, B_aud)

    # Image → Audio direction
    max_i = sim.max(dim=1, keepdim=True)[0].detach()
    lp_i2a = sim - max_i - torch.log(torch.exp(sim - max_i).sum(dim=1, keepdim=True) + 1e-8)
    loss_i2a = -(mask * lp_i2a).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    # Audio → Image direction
    max_a = sim.max(dim=0, keepdim=True)[0].detach()
    lp_a2i = sim - max_a - torch.log(torch.exp(sim - max_a).sum(dim=0, keepdim=True) + 1e-8)
    loss_a2i = -(mask * lp_a2i).sum(dim=0) / mask.sum(dim=0).clamp(min=1)

    return (loss_i2a.mean() + loss_a2i.mean()) / 2.0


def inter_prototype_separation_loss(proto: MomentumPrototypeModule, margin: float = 0.1) -> torch.Tensor:
    """Push image-disease prototypes away from unrelated acoustic prototypes.

    Allowed: crackle↔pneumonia+fibrosis, wheeze↔copd
    Not allowed: crackle↔copd, wheeze↔pneumonia+fibrosis
    """
    img_protos = torch.stack([proto.p_pneumonia, proto.p_copd, proto.p_fibrosis])
    aud_protos = torch.stack([proto.p_crackle, proto.p_wheeze])
    sim = torch.matmul(img_protos, aud_protos.T)  # (3, 2)
    # allowed_high[i,j]=1 means pair should be similar → don't penalize
    allowed_high = torch.tensor([[1, 0], [0, 1], [1, 0]], dtype=torch.float32, device=sim.device)
    not_allowed = 1.0 - allowed_high
    return (F.relu(sim - margin) * not_allowed).mean()


def intra_prototype_separation_loss(proto: MomentumPrototypeModule, margin: float = 0.1,
                                    normal_margin: float = 0.25) -> torch.Tensor:
    """Push same-modality prototypes away from each other.

    Image side: all 4 image prototypes separated + normal_img separated from disease
    Audio side: all 3 audio prototypes separated + normal_aud separated from disease
    """
    # --- Image side ---
    img_protos = torch.stack([proto.p_normal_img, proto.p_pneumonia, proto.p_copd, proto.p_fibrosis])
    img_sim = torch.matmul(img_protos, img_protos.T)  # (4, 4)
    img_mask = torch.ones_like(img_sim) - torch.eye(4, device=img_sim.device)
    img_sep_loss = (F.relu(img_sim - margin) * img_mask).mean()

    disease_img = torch.stack([proto.p_pneumonia, proto.p_copd, proto.p_fibrosis])
    sim_normal_disease_img = torch.matmul(proto.p_normal_img.unsqueeze(0), disease_img.T)
    img_normal_sep_loss = F.relu(sim_normal_disease_img - normal_margin).mean()

    # --- Audio side ---
    aud_protos = torch.stack([proto.p_normal_aud, proto.p_crackle, proto.p_wheeze])
    aud_sim = torch.matmul(aud_protos, aud_protos.T)  # (3, 3)
    aud_mask = torch.ones_like(aud_sim) - torch.eye(3, device=aud_sim.device)
    aud_sep_loss = (F.relu(aud_sim - margin) * aud_mask).mean()

    disease_aud = torch.stack([proto.p_crackle, proto.p_wheeze])
    sim_normal_disease_aud = torch.matmul(proto.p_normal_aud.unsqueeze(0), disease_aud.T)
    aud_normal_sep_loss = F.relu(sim_normal_disease_aud - normal_margin).mean()

    return img_sep_loss + aud_sep_loss + img_normal_sep_loss + aud_normal_sep_loss


def batch_hard_negative_triplet_loss(
    z_img: torch.Tensor | None, z_aud: torch.Tensor | None,
    img_labels: torch.Tensor | None, aud_labels: torch.Tensor | None,
    proto: MomentumPrototypeModule, margin: float = 0.15,
) -> torch.Tensor:
    """Hard-negative triplet: pull embedding toward its correct prototype,
    push away from look-alike prototype.

    Image: pneumonia→pneumonia (pos), fibrosis (neg); fibrosis→fibrosis (pos), pneumonia (neg)
    Audio: crackle→crackle (pos), wheeze (neg); wheeze→wheeze (pos), crackle (neg)
    """
    loss = torch.tensor(0.0, device=proto.p_crackle.device)
    valid_pairs = 0

    if z_img is not None and img_labels is not None:
        # Pneumonia → pos=pneumonia, neg=fibrosis
        pne_mask = img_labels[:, 1] == 1
        if pne_mask.sum() > 0:
            sim_pos = F.cosine_similarity(z_img[pne_mask], proto.target_p_pneumonia.unsqueeze(0))
            sim_neg = F.cosine_similarity(z_img[pne_mask], proto.target_p_fibrosis.unsqueeze(0))
            loss = loss + F.relu(sim_neg - sim_pos + margin).mean()
            valid_pairs += 1

        # Fibrosis → pos=fibrosis, neg=pneumonia
        fib_mask = img_labels[:, 3] == 1
        if fib_mask.sum() > 0:
            sim_pos = F.cosine_similarity(z_img[fib_mask], proto.target_p_fibrosis.unsqueeze(0))
            sim_neg = F.cosine_similarity(z_img[fib_mask], proto.target_p_pneumonia.unsqueeze(0))
            loss = loss + F.relu(sim_neg - sim_pos + margin).mean()
            valid_pairs += 1

    if z_aud is not None and aud_labels is not None:
        # Crackle → pos=crackle, neg=wheeze
        cra_mask = aud_labels[:, 1] == 1
        if cra_mask.sum() > 0:
            sim_pos = F.cosine_similarity(z_aud[cra_mask], proto.target_p_crackle.unsqueeze(0))
            sim_neg = F.cosine_similarity(z_aud[cra_mask], proto.target_p_wheeze.unsqueeze(0))
            loss = loss + F.relu(sim_neg - sim_pos + margin).mean()
            valid_pairs += 1

        # Wheeze → pos=wheeze, neg=crackle
        whe_mask = aud_labels[:, 2] == 1
        if whe_mask.sum() > 0:
            sim_pos = F.cosine_similarity(z_aud[whe_mask], proto.target_p_wheeze.unsqueeze(0))
            sim_neg = F.cosine_similarity(z_aud[whe_mask], proto.target_p_crackle.unsqueeze(0))
            loss = loss + F.relu(sim_neg - sim_pos + margin).mean()
            valid_pairs += 1

    return loss / max(1, valid_pairs)


# ---------------------------------------------------------------------------
# FedAvg Selective Aggregation
# ---------------------------------------------------------------------------

def fedavg_selective(client_states: list[OrderedDict],
                     client_weights: list[float],
                     client_modalities: list[str]) -> OrderedDict:
    """Selective FedAvg: image params only from image/multimodal clients,
    audio params only from audio/multimodal clients, prototypes from all.

    Ported from notebook fedavg_selective().
    """
    client_weights_sqrt = [w ** 0.5 for w in client_weights]
    global_state = OrderedDict()

    # --- Image projection (only image + multimodal clients) ---
    img_pairs = [
        (s, w) for s, w, m in zip(client_states, client_weights_sqrt, client_modalities)
        if m in ("image", "multimodal")
    ]
    img_total_w = sum(w for _, w in img_pairs)
    if img_total_w > 0:
        for k in [k for k in client_states[0] if k.startswith("img_proj.")]:
            global_state[k] = sum(s[k].float() * w for s, w in img_pairs) / img_total_w

    # --- Audio projection (only audio + multimodal clients) ---
    aud_pairs = [
        (s, w) for s, w, m in zip(client_states, client_weights_sqrt, client_modalities)
        if m in ("audio", "multimodal")
    ]
    aud_total_w = sum(w for _, w in aud_pairs)
    if aud_total_w > 0:
        for k in [k for k in client_states[0] if k.startswith("aud_proj.")]:
            global_state[k] = sum(s[k].float() * w for s, w in aud_pairs) / aud_total_w

    # --- Prototypes (all clients) ---
    total_w = sum(client_weights_sqrt)
    for k in [k for k in client_states[0] if k.startswith("proto.")]:
        global_state[k] = sum(
            s[k].float() * w for s, w in zip(client_states, client_weights_sqrt)
        ) / total_w

    return global_state


# ---------------------------------------------------------------------------
# Prototype FL Client (core logic, wrapped by Flower NumPyClient)
# ---------------------------------------------------------------------------

class PrototypeFLClient:
    """Core FL client logic — encoder training with prototype alignment.

    This class contains the training logic; it is wrapped by
    `PrototypeFlowerClient` (a fl.client.NumPyClient subclass) for Flower.
    """

    def __init__(
        self,
        client_id: str,
        modality: str,          # "image", "audio", or "multimodal"
        img_loader=None,
        aud_loader=None,
        device: str = "cpu",
        lr: float | None = None,
        config: dict | None = None,
    ):
        self.client_id = client_id
        self.modality = modality
        self.img_loader = img_loader
        self.aud_loader = aud_loader
        self.device = device
        self.cfg = config or FL_CONFIG

        self.num_samples = (
            (len(img_loader.dataset) if img_loader else 0) +
            (len(aud_loader.dataset) if aud_loader else 0)
        )

        # Build encoders
        self.img_enc = DenseNet121Encoder(
            embedding_dim=self.cfg["embed_dim"],
            dropout=self.cfg["dropout"],
        ).to(device)
        self.aud_enc = ASTEncoder(
            embedding_dim=self.cfg["embed_dim"],
            dropout=self.cfg["dropout"],
        ).to(device)

        # Freeze toàn bộ backbone (khớp notebook: chỉ train projection + prototypes)
        for param in self.img_enc.features.parameters():
            param.requires_grad = False
        for param in self.aud_enc.backbone.parameters():
            param.requires_grad = False

        # Prototype module (momentum=0.90 khớp notebook)
        self.proto = MomentumPrototypeModule(
            dim=self.cfg["embed_dim"],
            momentum=0.90,
        ).to(device)

        # Memory banks
        self.mb_img = MemoryBank(
            size=self.cfg["memory_bank_size"],
            dim=self.cfg["embed_dim"],
            num_classes=self.cfg["img_classes"],
            device=device,
        )
        self.mb_aud = MemoryBank(
            size=self.cfg["memory_bank_size"],
            dim=self.cfg["embed_dim"],
            num_classes=self.cfg["aud_classes"],
            device=device,
        )

        # Per-param-group LR (khớp notebook: backbone 2e-5, proj 1e-4)
        img_backbone_params, img_proj_params, aud_backbone_params, aud_proj_params = [], [], [], []
        for name, param in self.img_enc.named_parameters():
            if "projection" in name:
                img_proj_params.append(param)
            else:
                img_backbone_params.append(param)
        for name, param in self.aud_enc.named_parameters():
            if "projection" in name:
                aud_proj_params.append(param)
            else:
                aud_backbone_params.append(param)

        param_groups = [
            {"params": [p for p in img_backbone_params if p.requires_grad], "lr": self.cfg["lr_backbone"]},
            {"params": [p for p in aud_backbone_params if p.requires_grad], "lr": self.cfg["lr_backbone"]},
            {"params": [p for p in img_proj_params if p.requires_grad],        "lr": self.cfg["lr_proj"]},
            {"params": [p for p in aud_proj_params if p.requires_grad],        "lr": self.cfg["lr_proj"]},
            {"params": [p for p in self.proto.parameters() if p.requires_grad], "lr": self.cfg["lr_proj"]},
        ]
        self.opt = optim.AdamW(param_groups, weight_decay=self.cfg["weight_decay"])

        # Collect all trainable params (for grad clip)
        self._trainable_params = [p for g in param_groups for p in g["params"] if p.requires_grad]

        self.scaler = torch.cuda.amp.GradScaler() if device.startswith("cuda") else None
        self.scheduler = None  # created in train() once steps is known
        self._global_params_cache: dict[str, torch.Tensor] = {}

    # --- State management ---
    def get_sync_state(self) -> OrderedDict:
        """Return shareable state (prototypes + projection heads only)."""
        state = OrderedDict()
        for k, v in self.img_enc.projection.state_dict().items():
            state[f"img_proj.{k}"] = v.detach().cpu()
        for k, v in self.aud_enc.projection.state_dict().items():
            state[f"aud_proj.{k}"] = v.detach().cpu()
        for k, v in self.proto.named_parameters():
            state[f"proto.{k}"] = v.detach().cpu()
        return state

    def get_live_params(self) -> dict[str, torch.Tensor]:
        """Return a dict of trainable parameters (on device, with grad)."""
        params: dict[str, torch.Tensor] = {}
        for k, v in self.img_enc.projection.named_parameters():
            params[f"img_proj.{k}"] = v
        for k, v in self.aud_enc.projection.named_parameters():
            params[f"aud_proj.{k}"] = v
        for k, v in self.proto.named_parameters():
            params[f"proto.{k}"] = v
        return params

    def set_weights(self, global_state: OrderedDict):
        """Load aggregated weights (prototypes + projection heads)."""
        # Image projection
        if self.modality in ("image", "multimodal"):
            sub_sd = {}
            for k, v in global_state.items():
                if k.startswith("img_proj."):
                    sub_sd[k.replace("img_proj.", "")] = v.to(self.device)
            if sub_sd:
                self.img_enc.projection.load_state_dict(sub_sd)

        # Audio projection
        if self.modality in ("audio", "multimodal"):
            sub_sd = {}
            for k, v in global_state.items():
                if k.startswith("aud_proj."):
                    sub_sd[k.replace("aud_proj.", "")] = v.to(self.device)
            if sub_sd:
                self.aud_enc.projection.load_state_dict(sub_sd)

        # Prototypes
        proto_sd = self.proto.state_dict()
        for k, v in global_state.items():
            if k.startswith("proto."):
                proto_sd[k.replace("proto.", "")] = v.to(self.device)
        self.proto.load_state_dict(proto_sd)
        self.proto.normalize_and_update()

        # Cache for FedProx
        if self.cfg["use_fedprox"]:
            self._global_params_cache = {
                kk: vv.detach().clone().to(self.device)
                for kk, vv in global_state.items()
            }

    def _fedprox_term(self) -> torch.Tensor:
        """Compute FedProx proximal term."""
        if not self._global_params_cache:
            return torch.tensor(0.0, device=self.device)
        local = self.get_live_params()
        total = torch.tensor(0.0, device=self.device)
        for k, g_val in self._global_params_cache.items():
            if k in local:
                total += ((local[k] - g_val) ** 2).sum()
        return (self.cfg["mu"] / 2.0) * total

    # --- Training ---
    def train(self, global_state: OrderedDict, local_epochs: int = 1,
              round_idx: int = 0) -> float:
        """Run one round of local training.

        Args:
            global_state: aggregated state from server
            local_epochs: number of local epochs
            round_idx: current FL round (0-based), used for target_sim annealing

        Returns average loss.
        """
        self.set_weights(global_state)

        self.img_enc.train()
        self.aud_enc.train()
        self.proto.train()
        self.img_enc.features.eval()   # backbone frozen
        self.aud_enc.backbone.eval()   # backbone frozen

        self.mb_img.reset()
        self.mb_aud.reset()

        steps = max(
            len(self.img_loader) if self.img_loader else 0,
            len(self.aud_loader) if self.aud_loader else 0,
        )
        if steps == 0:
            return 0.0

        # target_sim annealing (khớp notebook: 0.3→0.6 theo round)
        target_sim_img = min(0.3 + round_idx * 0.02, 0.6)
        target_sim_aud = min(0.3 + round_idx * 0.02, 0.6)

        # Scheduler (CosineWarmup, khớp notebook)
        total_steps = steps * local_epochs
        from transformers import get_cosine_schedule_with_warmup
        self.scheduler = get_cosine_schedule_with_warmup(
            self.opt,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps,
        )

        total_loss_sum = 0.0
        accum_counter = 0

        for _epoch in range(local_epochs):
            iter_i = iter(self.img_loader) if self.img_loader else None
            iter_a = iter(self.aud_loader) if self.aud_loader else None

            for _step in range(steps):
                zi = za = yi = ya = None
                l_con_i = l_con_a = l_pr_i = l_pr_a = l_cr = torch.tensor(0.0, device=self.device)

                # --- Image batch ---
                if iter_i is not None:
                    try:
                        xi, yi = next(iter_i)
                    except StopIteration:
                        iter_i = iter(self.img_loader)
                        xi, yi = next(iter_i)
                    xi, yi = xi.to(self.device), yi.float().to(self.device)

                    if self.scaler:
                        with torch.cuda.amp.autocast():
                            zi = self.img_enc(xi)
                    else:
                        zi = self.img_enc(xi)

                    mf, ml = self.mb_img.get()
                    l_con_i = contrastive_loss(
                        zi, yi, mf, ml, temperature=self.cfg["temperature"],
                    )
                    l_pr_i = proto_consistency_loss(zi, self.proto.get_img_protos(), yi,
                                                    target_sim=target_sim_img)
                    self.mb_img.add(zi, yi)

                # --- Audio batch ---
                if iter_a is not None:
                    try:
                        xa, ya = next(iter_a)
                    except StopIteration:
                        iter_a = iter(self.aud_loader)
                        xa, ya = next(iter_a)
                    xa, ya = xa.to(self.device), ya.float().to(self.device)

                    if self.scaler:
                        with torch.cuda.amp.autocast():
                            za = self.aud_enc(xa)
                    else:
                        za = self.aud_enc(xa)

                    mf, ml = self.mb_aud.get()
                    l_con_a = contrastive_loss(
                        za, ya, mf, ml, temperature=self.cfg["temperature"],
                    )
                    l_pr_a = proto_consistency_loss(za, self.proto.get_aud_protos(), ya,
                                                    target_sim=target_sim_aud)
                    self.mb_aud.add(za, ya)

                # --- Cross-modal (only when both modalities available) ---
                if zi is not None and za is not None:
                    l_cr = cross_modal_loss(zi, za, yi, ya, temperature=self.cfg["temperature"])

                # --- Prototype regularization losses ---
                l_onto      = ontology_range_loss(self.proto)
                l_bridge    = normal_bridge_loss(self.proto)
                l_trip_proto = triplet_loss(self.proto)
                l_trip_batch = batch_hard_negative_triplet_loss(zi, za, yi, ya, self.proto)
                l_inter_sep = inter_prototype_separation_loss(self.proto)
                l_intra_sep = intra_prototype_separation_loss(self.proto)
                l_prox      = self._fedprox_term() if self.cfg["use_fedprox"] else torch.tensor(0.0, device=self.device)

                # Loss formula (khớp notebook)
                loss = (
                    self.cfg["contrastive_weight"]    * (l_con_i + l_con_a)
                    + self.cfg["prototype_weight"]    * (l_pr_i + l_pr_a)
                    + self.cfg["ontology_weight"]     * l_onto
                    + self.cfg["normal_bridge_weight"] * l_bridge
                    + self.cfg["triplet_weight"]      * (l_trip_batch + l_trip_proto)
                    + self.cfg["inter_sep_weight"]    * l_inter_sep
                    + self.cfg["intra_sep_weight"]    * l_intra_sep
                    + l_prox
                ) / self.cfg["accum_steps"]

                if torch.isnan(loss) or torch.isinf(loss):
                    self.opt.zero_grad()
                    continue

                # Gradient accumulation
                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                accum_counter += 1
                if accum_counter % self.cfg["accum_steps"] == 0:
                    if self.scaler:
                        self.scaler.unscale_(self.opt)
                        torch.nn.utils.clip_grad_norm_(self._trainable_params, 1.0)
                        self.scaler.step(self.opt)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self._trainable_params, 1.0)
                        self.opt.step()
                    self.scheduler.step()
                    self.opt.zero_grad()
                    self.proto.normalize_and_update()

                total_loss_sum += loss.item()

        # Handle remaining accumulated gradients
        if accum_counter % self.cfg["accum_steps"] != 0:
            if self.scaler:
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self._trainable_params, 1.0)
                self.scaler.step(self.opt)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self._trainable_params, 1.0)
                self.opt.step()
            self.scheduler.step()
            self.opt.zero_grad()
            self.proto.normalize_and_update()

        avg_loss = total_loss_sum / (local_epochs * steps) if steps > 0 else 0.0
        return avg_loss
