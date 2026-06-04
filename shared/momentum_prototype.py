"""Momentum-based prototype module and memory bank for prototype-guided FL.

Ported from notebook pbl7-fl.ipynb — MomentumPrototypeModule class.

Prototypes:
  - Disease: p_normal_img, p_pneumonia, p_copd, p_fibrosis
  - Acoustic: p_normal_aud, p_crackle, p_wheeze

Uses momentum (EMA) to maintain stable target prototypes during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Momentum Prototype Module
# ---------------------------------------------------------------------------

class MomentumPrototypeModule(nn.Module):
    """Learnable prototypes with momentum-updated target copies.

    Prototype names (7 total):
      Disease side:  p_normal_img,  p_pneumonia,  p_copd,  p_fibrosis
      Acoustic side: p_normal_aud,  p_crackle,    p_wheeze
    """

    _NAMES = [
        "p_normal_img", "p_normal_aud",
        "p_pneumonia", "p_copd", "p_fibrosis",
        "p_crackle", "p_wheeze",
    ]

    def __init__(self, dim: int = 256, momentum: float = 0.99):
        super().__init__()
        self.dim = dim
        self.m = momentum

        # Learnable prototypes
        for name in self._NAMES:
            setattr(self, name, nn.Parameter(torch.randn(dim)))

        # Momentum targets (not updated via optimizer)
        for name in self._NAMES:
            param = getattr(self, name)
            target = F.normalize(param.data.clone(), dim=0)
            self.register_buffer(f"target_{name}", target)

    @torch.no_grad()
    def normalize_and_update(self):
        """Normalize learnable params and update momentum targets (EMA)."""
        for name in self._NAMES:
            param = getattr(self, name)
            # Normalize the learnable parameter
            param.copy_(F.normalize(param.data, dim=0))
            # Update target with momentum
            target = getattr(self, f"target_{name}")
            target.mul_(self.m).add_(param.data * (1.0 - self.m))
            target.copy_(F.normalize(target, dim=0))

    # --- Image-side prototypes (4) ---
    def get_img_protos(self) -> torch.Tensor:
        """Return (4, dim): normal_img, pneumonia, copd, fibrosis."""
        return torch.stack([
            self.target_p_normal_img,
            self.target_p_pneumonia,
            self.target_p_copd,
            self.target_p_fibrosis,
        ])

    def get_img_class_protos(self) -> torch.Tensor:
        """Return (3, dim): pneumonia, copd, fibrosis (no normal)."""
        return torch.stack([
            self.target_p_pneumonia,
            self.target_p_copd,
            self.target_p_fibrosis,
        ])

    # --- Audio-side prototypes (3) ---
    def get_aud_protos(self) -> torch.Tensor:
        """Return (3, dim): normal_aud, crackle, wheeze."""
        return torch.stack([
            self.target_p_normal_aud,
            self.target_p_crackle,
            self.target_p_wheeze,
        ])

    def get_aud_class_protos(self) -> torch.Tensor:
        """Return (2, dim): crackle, wheeze (no normal)."""
        return torch.stack([
            self.target_p_crackle,
            self.target_p_wheeze,
        ])

    # --- All prototypes together ---
    def get_all_protos(self) -> dict[str, torch.Tensor]:
        """Return dict of all 7 target prototypes."""
        return {name: getattr(self, f"target_{name}") for name in self._NAMES}

    def get_all_protos_stacked(self) -> torch.Tensor:
        """Return (7, dim) stacked tensor."""
        return torch.stack([getattr(self, f"target_{name}") for name in self._NAMES])


# ---------------------------------------------------------------------------
# Memory Bank (cross-batch negative mining)
# ---------------------------------------------------------------------------

class MemoryBank:
    """Queue-based memory bank for hard negative mining across batches.

    Ported from notebook pbl7-fl.ipynb MemoryBank class.
    """

    def __init__(self, size: int = 1024, dim: int = 256, num_classes: int = 4, device: str = "cpu"):
        self.size = size
        self.ptr = 0
        self.is_filled = False
        self.device = device
        self.bank = F.normalize(torch.randn(size, dim, device=device), dim=1)
        self.labels = torch.zeros(size, num_classes, device=device)

    def add(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """Enqueue a batch of embeddings and labels."""
        e = embeddings.detach()
        lb = labels.detach()
        b = e.shape[0]

        if self.ptr + b < self.size:
            self.bank[self.ptr:self.ptr + b] = e
            self.labels[self.ptr:self.ptr + b] = lb
            self.ptr += b
        else:
            overflow = (self.ptr + b) - self.size
            self.bank[self.ptr:] = e[:b - overflow]
            self.labels[self.ptr:] = lb[:b - overflow]
            self.bank[:overflow] = e[b - overflow:]
            self.labels[:overflow] = lb[b - overflow:]
            self.ptr = overflow
            self.is_filled = True

    def get(self):
        """Return current bank contents (or partial if not filled)."""
        if self.is_filled:
            return self.bank, self.labels
        if self.ptr > 0:
            return self.bank[:self.ptr], self.labels[:self.ptr]
        return None, None

    def reset(self):
        """Reset the memory bank."""
        self.ptr = 0
        self.is_filled = False
