"""Prototype-guided multimodal alignment: learnable prototypes, ontology bridge,
contrastive losses, and memory bank for cross-modal alignment without paired data."""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Learnable Prototypes
# ---------------------------------------------------------------------------

class DiseasePrototypes(nn.Module):
    """Learnable prototypes for disease concepts in shared 256-d space.

    Indices: 0 = P_pneumonia, 1 = P_copd, 2 = P_fibrosis
    """

    def __init__(self, embedding_dim=256):
        super().__init__()
        self.pneumonia = nn.Parameter(torch.empty(embedding_dim))
        self.copd = nn.Parameter(torch.empty(embedding_dim))
        self.fibrosis = nn.Parameter(torch.empty(embedding_dim))
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.pneumonia.unsqueeze(0))
        nn.init.xavier_uniform_(self.copd.unsqueeze(0))
        nn.init.xavier_uniform_(self.fibrosis.unsqueeze(0))

    def forward(self):
        prototypes = torch.stack([self.pneumonia, self.copd, self.fibrosis])
        return F.normalize(prototypes, p=2, dim=-1)


class AcousticPrototypes(nn.Module):
    """Learnable prototypes for acoustic concepts in shared 256-d space.

    Indices: 0 = P_crackle, 1 = P_wheeze
    """

    def __init__(self, embedding_dim=256):
        super().__init__()
        self.crackle = nn.Parameter(torch.empty(embedding_dim))
        self.wheeze = nn.Parameter(torch.empty(embedding_dim))
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.crackle.unsqueeze(0))
        nn.init.xavier_uniform_(self.wheeze.unsqueeze(0))

    def forward(self):
        prototypes = torch.stack([self.crackle, self.wheeze])
        return F.normalize(prototypes, p=2, dim=-1)


# ---------------------------------------------------------------------------
# Ontology Bridge
# ---------------------------------------------------------------------------

class OntologyBridge:
    """Static ontology mapping between acoustic and disease concepts."""

    # crackle -> pneumonia + fibrosis, wheeze -> copd
    ACOUSTIC_TO_DISEASE = [[0, 2], [1]]

    @classmethod
    def get_positive_pairs(cls):
        """Return list of (acoustic_idx, disease_idx) pairs that should be close."""
        return [(a, d) for a, diseases in enumerate(cls.ACOUSTIC_TO_DISEASE) for d in diseases]

    @classmethod
    def get_negative_pairs(cls):
        """Return list of (acoustic_idx, disease_idx) pairs that should be far."""
        positive = set(cls.get_positive_pairs())
        all_pairs = [(a, d) for a in range(2) for d in range(3)]
        return [p for p in all_pairs if p not in positive]

    @classmethod
    def get_regularization_matrix(cls):
        """5x5 matrix (3 disease + 2 acoustic) for prototype-prototype relationships.

        Positive entries = should be similar (ontology-related)
        Zero/negative = should be far apart
        """
        n_total = 5
        matrix = torch.zeros(n_total, n_total)

        # Crackle(3) -> Pneumonia(0), Fibrosis(2)
        matrix[3, 0] = 1.0
        matrix[3, 2] = 0.5
        matrix[0, 3] = 1.0
        matrix[2, 3] = 0.5

        # Wheeze(4) -> COPD(1)
        matrix[4, 1] = 1.0
        matrix[1, 4] = 1.0

        return matrix


# ---------------------------------------------------------------------------
# Contrastive Losses
# ---------------------------------------------------------------------------

def supervised_contrastive_loss(embeddings, labels, prototypes, temperature=0.07):
    """Pull embeddings toward correct prototype(s), push away from others.

    Args:
        embeddings: (B, D) L2-normalized
        labels: (B, num_classes) multi-label, float
        prototypes: (num_classes, D) L2-normalized
        temperature: softmax temperature

    Returns:
        scalar loss
    """
    B, D = embeddings.shape
    num_classes = prototypes.shape[0]

    # Cosine similarity: (B, num_classes)
    sim = torch.matmul(embeddings, prototypes.T) / temperature

    # Multi-label: each sample can have multiple positive prototypes
    # Use binary cross-entropy on cosine similarities (scaled)
    loss = 0.0
    for c in range(num_classes):
        pos_mask = labels[:, c] > 0.5
        if pos_mask.sum() == 0:
            continue
        # Positive samples should have high similarity with prototype c
        pos_sim = sim[pos_mask, c]
        # Negative prototypes for this class
        neg_mask = torch.ones(num_classes, device=embeddings.device, dtype=torch.bool)
        neg_mask[c] = False
        neg_sim = sim[pos_mask][:, neg_mask]

        # InfoNCE-style: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        target = torch.zeros(pos_mask.sum(), dtype=torch.long, device=embeddings.device)
        loss += F.cross_entropy(logits, target)

    return loss / max(num_classes, 1)


def prototype_consistency_loss(disease_protos, acoustic_protos):
    """Ensure prototypes stay well-separated.

    - Minimize cosine similarity between unrelated prototypes
    - Maximize distance between all disease prototypes from each other
    """
    d_sim = torch.matmul(disease_protos, disease_protos.T)
    a_sim = torch.matmul(acoustic_protos, acoustic_protos.T)

    # Penalize high similarity between different disease prototypes
    d_off_diag = d_sim - torch.eye(3, device=d_sim.device) * d_sim.diag()
    a_off_diag = a_sim - torch.eye(2, device=a_sim.device) * a_sim.diag()

    return d_off_diag.abs().mean() + a_off_diag.abs().mean()


def ontology_regularization_loss(disease_protos, acoustic_protos):
    """Enforce ontology bridge relationships in prototype space.

    - Positive pairs (from ontology): minimize 1 - cosine_similarity
    - Negative pairs: push apart with margin
    """
    positive_pairs = OntologyBridge.get_positive_pairs()
    negative_pairs = OntologyBridge.get_negative_pairs()

    loss = 0.0

    for a_idx, d_idx in positive_pairs:
        sim = F.cosine_similarity(
            acoustic_protos[a_idx].unsqueeze(0),
            disease_protos[d_idx].unsqueeze(0)
        )
        loss += (1.0 - sim) ** 2

    for a_idx, d_idx in negative_pairs:
        sim = F.cosine_similarity(
            acoustic_protos[a_idx].unsqueeze(0),
            disease_protos[d_idx].unsqueeze(0)
        )
        loss += F.relu(sim - 0.1) ** 2

    return loss / (len(positive_pairs) + len(negative_pairs))


# ---------------------------------------------------------------------------
# Memory Bank
# ---------------------------------------------------------------------------

class EmbeddingMemoryBank:
    """Queue of past embeddings for hard negative mining across batches."""

    def __init__(self, dim=256, queue_size=1024):
        self.queue_size = queue_size
        self.dim = dim
        self.image_queue = None     # (N, D)
        self.audio_queue = None     # (N, D)
        self.image_labels = None    # (N, num_img_classes)
        self.audio_labels = None    # (N, num_audio_classes)
        self._ptr = 0
        self._filled = 0

    def enqueue_image(self, embeddings, labels):
        B = embeddings.shape[0]
        if self.image_queue is None:
            self.image_queue = torch.zeros(self.queue_size, self.dim)
            self.image_labels = torch.zeros(self.queue_size, labels.shape[1])

        for i in range(B):
            idx = self._ptr % self.queue_size
            self.image_queue[idx] = embeddings[i].detach().cpu()
            self.image_labels[idx] = labels[i].detach().cpu()
            self._ptr += 1
        self._filled = min(self._filled + B, self.queue_size)

    def enqueue_audio(self, embeddings, labels):
        B = embeddings.shape[0]
        if self.audio_queue is None:
            self.audio_queue = torch.zeros(self.queue_size, self.dim)
            self.audio_labels = torch.zeros(self.queue_size, labels.shape[1])

        for i in range(B):
            idx = self._ptr % self.queue_size
            self.audio_queue[idx] = embeddings[i].detach().cpu()
            self.audio_labels[idx] = labels[i].detach().cpu()
            self._ptr += 1
        self._filled = min(self._filled + B, self.queue_size)

    def get_hard_negatives(self, query_embedding, modality='image', topk=16):
        """Find hardest negatives from the opposite modality queue."""
        if modality == 'image' and self.audio_queue is not None and self._filled > 0:
            bank = self.audio_queue[:self._filled].to(query_embedding.device)
        elif modality == 'audio' and self.image_queue is not None and self._filled > 0:
            bank = self.image_queue[:self._filled].to(query_embedding.device)
        else:
            return None

        sim = torch.matmul(query_embedding, bank.T)
        _, indices = sim.topk(min(topk, self._filled), dim=-1)
        return bank[indices]
