"""t-SNE visualization utilities for embedding space analysis.

Computes t-SNE projections of image and audio embeddings for monitoring
the shared latent space during training.
"""

import os
import numpy as np
from typing import Optional


def compute_tsne(embeddings: np.ndarray, perplexity: float = 30.0, random_state: int = 42) -> np.ndarray:
    """Compute t-SNE 2D projection of embeddings.

    Args:
        embeddings: (N, D) array of embeddings
        perplexity: t-SNE perplexity parameter
        random_state: random seed

    Returns:
        (N, 2) array of 2D coordinates
    """
    try:
        from sklearn.manifold import TSNE
        n = embeddings.shape[0]
        # Adjust perplexity if needed
        perp = min(perplexity, max(1.0, (n - 1) / 3))
        tsne = TSNE(n_components=2, perplexity=perp, random_state=random_state, n_iter=300, verbose=0)
        coords = tsne.fit_transform(embeddings)
        return coords
    except ImportError:
        raise ImportError("scikit-learn required for t-SNE. Install: pip install scikit-learn")


def compute_joint_tsne(
    image_embeddings: np.ndarray,
    audio_embeddings: np.ndarray,
    perplexity: float = 30.0,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Joint t-SNE of image + audio embeddings in shared space.

    Projects all embeddings together so they occupy the same 2D space,
    enabling visual comparison of cross-modal alignment.

    Args:
        image_embeddings: (N_img, D)
        audio_embeddings: (N_aud, D)

    Returns:
        (image_coords (N_img, 2), audio_coords (N_aud, 2))
    """
    all_embeddings = np.concatenate([image_embeddings, audio_embeddings], axis=0)
    all_coords = compute_tsne(all_embeddings, perplexity=perplexity, random_state=random_state)
    n_img = image_embeddings.shape[0]
    return all_coords[:n_img], all_coords[n_img:]


def compute_tsne_visualization(
    image_embeddings: np.ndarray,
    audio_embeddings: np.ndarray,
    image_labels: np.ndarray,
    audio_labels: np.ndarray,
    image_class_names: Optional[list[str]] = None,
    audio_class_names: Optional[list[str]] = None,
) -> dict:
    """Compute full t-SNE visualization data for both modalities.

    Returns a dict ready for JSON serialization and frontend scatter plot.
    """
    # Joint t-SNE
    img_coords, aud_coords = compute_joint_tsne(image_embeddings, audio_embeddings)

    img_names = image_class_names or ["Class_0", "Class_1", "Class_2"]
    aud_names = audio_class_names or ["Class_0", "Class_1"]

    # For multi-label, find the dominant class (argmax)
    def get_dominant(labels, names):
        if labels.ndim == 1:
            return [names[int(l)] for l in labels]
        dominant = []
        for row in labels:
            max_idx = int(np.argmax(row))
            dominant.append(names[max_idx] if max_idx < len(names) else f"Class_{max_idx}")
        return dominant

    img_classes = get_dominant(image_labels, img_names)
    aud_classes = get_dominant(audio_labels, aud_names)

    return {
        "image_points": [
            {"x": float(c[0]), "y": float(c[1]), "class": cls}
            for c, cls in zip(img_coords, img_classes)
        ],
        "audio_points": [
            {"x": float(c[0]), "y": float(c[1]), "class": cls}
            for c, cls in zip(aud_coords, aud_classes)
        ],
        "image_class_names": img_names,
        "audio_class_names": aud_names,
    }




