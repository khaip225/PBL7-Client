"""t-SNE visualization API for embedding space analysis."""

import os
import json
import numpy as np
from fastapi import APIRouter, HTTPException, Request, Query

router = APIRouter()


@router.get("/api/tsne")
async def get_tsne(
    request: Request,
    max_samples: int = Query(100, ge=10, le=500),
):
    """Compute t-SNE visualization of stored embeddings (image + audio joint space)."""
    retrieval_svc = getattr(request.app.state, "retrieval_service", None)
    if retrieval_svc is None:
        raise HTTPException(503, "Retrieval service not initialized")

    image_predictor = request.app.state.diagnosis_service.image_predictor
    audio_predictor = request.app.state.diagnosis_service.audio_predictor

    # Build index if not done yet
    img_embeddings = retrieval_svc._image_embeddings
    aud_embeddings = retrieval_svc._audio_embeddings

    if not img_embeddings and not aud_embeddings:
        retrieval_svc.build_index()

    # Collect embeddings from cached index
    img_embs = []
    aud_embs = []

    for fname, emb in list(retrieval_svc._image_embeddings.items())[:max_samples]:
        img_embs.append(emb.numpy())

    for fname, emb in list(retrieval_svc._audio_embeddings.items())[:max_samples]:
        aud_embs.append(emb.numpy())

    if not img_embs and not aud_embs:
        return {"image_points": [], "audio_points": [], "image_class_names": [], "audio_class_names": []}

    # Pseudo-labels based on file name prefix
    to_arr = lambda lst: np.stack(lst) if lst else np.zeros((0, 256))

    img_arr = to_arr(img_embs)
    aud_arr = to_arr(aud_embs)

    # Compute joint t-SNE
    try:
        from shared.tsne_utils import compute_tsne_visualization
        # Generate dummy labels from file names (label is prefix before underscore)
        def extract_labels(embeddings_dict, class_names):
            labels = []
            for fname in list(embeddings_dict.keys())[:len(embeddings_dict)]:
                label_prefix = fname.split("_")[0] if "_" in fname else "Unknown"
                if label_prefix in class_names:
                    labels.append(class_names.index(label_prefix))
                else:
                    labels.append(0)
            if not labels:
                return np.zeros((0,))
            return np.array(labels)

        img_labels = extract_labels(
            dict(list(retrieval_svc._image_embeddings.items())[:max_samples]),
            ["Pneumonia", "COPD_Emphysema", "Fibrosis", "Normal"],
        )
        aud_labels = extract_labels(
            dict(list(retrieval_svc._audio_embeddings.items())[:max_samples]),
            ["Crackle", "Wheeze", "Normal"],
        )
        # Convert to one-hot for compatibility
        img_labels_oh = np.eye(4)[img_labels] if len(img_labels) > 0 else np.zeros((0, 4))
        aud_labels_oh = np.eye(3)[aud_labels] if len(aud_labels) > 0 else np.zeros((0, 3))

        result = compute_tsne_visualization(
            img_arr, aud_arr,
            img_labels_oh, aud_labels_oh,
            image_class_names=["Pneumonia", "COPD_Emphysema", "Fibrosis", "Normal"],
            audio_class_names=["Crackle", "Wheeze", "Normal"],
        )
        return result
    except ImportError as e:
        raise HTTPException(500, f"scikit-learn required for t-SNE: {e}")
    except Exception as e:
        raise HTTPException(500, f"t-SNE computation failed: {e}")
