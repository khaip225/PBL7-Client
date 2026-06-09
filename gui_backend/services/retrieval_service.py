"""Cross-modal retrieval: Audio↔Image search sử dụng PipelineEngine.

Dùng PipelineEngine để:
- Trích xuất embedding 256-d từ ảnh/audio
- Tìm kiếm nearest-neighbor trong database .npy (nếu có)
- Fallback: tìm trong thư mục local nếu chưa có database
"""

import os
import json
from pathlib import Path
from typing import Optional

import numpy as np


class RetrievalService:
    """Cross-modal retrieval sử dụng PipelineEngine."""

    def __init__(self, pipeline_engine, storage_manager):
        """
        Args:
            pipeline_engine: PipelineEngine instance (có sẵn audio/image database)
            storage_manager: StorageManager instance
        """
        self.engine = pipeline_engine
        self.storage_manager = storage_manager

        # Cached embeddings cho fallback search trong thư mục local
        self._image_embeddings: dict[str, np.ndarray] = {}
        self._audio_embeddings: dict[str, np.ndarray] = {}
        self._index_built = False

    # ══════════════════════════════════════════════════════════════════════════
    # Public API (giữ nguyên interface cũ để không phá vỡ API router)
    # ══════════════════════════════════════════════════════════════════════════

    def build_index(self, image_dir: str | None = None, audio_dir: str | None = None) -> int:
        """Build search index từ stored diagnosis files (fallback khi không có .npy).

        Returns tổng số files đã index.
        """
        img_dir = image_dir or self.storage_manager.image_dir
        aud_dir = audio_dir or self.storage_manager.audio_dir

        count = 0

        # Index images
        if os.path.exists(img_dir):
            for fname in os.listdir(img_dir):
                fpath = os.path.join(img_dir, fname)
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        tensor = self.engine.preprocess_image(fpath)
                        emb = self.engine.get_image_embedding(tensor)
                        self._image_embeddings[fname] = emb
                        count += 1
                    except Exception as e:
                        print(f"[Retrieval] Skip image {fname}: {e}")

        # Index audio
        if os.path.exists(aud_dir):
            for fname in os.listdir(aud_dir):
                fpath = os.path.join(aud_dir, fname)
                if fname.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
                    try:
                        tensor = self.engine.preprocess_audio(fpath)
                        emb = self.engine.get_audio_embedding(tensor)
                        self._audio_embeddings[fname] = emb
                        count += 1
                    except Exception as e:
                        print(f"[Retrieval] Skip audio {fname}: {e}")

        self._index_built = True
        print(f"[Retrieval] Index built: {len(self._image_embeddings)} images + {len(self._audio_embeddings)} audio")
        return count

    def audio_to_image(self, audio_path: str, top_k: int = 5) -> dict:
        """Retrieve top-k X-ray images matching an audio query.

        Ưu tiên dùng database .npy từ PipelineEngine, fallback về index local.
        """
        # Trích xuất audio embedding
        try:
            tensor = self.engine.preprocess_audio(audio_path)
            query_emb = self.engine.get_audio_embedding(tensor)
        except Exception as e:
            return {"error": f"Could not compute audio embedding: {e}"}

        # Ưu tiên database .npy
        if self.engine.image_database is not None:
            results = self._search_in_db(query_emb, self.engine.image_database, top_k)
        else:
            # Fallback: build index từ thư mục local
            if not self._image_embeddings:
                self.build_index()
            results = self._search_in_cache(query_emb, self._image_embeddings, top_k)

        return {
            "query": os.path.basename(audio_path),
            "modality": "audio",
            "target_modality": "image",
            "results": results,
        }

    def image_to_audio(self, image_path: str, top_k: int = 5) -> dict:
        """Retrieve top-k audio files matching an image query.

        Ưu tiên dùng database .npy từ PipelineEngine, fallback về index local.
        """
        # Trích xuất image embedding
        try:
            tensor = self.engine.preprocess_image(image_path)
            query_emb = self.engine.get_image_embedding(tensor)
        except Exception as e:
            return {"error": f"Could not compute image embedding: {e}"}

        # Ưu tiên database .npy
        if self.engine.audio_database is not None:
            results = self._search_in_db(query_emb, self.engine.audio_database, top_k)
        else:
            # Fallback: build index từ thư mục local
            if not self._audio_embeddings:
                self.build_index()
            results = self._search_in_cache(query_emb, self._audio_embeddings, top_k)

        return {
            "query": os.path.basename(image_path),
            "modality": "image",
            "target_modality": "audio",
            "results": results,
        }

    def get_prototype_similarities(self, modality: str, file_path: str) -> dict:
        """Get cosine similarity scores đến tất cả prototypes (cho radar/spider chart).

        Args:
            modality: "image" hoặc "audio"
            file_path: Đường dẫn file
        """
        try:
            if modality == "image":
                tensor = self.engine.preprocess_image(file_path)
                emb = self.engine.get_image_embedding(tensor)
                # Image prototypes: [normal_img, pneumonia, emphysema, fibrosis]
                proto_names = ["Normal", "Pneumonia", "COPD_Emphysema", "Fibrosis"]
                prototypes = self.engine.prototypes.get_img_protos().cpu().numpy()  # (4, 256)
            elif modality == "audio":
                tensor = self.engine.preprocess_audio(file_path)
                emb = self.engine.get_audio_embedding(tensor)
                # Audio prototypes: [normal_aud, crackle, wheeze]
                proto_names = ["Normal", "Crackle", "Wheeze"]
                prototypes = self.engine.prototypes.get_aud_protos().cpu().numpy()  # (3, 256)
            else:
                return {"error": f"Unknown modality: {modality}"}

            # Cosine similarity
            emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
            proto_norm = prototypes / (np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-8)
            sims = np.dot(proto_norm, emb_norm)

            return {name: round(float(s), 4) for name, s in zip(proto_names, sims)}

        except Exception as e:
            return {"error": str(e)}

    # ══════════════════════════════════════════════════════════════════════════
    # Internal helpers
    # ══════════════════════════════════════════════════════════════════════════

    def _search_in_db(self, query: np.ndarray, database: dict, top_k: int) -> list[dict]:
        """Cosine similarity search trong database .npy."""
        db_emb = database["embeddings"]       # (N, 256)
        db_files = database.get("files", [])
        db_labels = database.get("labels", [])
        db_cases = database.get("case_ids", [])

        query_norm = query / (np.linalg.norm(query) + 1e-8)
        db_norm = db_emb / (np.linalg.norm(db_emb, axis=1, keepdims=True) + 1e-8)
        similarities = np.dot(db_norm, query_norm)

        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        for idx in top_indices:
            sim = float(similarities[idx])
            if sim < 0.01:
                continue
            fp = db_files[idx] if idx < len(db_files) else ""
            fn = os.path.basename(fp)
            cid = db_cases[idx] if idx < len(db_cases) else f"#{idx + 1}"
            lb = db_labels[idx] if idx < len(db_labels) else {}
            dis = lb.get("disease", "") if isinstance(lb, dict) else ""
            aco = lb.get("acoustic", "") if isinstance(lb, dict) else ""
            results.append({
                "file": fn,
                "file_path": fp,
                "similarity": round(sim, 4),
                "case_id": cid,
                "disease_label": dis,
                "acoustic_label": aco,
            })
        return results

    def _search_in_cache(self, query: np.ndarray, cache: dict[str, np.ndarray], top_k: int) -> list[dict]:
        """Cosine similarity search trong cached embeddings (fallback)."""
        if not cache:
            return []

        items = list(cache.items())
        emb_matrix = np.stack([e for _, e in items])  # (N, 256)

        query_norm = query / (np.linalg.norm(query) + 1e-8)
        db_norm = emb_matrix / (np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-8)
        similarities = np.dot(db_norm, query_norm)

        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        for idx in top_indices:
            sim = float(similarities[idx])
            if sim < 0.01:
                continue
            fname = items[idx][0]
            results.append({
                "file": fname,
                "file_path": fname,
                "similarity": round(sim, 4),
                "case_id": "",
                "disease_label": "",
                "acoustic_label": "",
            })
        return results
