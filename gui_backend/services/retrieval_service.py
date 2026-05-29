"""Cross-modal retrieval: Audio↔Image search in shared embedding space.

Uses the prototype model to embed images and audio into the same 256-d space,
then retrieves the top-k nearest neighbors from the opposite modality.
"""

import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Optional


class RetrievalService:
    """Cross-modal retrieval using prototype-aligned embeddings."""

    def __init__(
        self,
        audio_predictor,
        image_predictor,
        storage_manager,
        prototype_model: Optional[torch.nn.Module] = None,
    ):
        self.audio_predictor = audio_predictor
        self.image_predictor = image_predictor
        self.storage_manager = storage_manager
        self.prototype_model = prototype_model
        self.device = next(prototype_model.parameters()).device if prototype_model else torch.device("cpu")

        # Cached embeddings for FL training history
        self._image_embeddings: dict[str, torch.Tensor] = {}
        self._audio_embeddings: dict[str, torch.Tensor] = {}
        self._index_built = False

    def build_index(self, image_dir: str | None = None, audio_dir: str | None = None) -> int:
        """Build search index from stored diagnosis files.

        Returns total number of files indexed.
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
                        if self.prototype_model is not None:
                            emb = self._get_image_embedding_prototype(fpath)
                        else:
                            emb = self._get_image_embedding_simple(fpath)
                        if emb is not None:
                            self._image_embeddings[fname] = emb
                            count += 1
                    except Exception as e:
                        print(f"[Retrieval] Skip image {fname}: {e}")

        # Index audio
        if os.path.exists(aud_dir):
            for fname in os.listdir(aud_dir):
                fpath = os.path.join(aud_dir, fname)
                if fname.lower().endswith('.wav'):
                    try:
                        if self.prototype_model is not None:
                            emb = self._get_audio_embedding_prototype(fpath)
                        else:
                            emb = self._get_audio_embedding_simple(fpath)
                        if emb is not None:
                            self._audio_embeddings[fname] = emb
                            count += 1
                    except Exception as e:
                        print(f"[Retrieval] Skip audio {fname}: {e}")

        self._index_built = True
        print(f"[Retrieval] Index built: {len(self._image_embeddings)} images + {len(self._audio_embeddings)} audio")
        return count

    def audio_to_image(self, audio_path: str, top_k: int = 5) -> dict:
        """Retrieve top-k X-ray images matching an audio query."""
        if not self._image_embeddings:
            self.build_index()

        if self.prototype_model is not None:
            query_emb = self._get_audio_embedding_prototype(audio_path)
        else:
            query_emb = self._get_audio_embedding_simple(audio_path)

        if query_emb is None:
            return {"error": "Could not compute audio embedding"}

        results = self._search(query_emb, list(self._image_embeddings.items()))
        return {
            "query": os.path.basename(audio_path),
            "modality": "audio",
            "target_modality": "image",
            "results": results[:top_k],
        }

    def image_to_audio(self, image_path: str, top_k: int = 5) -> dict:
        """Retrieve top-k audio files matching an image query."""
        if not self._audio_embeddings:
            self.build_index()

        if self.prototype_model is not None:
            query_emb = self._get_image_embedding_prototype(image_path)
        else:
            query_emb = self._get_image_embedding_simple(image_path)

        if query_emb is None:
            return {"error": "Could not compute image embedding"}

        results = self._search(query_emb, list(self._audio_embeddings.items()))
        return {
            "query": os.path.basename(image_path),
            "modality": "image",
            "target_modality": "audio",
            "results": results[:top_k],
        }

    def get_prototype_similarities(self, modality: str, file_path: str) -> dict:
        """Get similarity scores to all prototypes for visualization."""
        if self.prototype_model is None:
            return {"error": "Prototype model not loaded"}

        self.prototype_model.eval()
        with torch.no_grad():
            if modality == "image":
                emb = self._get_image_embedding_prototype(file_path)
                proto_names = ["Pneumonia", "COPD_Emphysema", "Fibrosis"]
                prototypes = self.prototype_model.disease_protos()  # (3, 256)
            else:
                emb = self._get_audio_embedding_prototype(file_path)
                proto_names = ["Crackle", "Wheeze"]
                prototypes = self.prototype_model.acoustic_protos()  # (2, 256)

            if emb is None:
                return {"error": "Could not compute embedding"}

            sim = torch.matmul(emb.unsqueeze(0), prototypes.T).squeeze(0)
            return {name: round(float(s), 4) for name, s in zip(proto_names, sim)}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _search(self, query: torch.Tensor, candidates: list[tuple[str, torch.Tensor]]) -> list[dict]:
        """Cosine similarity search."""
        results = []
        for fname, emb in candidates:
            sim = F.cosine_similarity(query, emb.to(query.device), dim=0)
            results.append({"file": fname, "similarity": round(float(sim.item()), 4)})
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results

    def _get_image_embedding_prototype(self, path: str) -> Optional[torch.Tensor]:
        from torchvision import transforms
        from PIL import Image

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        img = Image.open(path).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.prototype_model.image_encoder.get_embedding(tensor)
        return emb.squeeze(0).cpu()

    def _get_audio_embedding_prototype(self, path: str) -> Optional[torch.Tensor]:
        """Reuse audio predictor's preprocessing + prototype model's encoder."""
        import torchaudio
        import torch.nn.functional as F
        import soundfile as sf
        from PIL import Image
        import numpy as np

        waveform_np, sr = sf.read(path)
        waveform = torch.from_numpy(waveform_np).float().unsqueeze(0)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            waveform = resampler(waveform)

        target_samples = 16000 * 15
        if waveform.shape[1] < target_samples:
            waveform = F.pad(waveform, (0, target_samples - waveform.shape[1]))
        else:
            waveform = waveform[:, :target_samples]

        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=1024, hop_length=512, n_mels=128
        )(waveform)
        mel_db = torchaudio.transforms.AmplitudeToDB()(mel)

        mean = mel_db.mean()
        std = mel_db.std()
        mel_db = (mel_db - mean) / (std + 1e-6)

        mel_np = mel_db.squeeze(0).numpy()
        mel_img = Image.fromarray(
            ((mel_np - mel_np.min()) / (mel_np.max() - mel_np.min() + 1e-8) * 255).astype(np.uint8)
        ).convert("RGB").resize((224, 224), Image.BICUBIC)

        from torchvision import transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        tensor = transform(mel_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.prototype_model.audio_encoder.get_embedding(tensor)
        return emb.squeeze(0).cpu()

    def _get_image_embedding_simple(self, path: str) -> Optional[torch.Tensor]:
        """Fallback: use Grad-CAM target layer features as embedding proxy."""
        import torch.nn.functional as F
        from PIL import Image
        from torchvision import transforms
        try:
            # Access the last conv features from image predictor
            self.image_predictor.model.eval()
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            img = Image.open(path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                features = self.image_predictor.model.encoder.features(tensor)
                features = F.relu(features, inplace=True)
                features = F.adaptive_avg_pool2d(features, (1, 1))
                features = torch.flatten(features, 1)
                emb = self.image_predictor.model.encoder.projection(features)
                return F.normalize(emb.squeeze(0), p=2, dim=-1).cpu()
        except Exception:
            return None

    def _get_audio_embedding_simple(self, path: str) -> Optional[torch.Tensor]:
        """Fallback: use AST forward features as embedding proxy."""
        import torch.nn.functional as F
        from PIL import Image
        import numpy as np
        import torchaudio
        import soundfile as sf

        try:
            self.audio_predictor.model.eval()
            file_path = path  # Use predictor's preprocess
            waveform_np, sr = sf.read(file_path)
            waveform = torch.from_numpy(waveform_np).float().unsqueeze(0)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                waveform = resampler(waveform)

            target_samples = 16000 * 15
            if waveform.shape[1] < target_samples:
                waveform = F.pad(waveform, (0, target_samples - waveform.shape[1]))
            else:
                waveform = waveform[:, :target_samples]

            mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000, n_fft=1024, hop_length=512, n_mels=128
            )(waveform)
            mel_db = torchaudio.transforms.AmplitudeToDB()(mel)
            mean = mel_db.mean(); std = mel_db.std()
            mel_db = (mel_db - mean) / (std + 1e-6)
            mel_np = mel_db.squeeze(0).numpy()
            mel_img = Image.fromarray(
                ((mel_np - mel_np.min()) / (mel_np.max() - mel_np.min() + 1e-8) * 255).astype(np.uint8)
            ).convert("RGB").resize((224, 224), Image.BICUBIC)

            from torchvision import transforms
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            tensor = transform(mel_img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                emb = self.audio_predictor.model.get_embedding(tensor)
            return emb.squeeze(0).cpu()
        except Exception:
            return None

