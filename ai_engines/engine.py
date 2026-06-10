"""PipelineEngine — Điều phối toàn bộ 5 hành động chẩn đoán.

Models khớp chính xác với kiến trúc từ notebook Stage 4 (pbl7-fl.ipynb).

Kịch bản A (Upload Ảnh X-Quang):
  HĐ1 - Phân loại trực tiếp (Classification) — encoder+head từ Stage 5
  HĐ2 - Suy luận Zero-shot chéo (Cross-modal) — prototype từ Stage 4
  HĐ3 - Truy xuất bằng chứng (Retrieval) — vector DB
  HĐ4 - Trí tuệ giải thích được (XAI: Grad-CAM)
  HĐ5 - Chốt kết luận (Late Fusion)

Kịch bản B (Upload File Âm thanh):
  HĐ1 - Phân loại trực tiếp (Classification)
  HĐ2 - Suy luận cấu trúc phổi Zero-shot
  HĐ3 - Truy xuất X-quang (Retrieval)
  HĐ4 - Trí tuệ giải thích được (XAI: Audio Attention)
  HĐ5 - Chốt kết luận (Late Fusion)

Encoder:   Stage 4 → cross-modal (HĐ2), retrieval (HĐ3), embedding cho DB
           Stage 5 → classification (HĐ1), Grad-CAM (HĐ4)
"""

from __future__ import annotations

import os
import sys
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# ── Path setup ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ai_engines.constants import (
    IMAGE_CLASS_NAMES, IMAGE_DISEASE_NAMES,
    AUDIO_CLASS_NAMES, AUDIO_ATTR_NAMES,
)
from ai_engines.models import (
    DenseNet121EncoderNB, ASTEncoderNB, MomentumPrototypeModuleNB,
)
from ai_engines.schemas import (
    RetrievalItem, LateFusionResult, ImagePipelineResult, AudioPipelineResult,
)
from ai_engines.xai import GradCAMWrapper, GradCAM


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline Engine
# ══════════════════════════════════════════════════════════════════════════════

class PipelineEngine:
    """Engine điều phối toàn bộ pipeline chẩn đoán 5 hành động."""

    def __init__(
        self,
        stage4_path: str,
        image_head_path: str,
        audio_head_path: str,
        audio_db_path: Optional[str] = None,
        image_db_path: Optional[str] = None,
        device: Optional[str] = None,
        threshold: float = 0.5,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold

        # ── Load models ──────────────────────────────────────────────
        self._load_stage4(stage4_path)
        self._load_classifier_models(image_head_path, audio_head_path)
        self._load_databases(audio_db_path, image_db_path)

        # ── Image transform ──────────────────────────────────────────
        # NOTE: Must match Stage 5 notebook preprocessing:
        # cv2.imread GRAYSCALE → cv2.resize(224,224) → ToTensor() → repeat(3,1,1) → /255.0
        # NO ImageNet normalization!
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # Converts to [0,1]
        ])

        # ── Audio preprocessing params ───────────────────────────────
        self.target_sr = 16000
        self.max_duration = 15

        print(f"[PipelineEngine] Đã khởi tạo trên {self.device}")
        print(f"[PipelineEngine] Prototypes: {len(self.prototypes._NAMES)}")
        print(f"[PipelineEngine] Image cls: {self.num_image_classes} classes")
        print(f"[PipelineEngine] Audio cls: {self.num_audio_classes} classes")

    # ═══════════════════════════════════════════════════════════════════
    # Loading
    # ═══════════════════════════════════════════════════════════════════

    def _load_stage4(self, stage4_path: str):
        """Load stage4_best_model.pth: Encoders + Prototypes.

        Encoder từ stage4 dùng cho: cross-modal (HĐ2), retrieval (HĐ3),
        và radar/spider chart visualization. Prototypes dùng cho cross-modal.
        """
        print(f"[PipelineEngine] Đang load stage4 từ: {stage4_path}")
        checkpoint = torch.load(stage4_path, map_location=self.device, weights_only=False)

        # ── Image Encoder ──────────────────────────────────────────
        self.image_encoder = DenseNet121EncoderNB(embedding_dim=256, dropout=0.2)
        missing, unexpected = self.image_encoder.load_state_dict(
            checkpoint["image_encoder"], strict=False)
        if missing:
            print(f"[PipelineEngine] Image encoder missing: {len(missing)} keys")
        if unexpected:
            print(f"[PipelineEngine] Image encoder unexpected: {len(unexpected)} keys")
        self.image_encoder.to(self.device)
        self.image_encoder.eval()

        # ── Audio Encoder ──────────────────────────────────────────
        self.audio_encoder = ASTEncoderNB(embedding_dim=256, dropout=0.2)
        missing, unexpected = self.audio_encoder.load_state_dict(
            checkpoint["audio_encoder"], strict=False)
        if missing:
            print(f"[PipelineEngine] Audio encoder missing: {len(missing)} keys")
        if unexpected:
            print(f"[PipelineEngine] Audio encoder unexpected: {len(unexpected)} keys")
        self.audio_encoder.to(self.device)
        self.audio_encoder.eval()

        # ── Prototype Module ───────────────────────────────────────
        self.prototypes = MomentumPrototypeModuleNB(dim=256, momentum=0.90)
        self.prototypes.load_state_dict(checkpoint["prototype_module"], strict=False)
        self.prototypes.to(self.device)
        self.prototypes.eval()
        self.prototypes.normalize_and_update()

        self.stage4_round = checkpoint.get("round", "unknown")
        print(f"[PipelineEngine] Stage4 round {self.stage4_round} loaded")

    def _load_classifier_models(self, image_head_path: str, audio_head_path: str):
        """Load Global_Image_best.pth và Global_Audio_best.pth — encoder + head.

        Stage 5 train với freeze_backbone=False nên encoder trong Global_*
        đã được fine-tune. HĐ1 phải dùng encoder+head từ chính Global_*
        để encoder và head khớp nhau.
        """
        # ── Image: Encoder + Head từ Global_Image ──────────────────
        print(f"[PipelineEngine] Loading image model: {image_head_path}")
        img_ckpt = torch.load(image_head_path, map_location=self.device, weights_only=True)

        self.image_encoder_cls = DenseNet121EncoderNB(embedding_dim=256, dropout=0.2)
        enc_state = {k.replace("encoder.", ""): v
                     for k, v in img_ckpt.items() if k.startswith("encoder.")}
        missing, _ = self.image_encoder_cls.load_state_dict(enc_state, strict=False)
        if missing:
            print(f"[PipelineEngine] Image cls encoder missing: {len(missing)} keys")
        self.image_encoder_cls.to(self.device)
        self.image_encoder_cls.eval()

        self.num_image_classes = img_ckpt["head.3.weight"].shape[0]
        # MUST match Stage 5 notebook: GELU + Dropout(0.2)
        self.image_classifier_head = nn.Sequential(
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, self.num_image_classes),
        )
        head_state = {k.replace("head.", ""): v
                      for k, v in img_ckpt.items() if k.startswith("head.")}
        self.image_classifier_head.load_state_dict(head_state, strict=True)
        self.image_classifier_head.to(self.device)
        self.image_classifier_head.eval()
        print(f"[PipelineEngine] Image cls: encoder + head ({self.num_image_classes} classes) OK")

        # ── Audio: Encoder + Head từ Global_Audio ──────────────────
        print(f"[PipelineEngine] Loading audio model: {audio_head_path}")
        aud_ckpt = torch.load(audio_head_path, map_location=self.device, weights_only=True)

        self.audio_encoder_cls = ASTEncoderNB(embedding_dim=256, dropout=0.2)
        aud_enc_state = {k.replace("encoder.", ""): v
                         for k, v in aud_ckpt.items() if k.startswith("encoder.")}
        missing, _ = self.audio_encoder_cls.load_state_dict(aud_enc_state, strict=False)
        if missing:
            print(f"[PipelineEngine] Audio cls encoder missing: {len(missing)} keys")
        self.audio_encoder_cls.to(self.device)
        self.audio_encoder_cls.eval()

        self.num_audio_classes = aud_ckpt["head.3.weight"].shape[0]
        # MUST match Stage 5 notebook: GELU + Dropout(0.2)
        self.audio_classifier_head = nn.Sequential(
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, self.num_audio_classes),
        )
        aud_head_state = {k.replace("head.", ""): v
                          for k, v in aud_ckpt.items() if k.startswith("head.")}
        self.audio_classifier_head.load_state_dict(aud_head_state, strict=True)
        self.audio_classifier_head.to(self.device)
        self.audio_classifier_head.eval()
        print(f"[PipelineEngine] Audio cls: encoder + head ({self.num_audio_classes} classes) OK")

    def _load_databases(self, audio_db_path: Optional[str], image_db_path: Optional[str]):
        self.audio_database = None
        self.image_database = None

        if audio_db_path and os.path.exists(audio_db_path):
            self.audio_database = np.load(audio_db_path, allow_pickle=True).item()
            print(f"[PipelineEngine] Audio DB: {len(self.audio_database['embeddings'])} files")

        if image_db_path and os.path.exists(image_db_path):
            self.image_database = np.load(image_db_path, allow_pickle=True).item()
            print(f"[PipelineEngine] Image DB: {len(self.image_database['embeddings'])} files")

    # ═══════════════════════════════════════════════════════════════════
    # Preprocessing
    # ═══════════════════════════════════════════════════════════════════

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess exactly like Stage 5 notebook: GRAYSCALE + resize + /255 + repeat(3)."""
        import cv2
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Không đọc được ảnh: {image_path}")
        img = cv2.resize(img, (224, 224))
        tensor = torch.from_numpy(img).float().unsqueeze(0) / 255.0  # (1, H, W)
        tensor = tensor.repeat(3, 1, 1).unsqueeze(0)  # (1, 3, H, W)
        return tensor.to(self.device)

    def preprocess_audio(self, audio_path: str) -> torch.Tensor:
        """Audio → mel-spectrogram tensor (1, 128, 384) khớp notebook."""
        import torchaudio
        import soundfile as sf

        try:
            waveform, sr = torchaudio.load(audio_path)
        except Exception:
            waveform_np, sr = sf.read(audio_path)
            waveform = torch.from_numpy(waveform_np).float()
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            elif waveform.ndim == 2:
                waveform = waveform.transpose(0, 1)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
            waveform = resampler(waveform)

        target_samples = self.target_sr * self.max_duration
        current = waveform.shape[1]
        if current < target_samples:
            waveform = F.pad(waveform, (0, target_samples - current))
        else:
            waveform = waveform[:, :target_samples]

        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.target_sr, n_fft=1024, hop_length=512, n_mels=128,
        )
        mel_spec = mel_transform(waveform)  # (1, 128, T)
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-6)

        if mel_spec.shape[-1] < 384:
            mel_spec = F.pad(mel_spec, (0, 384 - mel_spec.shape[-1]))
        else:
            mel_spec = mel_spec[:, :, :384]

        return mel_spec.float().to(self.device)  # (1, 128, 384)

    # ═══════════════════════════════════════════════════════════════════
    # Hành động 1: Phân loại trực tiếp (Stage 5 encoder + head)
    # ═══════════════════════════════════════════════════════════════════

    def classify_image(self, image_tensor: torch.Tensor) -> dict:
        """Dùng encoder+head từ Global_Image_best.pth (Stage 5)."""
        with torch.no_grad():
            embedding = self.image_encoder_cls(image_tensor)
            logits = self.image_classifier_head(embedding)
            probs = torch.sigmoid(logits).squeeze(0)
        return {IMAGE_CLASS_NAMES[i]: round(probs[i].item(), 4)
                for i in range(self.num_image_classes)}

    def classify_audio(self, audio_tensor: torch.Tensor) -> dict:
        """Dùng encoder+head từ Global_Audio_best.pth (Stage 5)."""
        with torch.no_grad():
            embedding = self.audio_encoder_cls(audio_tensor)
            logits = self.audio_classifier_head(embedding)
            probs = torch.sigmoid(logits).squeeze(0)
        return {AUDIO_CLASS_NAMES[i]: round(probs[i].item(), 4)
                for i in range(self.num_audio_classes)}

    # ═══════════════════════════════════════════════════════════════════
    # Embedding extraction (Stage 4 — cho cross-modal & retrieval)
    # ═══════════════════════════════════════════════════════════════════

    def get_image_embedding(self, image_tensor: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            emb = self.image_encoder(image_tensor)
        return emb.squeeze(0).cpu().numpy()

    def get_audio_embedding(self, audio_tensor: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            emb = self.audio_encoder(audio_tensor)
        return emb.squeeze(0).cpu().numpy()

    # ═══════════════════════════════════════════════════════════════════
    # Hành động 2: Cross-modal (Prototype-based)
    # ═══════════════════════════════════════════════════════════════════

    def cross_modal_image_to_acoustic(self, image_embedding: np.ndarray) -> dict:
        """Vector ảnh → Cosine similarity với Audio Prototypes → Acoustic prediction."""
        emb_t = F.normalize(
            torch.from_numpy(image_embedding).float().unsqueeze(0).to(self.device),
            p=2, dim=-1)
        aud_protos = self.prototypes.get_aud_protos()  # (3, 256)
        sims = F.cosine_similarity(emb_t, aud_protos.unsqueeze(0), dim=-1).squeeze(0).cpu().numpy()
        sims_clamped = np.clip(sims, -1, 1)
        return {
            "Crackle": round(float((sims_clamped[1] + 1) / 2), 4),
            "Wheeze": round(float((sims_clamped[2] + 1) / 2), 4),
        }

    def cross_modal_audio_to_disease(self, audio_embedding: np.ndarray) -> dict:
        """Vector âm thanh → Cosine similarity với Image Prototypes → Disease prediction."""
        emb_t = F.normalize(
            torch.from_numpy(audio_embedding).float().unsqueeze(0).to(self.device),
            p=2, dim=-1)
        img_protos = self.prototypes.get_img_protos()  # (4, 256)
        sims = F.cosine_similarity(emb_t, img_protos.unsqueeze(0), dim=-1).squeeze(0).cpu().numpy()
        sims_clamped = np.clip(sims, -1, 1)
        return {
            "Normal": round(float((sims_clamped[0] + 1) / 2), 4),
            "Pneumonia": round(float((sims_clamped[1] + 1) / 2), 4),
            "COPD_Emphysema": round(float((sims_clamped[2] + 1) / 2), 4),
            "Fibrosis": round(float((sims_clamped[3] + 1) / 2), 4),
        }

    def _build_cross_modal_message(self, scores: dict, direction: str = "image_to_acoustic") -> str:
        if direction == "image_to_acoustic":
            crackle = scores.get("Crackle", 0)
            wheeze = scores.get("Wheeze", 0)
            if crackle > wheeze and crackle > 0.3:
                return (f"🔊 Hệ thống dự đoán nếu nghe bằng ống nghe, bệnh nhân này "
                        f"có khả năng phát ra tiếng Ran nổ (Crackle) rất cao ({(crackle*100):.0f}%).")
            elif wheeze > crackle and wheeze > 0.3:
                return (f"🔊 Hệ thống dự đoán nếu nghe bằng ống nghe, bệnh nhân này "
                        f"có khả năng phát ra tiếng Ran rít (Wheeze) rất cao ({(wheeze*100):.0f}%).")
            else:
                return "🔊 Dựa trên ảnh X-quang, chưa có dấu hiệu rõ ràng về bất thường âm thanh phổi."
        else:
            normal = scores.get("Normal", 0)
            pne = scores.get("Pneumonia", 0)
            copd = scores.get("COPD_Emphysema", 0)
            fibr = scores.get("Fibrosis", 0)
            best = max(pne, copd, fibr)
            if normal > best:
                return "🫁 Dựa trên âm thanh phổi, hệ thống dự đoán cấu trúc phổi có vẻ bình thường."
            elif pne >= best:
                return (f"🫁 Dựa trên âm thanh này, hệ thống dự đoán nếu chụp X-quang, "
                        f"hình ảnh cấu trúc phổi sẽ có dấu hiệu của Viêm phổi (Pneumonia) ({(pne*100):.0f}%).")
            elif copd >= best:
                return (f"🫁 Dựa trên tiếng rít (Wheeze) này, hệ thống dự đoán nếu chụp X-quang, "
                        f"hình ảnh cấu trúc phổi sẽ có dấu hiệu của Khí phế thũng (Emphysema) ({(copd*100):.0f}%).")
            else:
                return (f"🫁 Dựa trên âm thanh này, hệ thống dự đoán nếu chụp X-quang, "
                        f"hình ảnh cấu trúc phổi sẽ có dấu hiệu của Xơ phổi (Fibrosis) ({(fibr*100):.0f}%).")

    # ═══════════════════════════════════════════════════════════════════
    # Hành động 3: Retrieval (Vector DB)
    # ═══════════════════════════════════════════════════════════════════

    def retrieve_audio(self, image_embedding: np.ndarray, top_k: int = 3) -> list[RetrievalItem]:
        return self._retrieve(image_embedding, self.audio_database, top_k)

    def retrieve_images(self, audio_embedding: np.ndarray, top_k: int = 3) -> list[RetrievalItem]:
        return self._retrieve(audio_embedding, self.image_database, top_k)

    def _retrieve(self, query: np.ndarray, database: Optional[dict], top_k: int = 3) -> list[RetrievalItem]:
        if database is None or "embeddings" not in database:
            return []

        db_emb = database["embeddings"]
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
            cid = db_cases[idx] if idx < len(db_cases) else f"#{idx+1}"
            lb = db_labels[idx] if idx < len(db_labels) else {}
            dis = lb.get("disease", "") if isinstance(lb, dict) else ""
            aco = lb.get("acoustic", "") if isinstance(lb, dict) else ""
            results.append(RetrievalItem(fp, fn, round(sim, 4), cid, dis, aco))
        return results

    # ═══════════════════════════════════════════════════════════════════
    # Hành động 4: XAI
    # ═══════════════════════════════════════════════════════════════════

    def generate_gradcam(self, image_path: str, image_tensor: torch.Tensor,
                         save_dir: Optional[str] = None) -> str:
        """Grad-CAM dùng encoder+head từ Global_Image (Stage 5)."""
        wrapper = GradCAMWrapper(self.image_encoder_cls, self.image_classifier_head)
        wrapper.to(self.device).eval()

        target_layer = self.image_encoder_cls.features.denseblock4.denselayer16.conv2
        gradcam = GradCAM(wrapper, target_layer)

        with torch.no_grad():
            logits = wrapper(image_tensor)
            probs = torch.sigmoid(logits).squeeze(0)
        best_idx = int(probs.argmax().item())
        best_class = IMAGE_CLASS_NAMES[best_idx]

        heatmap = gradcam.generate(image_tensor, class_idx=best_idx)
        original = Image.open(image_path).convert("RGB")
        overlay_img = gradcam.overlay(original, heatmap)

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            ts = int(time.time() * 1000)
            path = os.path.join(save_dir, f"heatmap_{best_class}_{ts}.png")
            overlay_img.save(path)
            gradcam.remove()
            return path

        gradcam.remove()
        return ""

    def generate_audio_attention(self, audio_path: str, audio_tensor: torch.Tensor,
                                 save_dir: Optional[str] = None) -> str:
        """Audio attention map overlay trên mel-spectrogram."""
        import cv2

        with torch.no_grad():
            outputs = self.audio_encoder.backbone(
                audio_tensor, output_hidden_states=True, output_attentions=True
            )

        attentions = outputs.attentions
        if attentions:
            last_attn = attentions[-1]
            mean_attn = last_attn.mean(dim=1).squeeze(0)
            cls_attn = mean_attn[0, 2:]  # CLS → patches, skip CLS+dist
            n_patches = cls_attn.shape[0]
            grid_h = 12
            grid_w = n_patches // grid_h if n_patches % grid_h == 0 else 37
            if grid_h * grid_w != n_patches:
                grid_w = n_patches // grid_h
                if grid_h * grid_w < n_patches:
                    grid_w += 1
            cls_attn = cls_attn[:grid_h * grid_w]
            attn_map = cls_attn.reshape(grid_h, grid_w).cpu().numpy()
        else:
            attn_map = np.ones((12, 37))

        vmin, vmax = attn_map.min(), attn_map.max()
        if vmax > vmin:
            attn_map = (attn_map - vmin) / (vmax - vmin)

        mel_np = audio_tensor.squeeze(0).cpu().numpy()
        attn_resized = cv2.resize(attn_map, (mel_np.shape[1], mel_np.shape[0]))
        attn_color = cv2.applyColorMap(np.uint8(255 * attn_resized), cv2.COLORMAP_JET)
        attn_color = cv2.cvtColor(attn_color, cv2.COLOR_BGR2RGB)

        mel_vis = ((mel_np - mel_np.min()) / (mel_np.max() - mel_np.min() + 1e-8) * 255).astype(np.uint8)
        mel_rgb = cv2.cvtColor(mel_vis, cv2.COLOR_GRAY2RGB)

        overlay = (mel_rgb * 0.5 + attn_color * 0.5).astype(np.uint8)
        overlay_img = Image.fromarray(overlay)

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            ts = int(time.time() * 1000)
            path = os.path.join(save_dir, f"audio_attention_{ts}.png")
            overlay_img.save(path)
            return path
        return ""

    # ═══════════════════════════════════════════════════════════════════
    # Hành động 5: Late Fusion
    # ═══════════════════════════════════════════════════════════════════

    def late_fusion(self, disease_probs: dict, acoustic_probs: dict) -> LateFusionResult:
        """Kết hợp classifier + cross-modal để ra chẩn đoán cuối cùng.

        - Image pipeline: disease_probs=classifier, acoustic_probs=cross-modal (Crackle/Wheeze)
        - Audio pipeline: disease_probs=cross-modal (disease), acoustic_probs=classifier

        Luôn trả về disease diagnosis vì đó là thứ bác sĩ cần.
        """
        pne_cls = disease_probs.get("Pneumonia", 0)
        copd_cls = disease_probs.get("COPD_Emphysema", 0)
        fibr_cls = disease_probs.get("Fibrosis", 0)

        # Nếu acoustic_probs chứa disease keys → audio→disease cross-modal
        if any(k in acoustic_probs for k in ["Normal", "Pneumonia", "COPD_Emphysema", "Fibrosis"]):
            cross_pne = acoustic_probs.get("Pneumonia", 0)
            cross_copd = acoustic_probs.get("COPD_Emphysema", 0)
            cross_fibr = acoustic_probs.get("Fibrosis", 0)
            cross_normal = acoustic_probs.get("Normal", 0)

            w_cls, w_cm = 0.7, 0.3
            p_pne = w_cls * pne_cls + w_cm * cross_pne
            p_copd = w_cls * copd_cls + w_cm * cross_copd
            p_fib = w_cls * fibr_cls + w_cm * cross_fibr
            p_normal = w_cls * disease_probs.get("Normal", 0) + w_cm * cross_normal
            cm_top = max(cross_pne, cross_copd, cross_fibr)
        else:
            # image→acoustic: Crackle/Wheeze → ontology boost
            crackle = acoustic_probs.get("Crackle", 0)
            wheeze = acoustic_probs.get("Wheeze", 0)

            pne_boost = crackle
            copd_boost = wheeze
            fibr_boost = crackle * 0.5

            w_cls, w_cm = 0.7, 0.3
            p_pne = w_cls * pne_cls + w_cm * pne_boost
            p_copd = w_cls * copd_cls + w_cm * copd_boost
            p_fib = w_cls * fibr_cls + w_cm * fibr_boost
            p_normal = max(0.0, min(1.0, 1.0 - max(p_pne, p_copd, p_fib)))
            cm_top = max(pne_boost, copd_boost, fibr_boost)

        fusion_scores = {
            "Pneumonia": round(p_pne, 4),
            "COPD_Emphysema": round(p_copd, 4),
            "Fibrosis": round(p_fib, 4),
            "Normal": round(p_normal, 4),
        }

        disease_scores = {
            "Viêm phổi (Pneumonia)": p_pne,
            "Khí phế thũng (COPD/Emphysema)": p_copd,
            "Xơ phổi (Fibrosis)": p_fib,
        }
        best_disease, best_score = max(disease_scores.items(), key=lambda x: x[1])

        cls_top = max(pne_cls, copd_cls, fibr_cls)

        if cls_top >= self.threshold and cm_top >= 0.3:
            agreement = "Có sự đồng thuận từ đặc trưng âm thanh"
            agreement_level = 1.0
        elif cls_top >= self.threshold:
            agreement = "Dựa chủ yếu vào đặc trưng hình ảnh"
            agreement_level = 0.7
        elif cm_top >= 0.3:
            agreement = "Dựa chủ yếu vào suy luận chéo từ âm thanh"
            agreement_level = 0.5
        else:
            agreement = "Độ đồng thuận thấp giữa các modality"
            agreement_level = 0.3

        confidence = best_score * (0.7 + 0.3 * agreement_level)
        if confidence >= 0.85:
            level = "Rất cao"
        elif confidence >= 0.7:
            level = "Cao"
        elif confidence >= 0.5:
            level = "Trung bình"
        else:
            level = "Thấp"

        is_normal = p_normal > max(p_pne, p_copd, p_fib)

        if is_normal:
            primary = "Bình thường (Normal)"
            confidence = p_normal
        else:
            primary = best_disease

        return LateFusionResult(
            primary_diagnosis=primary,
            confidence=round(confidence, 4),
            confidence_level=level,
            agreement=agreement,
            fusion_scores=fusion_scores,
            is_normal=is_normal,
        )

    # ═══════════════════════════════════════════════════════════════════
    # Pipeline Runners
    # ═══════════════════════════════════════════════════════════════════

    def run_image_pipeline(self, image_path: str,
                           save_heatmap_dir: Optional[str] = None) -> ImagePipelineResult:
        """Kịch bản A: Upload Ảnh X-Quang → 5 hành động."""
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        image_tensor = self.preprocess_image(image_path)

        # HĐ1: Classification (Stage 5)
        disease_probs = self.classify_image(image_tensor)
        print(f"[Pipeline] HĐ1 - Image Classification: {disease_probs}")

        # HĐ2: Cross-modal (Stage 4 prototype)
        image_emb = self.get_image_embedding(image_tensor)
        cross_modal_acoustic = self.cross_modal_image_to_acoustic(image_emb)
        cross_modal_msg = self._build_cross_modal_message(cross_modal_acoustic, "image_to_acoustic")
        print(f"[Pipeline] HĐ2 - Cross-modal: {cross_modal_acoustic}")

        # HĐ3: Retrieval
        retrieved_audio = self.retrieve_audio(image_emb, top_k=3)
        print(f"[Pipeline] HĐ3 - Retrieval: {len(retrieved_audio)} audio files")

        # HĐ4: XAI
        heatmap_path = ""
        gradcam_enabled = True
        try:
            heatmap_path = self.generate_gradcam(image_path, image_tensor, save_heatmap_dir)
            print(f"[Pipeline] HĐ4 - GradCAM: {heatmap_path}")
        except Exception as e:
            print(f"[Pipeline] HĐ4 - GradCAM error: {e}")
            gradcam_enabled = False

        # HĐ5: Late Fusion
        late_fusion = self.late_fusion(disease_probs, cross_modal_acoustic)
        print(f"[Pipeline] HĐ5 - Late Fusion: {late_fusion.primary_diagnosis} ({late_fusion.confidence_level})")

        return ImagePipelineResult(
            disease_probs=disease_probs,
            cross_modal_acoustic=cross_modal_acoustic,
            cross_modal_message=cross_modal_msg,
            retrieved_audio=retrieved_audio,
            heatmap_path=heatmap_path,
            gradcam_enabled=gradcam_enabled,
            late_fusion=late_fusion,
            embedding=image_emb,
            timestamp=ts,
            mode="image",
        )

    def run_audio_pipeline(self, audio_path: str,
                           save_attention_dir: Optional[str] = None) -> AudioPipelineResult:
        """Kịch bản B: Upload File Âm thanh → 5 hành động."""
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        audio_tensor = self.preprocess_audio(audio_path)

        # HĐ1: Classification (Stage 5)
        acoustic_probs = self.classify_audio(audio_tensor)
        print(f"[Pipeline] HĐ1 - Audio Classification: {acoustic_probs}")

        # HĐ2: Cross-modal (Stage 4 prototype)
        audio_emb = self.get_audio_embedding(audio_tensor)
        cross_modal_disease = self.cross_modal_audio_to_disease(audio_emb)
        cross_modal_msg = self._build_cross_modal_message(cross_modal_disease, "audio_to_disease")
        print(f"[Pipeline] HĐ2 - Cross-modal: {cross_modal_disease}")

        # HĐ3: Retrieval
        retrieved_images = self.retrieve_images(audio_emb, top_k=3)
        print(f"[Pipeline] HĐ3 - Retrieval: {len(retrieved_images)} images")

        # HĐ4: XAI
        attention_path = ""
        attention_enabled = True
        try:
            attention_path = self.generate_audio_attention(audio_path, audio_tensor, save_attention_dir)
            print(f"[Pipeline] HĐ4 - Audio Attention: {attention_path}")
        except Exception as e:
            print(f"[Pipeline] HĐ4 - Audio Attention error: {e}")
            attention_enabled = False

        # HĐ5: Late Fusion (đảo vai trò: disease = cross-modal, acoustic = classifier)
        late_fusion = self.late_fusion(cross_modal_disease, acoustic_probs)
        print(f"[Pipeline] HĐ5 - Late Fusion: {late_fusion.primary_diagnosis} ({late_fusion.confidence_level})")

        return AudioPipelineResult(
            acoustic_probs=acoustic_probs,
            cross_modal_disease=cross_modal_disease,
            cross_modal_message=cross_modal_msg,
            retrieved_images=retrieved_images,
            attention_map_path=attention_path,
            attention_enabled=attention_enabled,
            late_fusion=late_fusion,
            embedding=audio_emb,
            timestamp=ts,
            mode="audio",
        )
