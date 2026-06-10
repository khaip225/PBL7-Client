"""AudioPredictor — dùng AST mô hình thực sự với mel-spectrogram [1, T, F]."""

import torch
import torchaudio
import torch.nn.functional as F
import numpy as np
import soundfile as sf

from config import config


ACOUSTIC_CLASS_NAMES = ["normal", "crackle", "wheeze"]


class AudioPredictor:
    def __init__(self, model_path, device=None, threshold=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.threshold = threshold if threshold is not None else config.PREDICTION_THRESHOLD

        from ai_engines.audio_engine.ast_model import ASTMultiLabel
        self.model = ASTMultiLabel(num_classes=3).to(self.device)
        self._load_model()

        self.target_sr = 16000
        self.max_duration = 15
        self.n_mels = 128
        self.max_frames = 384

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.target_sr, n_fft=1024, hop_length=512, n_mels=self.n_mels,
        )

    def _load_model(self):
        try:
            state = torch.load(self.model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state, strict=False)
            self.model.eval()
        except FileNotFoundError:
            print(f"[AudioPredictor] Model file not found: {self.model_path}, using random weights")
            self.model.eval()
        except Exception as e:
            print(f"[AudioPredictor] Failed to load model: {e}, using random weights")
            self.model.eval()

    def preprocess(self, audio_path):
        """Match Stage 5 / Stage 4 notebook: waveform → mel-spec z-score → tensor [1, T, F]."""
        # Load audio
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

        # Mel-spectrogram
        mel_spec = self.mel_transform(waveform)  # (1, 128, T)
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-6)

        if mel_spec.shape[-1] < self.max_frames:
            mel_spec = F.pad(mel_spec, (0, self.max_frames - mel_spec.shape[-1]))
        else:
            mel_spec = mel_spec[:, :, :self.max_frames]

        return mel_spec.float().to(self.device)  # (1, 128, 384)

    def predict(self, audio_path):
        """Tra ve multi-label acoustic probabilities."""
        self.model.eval()
        with torch.no_grad():
            tensor = self.preprocess(audio_path)
            logits = self.model(tensor)
            probs = torch.sigmoid(logits).squeeze(0)
        return {
            "normal": round(probs[0].item(), 4),
            "Crackle": round(probs[1].item(), 4),
            "Wheeze": round(probs[2].item(), 4),
        }

    def predict_with_label(self, audio_path):
        probs_dict = self.predict(audio_path)
        crackle_p = probs_dict["Crackle"]
        wheeze_p = probs_dict["Wheeze"]
        normal_p = probs_dict["normal"]
        binary_score = max(crackle_p, wheeze_p)

        detected = binary_score >= self.threshold
        if detected:
            if crackle_p >= self.threshold and wheeze_p >= self.threshold:
                label = "Crackle + Wheeze"
            elif crackle_p >= self.threshold:
                label = "Crackle"
            else:
                label = "Wheeze"
        else:
            label = "Normal"

        return {
            "probabilities": probs_dict,
            "crackle_prob": crackle_p,
            "wheeze_prob": wheeze_p,
            "normal_prob": normal_p,
            "binary_score": binary_score,
            "label": label,
            "confidence": round(binary_score * 100, 1) if detected else round((1 - binary_score) * 100, 1),
            "threshold": self.threshold,
            "detected": detected,
        }

