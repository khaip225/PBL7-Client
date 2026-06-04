import torch
import torchaudio
import torch.nn.functional as F
import numpy as np
import soundfile as sf
from PIL import Image

from config import config


ACOUSTIC_CLASS_NAMES = ["Crackle", "Wheeze"]


class AudioPredictor:
    def __init__(self, model_path, device=None, threshold=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.threshold = threshold if threshold is not None else config.PREDICTION_THRESHOLD

        from ai_engines.audio_engine.ast_model import ASTMultiLabel
        self.model = ASTMultiLabel(num_classes=2).to(self.device)
        self._load_model()

        self.target_sr = 16000
        self.chunk_duration = config.AUDIO_CHUNK_DURATION if hasattr(config, 'AUDIO_CHUNK_DURATION') else 15
        self.overlap = config.AUDIO_CHUNK_OVERLAP if hasattr(config, 'AUDIO_CHUNK_OVERLAP') else 0.0
        self.max_length = 15
        self.n_mels = 128

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.target_sr, n_fft=1024, hop_length=512, n_mels=self.n_mels
        ).to(self.device)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB().to(self.device)

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

    def _load_audio(self, audio_path: str) -> torch.Tensor:
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

        return waveform

    def preprocess(self, audio_path):
        """AST preprocessing: mel -> Image -> ViT input."""
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

        target_samples = self.target_sr * self.max_length
        current_samples = waveform.shape[1]
        if current_samples < target_samples:
            padding = target_samples - current_samples
            waveform = F.pad(waveform, (0, padding))
        elif current_samples > target_samples:
            waveform = waveform[:, :target_samples]

        waveform = waveform.to(self.device)
        mel_spec = self.mel_spectrogram(waveform)
        mel_spec_db = self.amplitude_to_db(mel_spec)

        mean = mel_spec_db.mean()
        std = mel_spec_db.std()
        mel_spec_db = (mel_spec_db - mean) / (std + 1e-6)

        mel_np = mel_spec_db.squeeze(0).cpu().numpy()
        mel_img = Image.fromarray(
            ((mel_np - mel_np.min()) / (mel_np.max() - mel_np.min() + 1e-8) * 255).astype(np.uint8)
        ).convert("RGB")
        mel_img = mel_img.resize((224, 224), Image.BICUBIC)

        from torchvision import transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        return transform(mel_img).unsqueeze(0).to(self.device)

    def predict(self, audio_path):
        """Tra ve multi-label acoustic probabilities."""
        self.model.eval()
        with torch.no_grad():
            tensor = self.preprocess(audio_path)
            logits = self.model(tensor)
            probs = torch.sigmoid(logits).squeeze(0)
        return {
            "Crackle": round(probs[0].item(), 4),
            "Wheeze": round(probs[1].item(), 4),
        }

    def predict_with_label(self, audio_path):
        probs = self.predict(audio_path)
        crackle_p = probs["Crackle"]
        wheeze_p = probs["Wheeze"]
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
            "probabilities": probs,
            "crackle_prob": crackle_p,
            "wheeze_prob": wheeze_p,
            "binary_score": binary_score,
            "label": label,
            "confidence": round(binary_score * 100, 1) if detected else round((1 - binary_score) * 100, 1),
            "threshold": self.threshold,
            "detected": detected,
        }
