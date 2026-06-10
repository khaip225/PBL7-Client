"""AudioPredictor — dùng AST mô hình thực sự với mel-spectrogram [1, T, F]."""

import torch
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
        """Match notebook Stage 1: librosa mel-spec → power_to_db → z-score → tensor [1, T, F].

        Notebook create_mel_spectrogram + preprocess_audio:
        1. librosa.load(sr=None) — giữ sample rate gốc
        2. melspectrogram(n_fft=1024, hop_length=320, n_mels=128)
        3. power_to_db(ref=np.max)
        4. z-score normalize
        5. tile nếu < 384 frames, crop nếu > 384 frames
        """
        import librosa
        import numpy as np

        # Đọc WAV → mono, giữ nguyên sample rate gốc
        try:
            y, sr = librosa.load(audio_path, sr=None, mono=True)
        except Exception:
            waveform_np, sr = sf.read(audio_path)
            if waveform_np.ndim > 1:
                y = waveform_np.mean(axis=1)
            else:
                y = waveform_np.astype(np.float32)

        # Mel-spectrogram khớp notebook: n_fft=1024, hop_length=320, n_mels=128, power_to_db
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=1024, hop_length=320, n_mels=self.n_mels,
        )
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)  # (128, T)

        # Z-score normalize — khớp preprocess_audio trong notebook
        log_mel_spec = (log_mel_spec - log_mel_spec.mean()) / (log_mel_spec.std() + 1e-6)

        # Pad/crop time frames về max_frames — dùng tile như notebook
        if log_mel_spec.shape[1] < self.max_frames:
            repeats = int(np.ceil(self.max_frames / log_mel_spec.shape[1]))
            log_mel_spec = np.tile(log_mel_spec, (1, repeats))[:, :self.max_frames]
        else:
            log_mel_spec = log_mel_spec[:, :self.max_frames]

        # spec_tensor shape: (1, T, F) — khớp notebook
        spec_tensor = torch.from_numpy(log_mel_spec.T).float().unsqueeze(0)  # (1, 384, 128)
        return spec_tensor.to(self.device)

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

