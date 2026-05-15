import torch
import torchaudio
import torch.nn.functional as F
import soundfile as sf

from config import config


class AudioPredictor:
    def __init__(self, model_path, device=None, threshold=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.threshold = threshold if threshold is not None else config.PREDICTION_THRESHOLD

        from ai_engines.audio_engine.cnn14_model import CNN14
        self.model = CNN14().to(self.device)
        self._load_model()

        self.target_sr = 16000
        self.max_length = 15
        self.n_mels = 64

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.target_sr, n_fft=1024, hop_length=512, n_mels=self.n_mels
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

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

        mel_spec = self.mel_spectrogram(waveform)
        mel_spec_db = self.amplitude_to_db(mel_spec)

        mean = mel_spec_db.mean()
        std = mel_spec_db.std()
        mel_spec_db = (mel_spec_db - mean) / (std + 1e-6)

        return mel_spec_db.unsqueeze(0).to(self.device)

    def predict(self, audio_path):
        self.model.eval()
        with torch.no_grad():
            mel = self.preprocess(audio_path)
            output = self.model(mel)
            prob = torch.sigmoid(output).item()
            return prob

    def predict_with_label(self, audio_path):
        prob = self.predict(audio_path)
        label = "Abnormal" if prob >= self.threshold else "Normal"
        confidence = prob if prob >= self.threshold else (1 - prob)
        return {
            "probability": prob,
            "label": label,
            "confidence": round(confidence * 100, 1),
            "threshold": self.threshold,
        }
