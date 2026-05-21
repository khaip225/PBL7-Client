import torch
import torchaudio
import torch.nn.functional as F
import numpy as np
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
        self.chunk_duration = config.AUDIO_CHUNK_DURATION
        self.overlap = config.AUDIO_CHUNK_OVERLAP
        self.n_mels = 64

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

    def _segment_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        chunk_samples = int(self.target_sr * self.chunk_duration)
        hop_samples = int(chunk_samples * (1 - self.overlap))
        total_samples = waveform.shape[1]

        chunks = []
        for start_idx in range(0, total_samples, hop_samples):
            end_idx = start_idx + chunk_samples
            chunk = waveform[:, start_idx:end_idx]

            if chunk.shape[1] < chunk_samples:
                padding = chunk_samples - chunk.shape[1]
                chunk = F.pad(chunk, (0, padding))

            chunks.append(chunk)

            if end_idx >= total_samples:
                break

        return torch.stack(chunks).to(self.device)

    def _preprocess_chunks(self, chunks: torch.Tensor) -> torch.Tensor:
        mel_spec = self.mel_spectrogram(chunks)
        mel_spec_db = self.amplitude_to_db(mel_spec)

        mean = mel_spec_db.mean(dim=[1, 2, 3], keepdim=True)
        std = mel_spec_db.std(dim=[1, 2, 3], keepdim=True)
        mel_spec_norm = (mel_spec_db - mean) / (std + 1e-6)

        return mel_spec_norm

    def predict(self, audio_path):
        return self.predict_segments(audio_path)["final_score"]

    def predict_segments(self, audio_path):
        self.model.eval()
        waveform = self._load_audio(audio_path)
        chunks = self._segment_waveform(waveform)
        mel = self._preprocess_chunks(chunks)

        with torch.no_grad():
            outputs = self.model(mel)
            probs = torch.sigmoid(outputs).squeeze(-1)

        probs_np = probs.detach().cpu().numpy()

        hop = self.chunk_duration * (1 - self.overlap)
        segments = []
        for idx, score in enumerate(probs_np):
            start = idx * hop
            end = start + self.chunk_duration
            segments.append({
                "index": int(idx),
                "start": float(start),
                "end": float(end),
                "score": float(score),
            })

        max_score = float(np.max(probs_np))
        avg_score = float(np.mean(probs_np))
        abnormal_chunks = np.sum(probs_np > self.threshold)
        abnormal_ratio = float(abnormal_chunks / len(probs_np))

        final_score = (max_score * 0.7) + (avg_score * 0.3)

        return {
            "final_score": final_score,
            "max_score": max_score,
            "avg_score": avg_score,
            "abnormal_chunks_ratio": abnormal_ratio,
            "total_chunks_analyzed": len(probs_np),
            "chunk_duration": self.chunk_duration,
            "overlap": self.overlap,
            "threshold": self.threshold,
            "segments": segments,
        }

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
