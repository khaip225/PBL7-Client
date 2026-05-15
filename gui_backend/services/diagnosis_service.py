import os
import shutil
import tempfile
from datetime import datetime

from config import config


class DiagnosisService:
    def __init__(self, audio_predictor, image_predictor, fusion_engine, storage_manager):
        self.audio_predictor = audio_predictor
        self.image_predictor = image_predictor
        self.fusion_engine = fusion_engine
        self.storage_manager = storage_manager
        self.threshold = config.PREDICTION_THRESHOLD

    def run(self, mode: str, audio_file_path: str | None, image_file_path: str | None) -> dict:
        timestamp = datetime.now()

        if mode == "fusion":
            if not audio_file_path or not image_file_path:
                raise ValueError("Fusion mode requires both audio and image files")
            audio_score = self.audio_predictor.predict(audio_file_path)
            image_score = self.image_predictor.predict(image_file_path)
            final_score = self.fusion_engine.fuse(audio_score, image_score)
            scores = {"audio_score": audio_score, "image_score": image_score, "fusion_score": final_score}
        elif mode == "audio":
            if not audio_file_path:
                raise ValueError("Audio mode requires an audio file")
            audio_score = self.audio_predictor.predict(audio_file_path)
            final_score = audio_score
            scores = {"audio_score": audio_score, "image_score": None, "fusion_score": None}
        elif mode == "image":
            if not image_file_path:
                raise ValueError("Image mode requires an image file")
            image_score = self.image_predictor.predict(image_file_path)
            final_score = image_score
            scores = {"audio_score": None, "image_score": image_score, "fusion_score": None}
        else:
            raise ValueError(f"Unknown mode: {mode}")

        label = "Abnormal" if final_score > self.threshold else "Normal"
        confidence = (final_score * 100) if label == "Abnormal" else ((1 - final_score) * 100)

        audio_dest, image_dest = self.storage_manager.save_files(
            audio_file_path, image_file_path, label
        )

        return {
            "mode": mode,
            "result": {
                "label": label,
                "confidence": round(confidence, 1),
                "threshold": self.threshold,
            },
            "scores": {
                "audio_score": round(scores["audio_score"], 6) if scores["audio_score"] is not None else None,
                "image_score": round(scores["image_score"], 6) if scores["image_score"] is not None else None,
                "fusion_score": round(scores["fusion_score"], 6) if scores["fusion_score"] is not None else None,
            },
            "saved": {
                "audio_dest": audio_dest,
                "image_dest": image_dest,
            },
            "timestamp": timestamp.isoformat(),
        }
