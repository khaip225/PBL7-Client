import os
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
        heatmap_path = None

        if mode == "fusion":
            if not audio_file_path or not image_file_path:
                raise ValueError("Fusion mode requires both audio and image files")
            audio_result = self.audio_predictor.predict_with_label(audio_file_path)
            image_result = self.image_predictor.predict_with_gradcam(image_file_path)
            fusion_scores = self.fusion_engine.fuse(
                audio_result["probabilities"], image_result["probabilities"]
            )
            primary_disease, confidence = self.fusion_engine.get_primary_disease(fusion_scores)
            scores = {
                "audio_scores": audio_result["probabilities"],
                "image_scores": image_result["probabilities"],
                "fusion_scores": fusion_scores,
            }
            heatmap_path = image_result.get("heatmap_path")
            label = primary_disease
            conf_pct = round(confidence * 100, 1)

        elif mode == "audio":
            if not audio_file_path:
                raise ValueError("Audio mode requires an audio file")
            audio_result = self.audio_predictor.predict_with_label(audio_file_path)
            scores = {
                "audio_scores": audio_result["probabilities"],
                "image_scores": None,
                "fusion_scores": None,
            }
            label = audio_result["label"]
            conf_pct = audio_result["confidence"]

        elif mode == "image":
            if not image_file_path:
                raise ValueError("Image mode requires an image file")
            heatmap_dir = os.path.join(self.storage_manager.client_dir, "heatmaps")
            image_result = self.image_predictor.predict_with_gradcam(
                image_file_path, save_dir=heatmap_dir
            )
            scores = {
                "audio_scores": None,
                "image_scores": image_result["probabilities"],
                "fusion_scores": None,
            }
            heatmap_path = image_result.get("heatmap_path")
            label = image_result["best_class"] if image_result["detected"] else "Normal"
            conf_pct = round(image_result["best_probability"] * 100, 1)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        audio_dest, image_dest = self.storage_manager.save_files(
            audio_file_path, image_file_path, label
        )

        return {
            "mode": mode,
            "result": {
                "label": label,
                "confidence": conf_pct,
                "threshold": self.threshold,
            },
            "scores": {
                "audio_scores": scores["audio_scores"],
                "image_scores": scores["image_scores"],
                "fusion_scores": scores["fusion_scores"],
            },
            "saved": {
                "audio_dest": audio_dest,
                "image_dest": image_dest,
            },
            "heatmap_path": heatmap_path,
            "timestamp": timestamp.isoformat(),
        }
