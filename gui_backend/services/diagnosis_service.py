import json
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
        audio_detail = None

        if mode == "fusion":
            if not audio_file_path or not image_file_path:
                raise ValueError("Fusion mode requires both audio and image files")
            audio_result = self.audio_predictor.predict_with_label(audio_file_path)
            audio_detail = {"final_score": audio_result["binary_score"],
                           "probabilities": audio_result["probabilities"],
                           "label": audio_result["label"]}
            image_result = self.image_predictor.predict_with_gradcam(image_file_path)
            fusion_scores = self.fusion_engine.fuse(
                audio_result["probabilities"], image_result["probabilities"]
            )
            scores = {
                "audio_scores": audio_result["probabilities"],
                "image_scores": image_result["probabilities"],
                "fusion_scores": fusion_scores,
            }
            heatmap_path = image_result.get("heatmap_path")
            # Multi-label: collect all classes above threshold
            labels = self._collect_labels(
                disease_scores=fusion_scores,
                acoustic_scores=audio_result["probabilities"],
            )
            conf_pct = round(max(
                [v for v in fusion_scores.values() if isinstance(v, (int, float))] + [0]
            ) * 100, 1)

        elif mode == "audio":
            if not audio_file_path:
                raise ValueError("Audio mode requires an audio file")
            audio_result = self.audio_predictor.predict_with_label(audio_file_path)
            audio_detail = {"final_score": audio_result["binary_score"],
                           "probabilities": audio_result["probabilities"],
                           "label": audio_result["label"]}

            # Cross-modal inference: audio → disease probabilities
            inferred_image = self.fusion_engine.audio_to_disease(audio_result["probabilities"])

            scores = {
                "audio_scores": audio_result["probabilities"],
                "image_scores": inferred_image,
                "fusion_scores": None,
            }
            labels = self._collect_labels(
                disease_scores=inferred_image,
                acoustic_scores=audio_result["probabilities"],
            )
            conf_pct = round(max(
                [v for v in audio_result["probabilities"].values()] + [0]
            ) * 100, 1)

        elif mode == "image":
            if not image_file_path:
                raise ValueError("Image mode requires an image file")
            heatmap_dir = os.path.join(self.storage_manager.client_dir, "heatmaps") if hasattr(self.storage_manager, 'client_dir') else os.path.join(self.storage_manager.pending_dir, "heatmaps")
            image_result = self.image_predictor.predict_with_gradcam(
                image_file_path, save_dir=heatmap_dir
            )

            # Cross-modal inference: image → acoustic attributes
            inferred_audio = self.fusion_engine.image_to_acoustic(image_result["probabilities"])

            scores = {
                "audio_scores": inferred_audio,
                "image_scores": image_result["probabilities"],
                "fusion_scores": None,
            }
            heatmap_path = image_result.get("heatmap_path")
            labels = self._collect_labels(
                disease_scores=image_result["probabilities"],
                acoustic_scores=inferred_audio,
            )
            conf_pct = round(max(
                [v for v in image_result["probabilities"].values()] + [0]
            ) * 100, 1)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # If nothing detected, default to Normal
        if not labels:
            labels = ["Normal"]

        audio_dest, image_dest = self.storage_manager.save_files(
            audio_file_path,
            image_file_path,
            labels,
            mode=mode,
            confidence=conf_pct / 100,
            scores=scores,
        )

        # Save audio detail JSON alongside the audio file
        if audio_dest and audio_detail is not None:
            json_path = os.path.splitext(audio_dest)[0] + ".json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(audio_detail, f, ensure_ascii=True, indent=2)

        return {
            "mode": mode,
            "result": {
                "labels": labels,
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

    def _collect_labels(self, disease_scores: dict, acoustic_scores: dict) -> list[str]:
        """Collect all disease/acoustic classes with prob >= threshold."""
        labels = []
        for name in ["Pneumonia", "COPD_Emphysema", "Fibrosis"]:
            if disease_scores.get(name, 0) >= self.threshold:
                labels.append(name)
        for name in ["Crackle", "Wheeze"]:
            if acoustic_scores.get(name, 0) >= self.threshold:
                labels.append(name)
        return labels
