"""Diagnosis Service — sử dụng PipelineEngine cho pipeline 5 hành động."""

import json
import os
from datetime import datetime

from config import config


class DiagnosisService:
    """Điều phối chẩn đoán qua PipelineEngine (5 hành động)."""

    def __init__(self, pipeline_engine, storage_manager):
        self.engine = pipeline_engine
        self.storage_manager = storage_manager
        self.threshold = config.PREDICTION_THRESHOLD

    def run(self, mode: str, audio_file_path: str | None, image_file_path: str | None) -> dict:
        timestamp = datetime.now()

        if mode == "fusion":
            if not audio_file_path or not image_file_path:
                raise ValueError("Fusion mode requires both audio and image files")

            # Chạy cả 2 pipeline
            heatmap_dir = os.path.join(self.storage_manager.client_dir, "heatmaps") \
                if hasattr(self.storage_manager, 'client_dir') \
                else os.path.join(self.storage_manager.pending_dir, "heatmaps")
            attention_dir = os.path.join(self.storage_manager.client_dir, "attention") \
                if hasattr(self.storage_manager, 'client_dir') \
                else os.path.join(self.storage_manager.pending_dir, "attention")

            img_result = self.engine.run_image_pipeline(
                image_file_path, save_heatmap_dir=heatmap_dir
            )
            aud_result = self.engine.run_audio_pipeline(
                audio_file_path, save_attention_dir=attention_dir
            )

            # Kết hợp scores từ cả 2 pipeline
            disease_probs = img_result.disease_probs
            acoustic_probs = aud_result.acoustic_probs

            # Late fusion kết hợp
            late_fusion = self.engine.late_fusion(disease_probs, acoustic_probs)

            scores = {
                "audio_scores": acoustic_probs,
                "image_scores": disease_probs,
                "fusion_scores": late_fusion.fusion_scores,
            }
            heatmap_path = img_result.heatmap_path
            attention_path = aud_result.attention_map_path
            cross_modal = None
            retrieval = img_result.retrieved_audio + aud_result.retrieved_images

            labels = self._collect_labels(disease_probs, acoustic_probs)
            conf_pct = round(late_fusion.confidence * 100, 1)

            audio_detail = {
                "final_score": max(acoustic_probs.values()) if acoustic_probs else 0,
                "probabilities": acoustic_probs,
                "label": max(acoustic_probs, key=acoustic_probs.get) if acoustic_probs else "Normal",
            }

        elif mode == "image":
            if not image_file_path:
                raise ValueError("Image mode requires an image file")

            heatmap_dir = os.path.join(self.storage_manager.client_dir, "heatmaps") \
                if hasattr(self.storage_manager, 'client_dir') \
                else os.path.join(self.storage_manager.pending_dir, "heatmaps")

            result = self.engine.run_image_pipeline(
                image_file_path, save_heatmap_dir=heatmap_dir
            )

            disease_probs = result.disease_probs
            acoustic_probs = result.cross_modal_acoustic
            late_fusion = result.late_fusion

            scores = {
                "audio_scores": result.cross_modal_acoustic,
                "image_scores": result.disease_probs,
                "fusion_scores": late_fusion.fusion_scores if late_fusion else None,
            }
            heatmap_path = result.heatmap_path
            attention_path = None
            cross_modal = {
                "scores": result.cross_modal_acoustic,
                "message": result.cross_modal_message,
            }
            retrieval = [
                {
                    "file_path": r.file_path,
                    "file_name": r.file_name,
                    "similarity": r.similarity,
                    "case_id": r.case_id,
                    "disease_label": r.disease_label,
                    "acoustic_label": r.acoustic_label,
                }
                for r in result.retrieved_audio
            ]

            labels = self._collect_labels(disease_probs, acoustic_probs)
            conf_pct = round(late_fusion.confidence * 100, 1) if late_fusion else round(max(disease_probs.values()) * 100, 1)
            audio_detail = None

        elif mode == "audio":
            if not audio_file_path:
                raise ValueError("Audio mode requires an audio file")

            attention_dir = os.path.join(self.storage_manager.client_dir, "attention") \
                if hasattr(self.storage_manager, 'client_dir') \
                else os.path.join(self.storage_manager.pending_dir, "attention")

            result = self.engine.run_audio_pipeline(
                audio_file_path, save_attention_dir=attention_dir
            )

            acoustic_probs = result.acoustic_probs
            disease_probs = result.cross_modal_disease
            late_fusion = result.late_fusion

            scores = {
                "audio_scores": result.acoustic_probs,
                "image_scores": result.cross_modal_disease,
                "fusion_scores": late_fusion.fusion_scores if late_fusion else None,
            }
            heatmap_path = None
            attention_path = result.attention_map_path
            cross_modal = {
                "scores": result.cross_modal_disease,
                "message": result.cross_modal_message,
            }
            retrieval = [
                {
                    "file_path": r.file_path,
                    "file_name": r.file_name,
                    "similarity": r.similarity,
                    "case_id": r.case_id,
                    "disease_label": r.disease_label,
                    "acoustic_label": r.acoustic_label,
                }
                for r in result.retrieved_images
            ]

            labels = self._collect_labels(disease_probs, acoustic_probs)
            conf_pct = round(late_fusion.confidence * 100, 1) if late_fusion else round(max(acoustic_probs.values()) * 100, 1)
            audio_detail = {
                "final_score": max(acoustic_probs.values()) if acoustic_probs else 0,
                "probabilities": acoustic_probs,
                "label": max(acoustic_probs, key=acoustic_probs.get) if acoustic_probs else "Normal",
            }

        else:
            raise ValueError(f"Unknown mode: {mode}")

        # If nothing detected, default to Normal
        if not labels:
            labels = ["Normal"]

        # Late fusion detail
        lf_detail = None
        if late_fusion:
            lf_detail = {
                "primary_diagnosis": late_fusion.primary_diagnosis,
                "confidence": late_fusion.confidence,
                "confidence_level": late_fusion.confidence_level,
                "agreement": late_fusion.agreement,
                "fusion_scores": late_fusion.fusion_scores,
                "is_normal": late_fusion.is_normal,
            }

        audio_dest, image_dest = self.storage_manager.save_files(
            audio_file_path,
            image_file_path,
            labels,
            mode=mode,
            confidence=conf_pct / 100,
            scores=scores,
        )

        # Save audio detail JSON
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
            "cross_modal": cross_modal,
            "retrieval": retrieval,
            "late_fusion": lf_detail,
            "attention_map_path": attention_path,
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
