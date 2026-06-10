"""Dataclass result types for pipeline outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class RetrievalItem:
    file_path: str
    file_name: str
    similarity: float
    case_id: str = ""
    disease_label: str = ""
    acoustic_label: str = ""


@dataclass
class LateFusionResult:
    primary_diagnosis: str
    confidence: float
    confidence_level: str    # "Rất cao" / "Cao" / "Trung bình" / "Thấp"
    agreement: str
    fusion_scores: dict
    is_normal: bool


@dataclass
class ImagePipelineResult:
    disease_probs: dict
    cross_modal_acoustic: dict
    cross_modal_message: str
    retrieved_audio: list = field(default_factory=list)
    heatmap_path: str = ""
    gradcam_enabled: bool = True
    late_fusion: Optional[LateFusionResult] = None
    embedding: Optional[np.ndarray] = None
    timestamp: str = ""
    mode: str = "image"


@dataclass
class AudioPipelineResult:
    acoustic_probs: dict
    cross_modal_disease: dict
    cross_modal_message: str
    retrieved_images: list = field(default_factory=list)
    attention_map_path: str = ""
    attention_enabled: bool = True
    late_fusion: Optional[LateFusionResult] = None
    embedding: Optional[np.ndarray] = None
    timestamp: str = ""
    mode: str = "audio"
