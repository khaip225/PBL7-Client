"""Backward-compatibility shim — re-exports from the modular package.

All existing `from ai_engines.pipeline_engine import PipelineEngine` imports
continue to work unchanged after the split into:
    __init__.py   — package marker
    constants.py  — IMAGE_CLASS_NAMES, AUDIO_CLASS_NAMES, PROTO_DISPLAY, PROTO_NAME_MAP
    models.py     — DenseNet121EncoderNB, ASTEncoderNB, MomentumPrototypeModuleNB
    schemas.py    — RetrievalItem, LateFusionResult, ImagePipelineResult, AudioPipelineResult
    xai.py        — GradCAMWrapper, GradCAM
    engine.py     — PipelineEngine (the orchestrator)
"""

import os
import sys

# Path setup (keep for consumers that rely on this module adding BASE_DIR to path)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Re-export the public API
from ai_engines.engine import PipelineEngine
from ai_engines.schemas import (
    RetrievalItem, LateFusionResult, ImagePipelineResult, AudioPipelineResult,
)
from ai_engines.models import (
    DenseNet121EncoderNB, ASTEncoderNB, MomentumPrototypeModuleNB,
)
from ai_engines.constants import (
    IMAGE_CLASS_NAMES, AUDIO_CLASS_NAMES,
    IMAGE_DISEASE_NAMES, AUDIO_ATTR_NAMES,
    PROTO_DISPLAY, PROTO_NAME_MAP,
)

__all__ = [
    "PipelineEngine",
    "RetrievalItem", "LateFusionResult",
    "ImagePipelineResult", "AudioPipelineResult",
    "DenseNet121EncoderNB", "ASTEncoderNB", "MomentumPrototypeModuleNB",
    "IMAGE_CLASS_NAMES", "AUDIO_CLASS_NAMES",
    "IMAGE_DISEASE_NAMES", "AUDIO_ATTR_NAMES",
    "PROTO_DISPLAY", "PROTO_NAME_MAP",
]
