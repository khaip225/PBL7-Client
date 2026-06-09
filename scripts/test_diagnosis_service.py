"""Test DiagnosisService.run() và kiểm tra API response format."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_engines.pipeline_engine import PipelineEngine
from local_managers.storage_manager import StorageManager
from gui_backend.services.diagnosis_service import DiagnosisService
from config import config

STAGE4_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "stage4_best_model.pth")
IMAGE_HEAD_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "Global_Image_best.pth")
AUDIO_HEAD_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "Global_Audio_best.pth")

engine = PipelineEngine(
    stage4_path=STAGE4_PATH,
    image_head_path=IMAGE_HEAD_PATH,
    audio_head_path=AUDIO_HEAD_PATH,
    device='cpu',
)

storage = StorageManager(client_id=config.CLIENT_ID)
service = DiagnosisService(engine, storage)

test_image = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "Local_Data", "trash", "image_files", "Normal_20260604_160504.jpeg")

print("=== Test diagnosis_service.run(mode=image) ===")
result = service.run("image", None, test_image)

# Print keys and key info
print("Mode:", result["mode"])
print("Result:", result["result"])
print("Has cross_modal:", result["cross_modal"] is not None)
print("Has retrieval:", result["retrieval"] is not None, "count:", len(result["retrieval"]))
print("Has late_fusion:", result["late_fusion"] is not None)
print("Has heatmap:", "heatmap_path" in result and result["heatmap_path"] is not None)
print("Has attention_map:", result.get("attention_map_path"))
print()

# Full JSON dump
print("Full scores:")
print(json.dumps(result["scores"], indent=2))
print()
if result["cross_modal"]:
    cm = result["cross_modal"]
    print("Cross-modal:")
    print("  scores:", cm["scores"])
    if cm["message"]:
        print("  message:", cm["message"][:100], "...")
print()
if result["late_fusion"]:
    print("Late Fusion:")
    print(json.dumps(result["late_fusion"], indent=2, ensure_ascii=False))

print("\n=== API Response Format OK! ===")
