import torch
from torchvision import transforms
from PIL import Image
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from ai_engines.image_engine.resnet18_model import ResNet18
from ai_engines.image_engine.gradcam import GradCAM


class ImagePredictor:
    def __init__(self, model_path, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.model = ResNet18(num_classes=1, in_channels=3)
        self.load_model()
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        # Grad-CAM: hook vào conv cuối của layer4
        self.gradcam = GradCAM(self.model, self.model.layer4[-1].conv2)

    def load_model(self):
        if os.path.exists(self.model_path):
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=True)
            print(f"[ImagePredictor] Da nap weights tu {self.model_path}")
        else:
            print(f"[ImagePredictor] CANH BAO: Khong tim thay {self.model_path}. Dung random.")

    def preprocess(self, image_path):
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0)
        return tensor.to(self.device)

    def predict(self, image_path):
        """Tra ve probability cua pneumonia (0-1)."""
        tensor = self.preprocess(image_path)
        with torch.no_grad():
            output = self.model(tensor)
            prob = torch.sigmoid(output).item()
        return prob

    def predict_with_gradcam(self, image_path, save_dir=None):
        """Chay inference + sinh Grad-CAM heatmap.

        Returns:
            dict: {
                "prob": float (0-1),
                "label": "Pneumonia" | "Normal",
                "heatmap_path": str | None  (duong dan file heatmap da luu)
            }
        """
        tensor = self.preprocess(image_path)

        # Forward lan 1: lay output binh thuong
        with torch.no_grad():
            output = self.model(tensor)
            prob = torch.sigmoid(output).item()

        label = "Pneumonia" if prob > 0.5 else "Normal"

        # Sinh heatmap (can backward nen khong torch.no_grad)
        heatmap = self.gradcam.generate(tensor, class_idx=0)

        # Chồng heatmap lên ảnh gốc
        original_img = Image.open(image_path).convert("RGB")
        overlay_img = self.gradcam.overlay(original_img, heatmap)

        heatmap_path = None
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            ts = int(time.time() * 1000)
            heatmap_path = os.path.join(save_dir, f"heatmap_{ts}.png")
            overlay_img.save(heatmap_path)
            print(f"[ImagePredictor] Heatmap da luu: {heatmap_path}")

        return {
            "prob": prob,
            "label": label,
            "heatmap_path": heatmap_path,
        }