import torch
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from ai_engines.image_engine.densenet121_model import DenseNet121MultiLabel
from ai_engines.image_engine.gradcam import GradCAM


CLASS_NAMES = ["Normal", "Pneumonia", "COPD_Emphysema", "Fibrosis"]


class ImagePredictor:
    def __init__(self, model_path, device=None, threshold=0.5):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.threshold = threshold
        self.model = DenseNet121MultiLabel(num_classes=4)
        self.load_model()
        self.model.to(self.device)
        self.model.eval()

        # Match Stage 5 notebook: NO ImageNet normalization
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # [0,1]
        ])

        target_layer = self.model.encoder.features.denseblock4.denselayer16.conv2
        self.gradcam = GradCAM(self.model, target_layer)

    def load_model(self):
        if os.path.exists(self.model_path):
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=True)
            print(f"[ImagePredictor] Da nap weights tu {self.model_path}")
        else:
            print(f"[ImagePredictor] CANH BAO: Khong tim thay {self.model_path}. Dung random.")

    def preprocess(self, image_path):
        """Match Stage 5 notebook: cv2.imread GRAYSCALE → resize → tensor → repeat(3)."""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Không đọc được ảnh: {image_path}")
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0
        tensor = torch.from_numpy(img).unsqueeze(0).repeat(3, 1, 1)  # (3, H, W)
        return tensor.unsqueeze(0).to(self.device)  # (1, 3, H, W)

    def predict(self, image_path):
        """Tra ve multi-label probabilities."""
        tensor = self.preprocess(image_path)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.sigmoid(logits).squeeze(0)
        return {
            CLASS_NAMES[0]: round(probs[0].item(), 4),
            CLASS_NAMES[1]: round(probs[1].item(), 4),
            CLASS_NAMES[2]: round(probs[2].item(), 4),
            CLASS_NAMES[3]: round(probs[3].item(), 4),
        }

    def predict_with_gradcam(self, image_path, save_dir=None):
        """Inference + Grad-CAM heatmap cho class co probability cao nhat."""
        tensor = self.preprocess(image_path)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.sigmoid(logits).squeeze(0)

        probs_np = probs.cpu().numpy()
        best_idx = int(probs_np.argmax())
        best_class = CLASS_NAMES[best_idx]
        best_prob = float(probs_np[best_idx])

        heatmap = self.gradcam.generate(tensor, class_idx=best_idx)

        original_img = Image.open(image_path).convert("RGB")
        overlay_img = self.gradcam.overlay(original_img, heatmap)

        heatmap_path = None
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            ts = int(time.time() * 1000)
            heatmap_path = os.path.join(save_dir, f"heatmap_{best_class}_{ts}.png")
            overlay_img.save(heatmap_path)
            print(f"[ImagePredictor] Heatmap da luu: {heatmap_path}")

        labels = {}
        for i, name in enumerate(CLASS_NAMES):
            p = float(probs_np[i])
            labels[name] = {
                "probability": round(p, 4),
                "label": name if p >= self.threshold else "Normal",
                "detected": p >= self.threshold,
            }

        probs_out = {CLASS_NAMES[i]: float(probs_np[i]) for i in range(len(CLASS_NAMES))}
        return {
            "probabilities": probs_out,
            "best_class": best_class,
            "best_probability": best_prob,
            "binary_score": best_prob,
            "detected": best_prob >= self.threshold,
            "labels": labels,
            "heatmap_path": heatmap_path,
        }
