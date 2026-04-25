import torch
from torchvision import transforms
from PIL import Image
import os
import sys

# Ensure imports work regardless of execution location
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from ai_engines.image_engine.resnet18_model import ResNet18

class ImagePredictor:
    """
    Image predictor for binary classification: Normal vs Pneumonia
    
    Model architecture:
        - Input: X-ray images (224x224 RGB)
        - Model: ResNet18 with 1 output neuron
        - Activation: Sigmoid (probability between 0-1)
        - Threshold: 0.5 (>0.5 = Pneumonia, <=0.5 = Normal)
    """
    
    def __init__(self, model_path, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.model = ResNet18(num_classes=1, in_channels=3)  # 1 output for binary classification
        self.load_model()
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
    def load_model(self):
        if os.path.exists(self.model_path):
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=True)
            print(f"[ImagePredictor] ✅ Đã nạp weights từ {self.model_path}")
        else:
            print(f"[ImagePredictor] ⚠️ CẢNH BÁO: Không tìm thấy file {self.model_path}. Dùng tạ khởi tạo ngẫu nhiên.")

    def preprocess(self, image_path):
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0)
        return tensor.to(self.device)

    def predict(self, image_path):
        """
        Predict probability of pneumonia for given X-ray image.
        
        Returns:
            float: Probability score (0-1)
                  >0.5: Pneumonia (Abnormal)
                  <=0.5: Normal
        """
        tensor = self.preprocess(image_path)
        with torch.no_grad():
            output = self.model(tensor)
            raw_output = output.item()
            prob = torch.sigmoid(output).item()  # Output is 1 value, apply sigmoid
        print(f"[ImagePredictor] Raw output: {raw_output:.6f}, Sigmoid prob: {prob:.6f}")
        return prob