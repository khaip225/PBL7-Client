import torch
from PIL import Image

class ImagePredictor:
    def __init__(self, model_path, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_path = model_path

    def preprocess(self, image_path):
        pass 

    def predict(self, image_path):
        return 0.70