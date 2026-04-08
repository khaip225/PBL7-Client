import torch
import torchaudio

class AudioPredictor:
    def __init__(self, model_path, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None 
        self.model_path = model_path

    def preprocess(self, audio_path):
        pass 

    def predict(self, audio_path):
        return 0.65