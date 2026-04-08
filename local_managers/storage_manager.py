import os
import shutil
from datetime import datetime

class StorageManager:
    def __init__(self, base_dir="C:\\Users\\phant\\Documents\\Study\\PBL7\\Local_Data", client_id=1):
        self.client_dir = os.path.join(base_dir, f"Client_{client_id}")
        self.audio_dir = os.path.join(self.client_dir, "audio_files")
        self.image_dir = os.path.join(self.client_dir, "image_files")
        
        os.makedirs(self.audio_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)

    def save_files(self, audio_source, image_source, label):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        audio_ext = os.path.splitext(audio_source)[1]
        new_audio_name = f"{label}_{timestamp}{audio_ext}"
        audio_dest = os.path.join(self.audio_dir, new_audio_name)
        shutil.copy2(audio_source, audio_dest)
        
        image_ext = os.path.splitext(image_source)[1]
        new_image_name = f"{label}_{timestamp}{image_ext}"
        image_dest = os.path.join(self.image_dir, new_image_name)
        shutil.copy2(image_source, image_dest)
        
        return audio_dest, image_dest