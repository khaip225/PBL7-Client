import os
import shutil
from datetime import datetime
from config import config


class StorageManager:
    def __init__(self, base_dir="./Local_Data", client_id=1):
        self.client_id = client_id
        self.client_dir = os.path.join(base_dir, f"Client_{client_id}")
        self.audio_dir = os.path.join(self.client_dir, "audio_files")
        self.image_dir = os.path.join(self.client_dir, "image_files")

        os.makedirs(self.audio_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)

        self.fl_data_dir = config.FL_DATA_DIR
        self.fl_image_dir = os.path.join(self.fl_data_dir, "fl_image")

    def save_files(self, audio_source, image_source, label):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_dest, image_dest = None, None

        if audio_source is not None:
            audio_ext = os.path.splitext(audio_source)[1]
            new_audio_name = f"{label}_{timestamp}{audio_ext}"
            audio_dest = os.path.join(self.audio_dir, new_audio_name)
            shutil.copy2(audio_source, audio_dest)

        if image_source is not None:
            image_ext = os.path.splitext(image_source)[1]
            new_image_name = f"{label}_{timestamp}{image_ext}"
            image_dest = os.path.join(self.image_dir, new_image_name)
            shutil.copy2(image_source, image_dest)

        self._sync_to_fl_data(audio_source, image_source, label, timestamp)

        return audio_dest, image_dest

    def _sync_to_fl_data(self, audio_source, image_source, label, timestamp):
        # Map disease label to multi-label codes
        disease_map = {
            "Normal": (0, 0, 0),
            "Pneumonia": (1, 0, 0),
            "COPD_Emphysema": (0, 1, 0),
            "Fibrosis": (0, 0, 1),
            "Crackle": (1, 0, 0),
            "Wheeze": (0, 1, 0),
            "Crackle + Wheeze": (1, 1, 0),
        }
        pneu, copd, fibr = disease_map.get(label, (0, 0, 0))

        # Image -> fl_image/ with multi-label CSV
        if image_source:
            dest_dir = self.fl_image_dir
            os.makedirs(dest_dir, exist_ok=True)
            ext = os.path.splitext(image_source)[1]
            dest_name = f"{label}_{timestamp}{ext}"
            dest = os.path.join(dest_dir, dest_name)
            shutil.copy2(image_source, dest)

            csv_dir = os.path.join(self.fl_data_dir, "metadata", "image_fl")
            os.makedirs(csv_dir, exist_ok=True)
            csv_path = os.path.join(csv_dir, f"client_{self.client_id}_train.csv")

            if not os.path.exists(csv_path):
                with open(csv_path, "w", encoding="utf-8") as f:
                    f.write("path,Pneumonia,COPD_Emphysema,Fibrosis\n")

            with open(csv_path, "a", encoding="utf-8") as f:
                f.write(f"{dest_name},{pneu},{copd},{fibr}\n")

        # Audio -> fl_audio/ + multi-label CSV
        if audio_source:
            audio_dir = os.path.join(self.fl_data_dir, "fl_audio")
            os.makedirs(audio_dir, exist_ok=True)
            ext = os.path.splitext(audio_source)[1]
            new_name = f"{label}_{timestamp}{ext}"
            dest = os.path.join(audio_dir, new_name)
            shutil.copy2(audio_source, dest)

            csv_dir = os.path.join(self.fl_data_dir, "metadata", "audio_fl")
            os.makedirs(csv_dir, exist_ok=True)
            csv_path = os.path.join(csv_dir, f"client_{self.client_id}_train.csv")

            if not os.path.exists(csv_path):
                with open(csv_path, "w", encoding="utf-8") as f:
                    f.write("path,crackle,wheeze\n")

            # Audio labels: Crackle=1, Wheeze=1 for "Crackle + Wheeze"
            audio_label_map = {
                "Normal": (0, 0),
                "Crackle": (1, 0),
                "Wheeze": (0, 1),
                "Crackle + Wheeze": (1, 1),
                "Pneumonia": (1, 0),  # Pneumonia implies crackle
                "COPD_Emphysema": (0, 1),  # COPD implies wheeze
                "Fibrosis": (1, 0),  # Fibrosis implies fine crackle
            }
            cr, wh = audio_label_map.get(label, (0, 0))

            with open(csv_path, "a", encoding="utf-8") as f:
                f.write(f"{new_name},{cr},{wh}\n")
