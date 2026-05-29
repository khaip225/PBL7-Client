import customtkinter as ctk
import os
from tkinter import filedialog
from ai_engines.audio_engine.predictor import AudioPredictor
from ai_engines.image_engine.predictor import ImagePredictor
from ai_engines.fusion import OntologyFusion
from local_managers.storage_manager import StorageManager
from config import config

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

THRESHOLD = config.PREDICTION_THRESHOLD

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_MODEL = os.path.join(BASE_DIR, "ai_engines", "current_weights", "best_global_audio.pth")
IMAGE_MODEL = os.path.join(BASE_DIR, "ai_engines", "current_weights", "best_global_image.pth")

DISEASE_NAMES = ["Pneumonia", "COPD_Emphysema", "Fibrosis"]
ACOUSTIC_NAMES = ["Crackle", "Wheeze"]


class DiagnosisApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("PBL7 - Chuan doan benh ho hap da phuong thuc")
        self.geometry("750x750")

        self.audio_predictor = AudioPredictor(AUDIO_MODEL, threshold=THRESHOLD)
        self.image_predictor = ImagePredictor(IMAGE_MODEL)
        self.fusion_engine = OntologyFusion(audio_weight=0.4)
        self.storage_manager = StorageManager(client_id=1)

        self.audio_path = None
        self.image_path = None

        # Mode selection
        self.lbl_mode = ctk.CTkLabel(self, text="Chon che do chan doan:", font=("Arial", 14, "bold"))
        self.lbl_mode.pack(pady=10)

        self.mode_var = ctk.StringVar(value="fusion")
        self.radio_fusion = ctk.CTkRadioButton(
            self, text="Che do Fusion (Am thanh + X-quang)",
            variable=self.mode_var, value="fusion", command=self.update_mode
        )
        self.radio_fusion.pack(pady=5)

        self.radio_audio = ctk.CTkRadioButton(
            self, text="Chi Am thanh",
            variable=self.mode_var, value="audio", command=self.update_mode
        )
        self.radio_audio.pack(pady=5)

        self.radio_image = ctk.CTkRadioButton(
            self, text="Chi X-quang",
            variable=self.mode_var, value="image", command=self.update_mode
        )
        self.radio_image.pack(pady=5)

        self.separator1 = ctk.CTkLabel(self, text="─" * 50)
        self.separator1.pack(pady=15)

        self.btn_load_audio = ctk.CTkButton(self, text="Tai file Am thanh (.wav)", command=self.load_audio)
        self.btn_load_audio.pack(pady=10)
        self.lbl_audio = ctk.CTkLabel(self, text="Chua co file am thanh", text_color="gray")
        self.lbl_audio.pack()

        self.btn_load_image = ctk.CTkButton(self, text="Tai file X-quang (.png/.jpg)", command=self.load_image)
        self.btn_load_image.pack(pady=10)
        self.lbl_image = ctk.CTkLabel(self, text="Chua co file X-quang", text_color="gray")
        self.lbl_image.pack()

        self.separator2 = ctk.CTkLabel(self, text="─" * 50)
        self.separator2.pack(pady=15)

        self.btn_predict = ctk.CTkButton(
            self, text="CHAN DOAN & LUU TRU", command=self.process_diagnosis,
            fg_color="green", height=40
        )
        self.btn_predict.pack(pady=20)

        self.lbl_result = ctk.CTkLabel(self, text="Ket qua: ...", font=("Arial", 18, "bold"))
        self.lbl_result.pack(pady=5)

        self.lbl_details = ctk.CTkLabel(self, text="", font=("Arial", 12))
        self.lbl_details.pack(pady=10)

    def load_audio(self):
        self.audio_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])
        if self.audio_path:
            self.lbl_audio.configure(text=os.path.basename(self.audio_path), text_color="white")

    def load_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if self.image_path:
            self.lbl_image.configure(text=os.path.basename(self.image_path), text_color="white")

    def update_mode(self):
        mode = self.mode_var.get()
        if mode == "fusion":
            self.btn_load_audio.configure(state="normal")
            self.btn_load_image.configure(state="normal")
        elif mode == "audio":
            self.btn_load_audio.configure(state="normal")
            self.btn_load_image.configure(state="disabled")
            self.image_path = None
            self.lbl_image.configure(text="Chua co file X-quang", text_color="gray")
        else:
            self.btn_load_audio.configure(state="disabled")
            self.btn_load_image.configure(state="normal")
            self.audio_path = None
            self.lbl_audio.configure(text="Chua co file am thanh", text_color="gray")

    def process_diagnosis(self):
        mode = self.mode_var.get()

        if mode == "fusion":
            if not self.audio_path or not self.image_path:
                self.lbl_result.configure(text="Loi: Vui long tai du 2 file!", text_color="red")
                return

            audio_result = self.audio_predictor.predict_with_label(self.audio_path)
            image_result = self.image_predictor.predict_with_gradcam(self.image_path)
            fusion_scores = self.fusion_engine.fuse(
                audio_result["probabilities"], image_result["probabilities"]
            )
            primary, conf = self.fusion_engine.get_primary_disease(fusion_scores)

            label = primary if conf >= THRESHOLD else "Normal"
            display = f"Ket qua (Fusion): {label} ({conf*100:.1f}%)"
            self.lbl_result.configure(text=display, text_color="white")

            details = "  ".join(
                f"{d}: {fusion_scores[d]*100:.1f}%" for d in DISEASE_NAMES
            )
            self.lbl_details.configure(text=f"🫁 {details}\n\n🔊 Am thanh: Crackle={audio_result['probabilities']['Crackle']*100:.1f}%  Wheeze={audio_result['probabilities']['Wheeze']*100:.1f}%")
            self.storage_manager.save_files(self.audio_path, self.image_path, label)

        elif mode == "audio":
            if not self.audio_path:
                self.lbl_result.configure(text="Loi: Vui long tai file am thanh!", text_color="red")
                return

            audio_result = self.audio_predictor.predict_with_label(self.audio_path)
            label = audio_result["label"]
            conf = audio_result["confidence"]
            display = f"Ket qua (Am thanh): {label} ({conf:.1f}%)"
            self.lbl_result.configure(text=display, text_color="white")
            self.lbl_details.configure(
                text=f"🔊 Crackle: {audio_result['crackle_prob']*100:.1f}%  |  "
                     f"Wheeze: {audio_result['wheeze_prob']*100:.1f}%"
            )
            self.storage_manager.save_files(self.audio_path, None, label)

        else:  # image
            if not self.image_path:
                self.lbl_result.configure(text="Loi: Vui long tai file X-quang!", text_color="red")
                return

            image_result = self.image_predictor.predict_with_gradcam(self.image_path)
            probs = image_result["probabilities"]
            best = image_result["best_class"]
            conf = image_result["best_probability"]
            label = best if image_result["detected"] else "Normal"

            display = f"Ket qua (X-quang): {label} ({conf*100:.1f}%)"
            self.lbl_result.configure(text=display, text_color="white")
            self.lbl_details.configure(
                text=f"🫁  " + "  ".join(
                    f"{d}: {probs[d]*100:.1f}%" for d in DISEASE_NAMES
                ) + f"\n📸 Heatmap: {image_result.get('heatmap_path', 'N/A')}"
            )
            self.storage_manager.save_files(None, self.image_path, label)


if __name__ == "__main__":
    app = DiagnosisApp()
    app.mainloop()
