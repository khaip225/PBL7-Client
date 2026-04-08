import customtkinter as ctk
from tkinter import filedialog
from ai_engines.audio_engine.predictor import AudioPredictor
from ai_engines.image_engine.predictor import ImagePredictor
from ai_engines.fusion import LateFusion
from local_managers.storage_manager import StorageManager

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class DiagnosisApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("PBL7 - Chẩn đoán viêm phổi đa phương thức")
        self.geometry("600x500")
        
        self.audio_predictor = AudioPredictor("ai_engines/current_weights/best_global_audio.pth")
        self.image_predictor = ImagePredictor("ai_engines/current_weights/best_global_image.pth")
        self.fusion_engine = LateFusion(audio_weight=0.6, image_weight=0.4)
        self.storage_manager = StorageManager(client_id=1)
        
        self.audio_path = None
        self.image_path = None
        
        self.btn_load_audio = ctk.CTkButton(self, text="Tải file Âm thanh (.wav)", command=self.load_audio)
        self.btn_load_audio.pack(pady=20)
        
        self.lbl_audio = ctk.CTkLabel(self, text="Chưa có file âm thanh")
        self.lbl_audio.pack()
        
        self.btn_load_image = ctk.CTkButton(self, text="Tải file X-quang (.png/.jpg)", command=self.load_image)
        self.btn_load_image.pack(pady=20)
        
        self.lbl_image = ctk.CTkLabel(self, text="Chưa có file X-quang")
        self.lbl_image.pack()
        
        self.btn_predict = ctk.CTkButton(self, text="CHẨN ĐOÁN & LƯU TRỮ", command=self.process_diagnosis, fg_color="green", height=40)
        self.btn_predict.pack(pady=30)
        
        self.lbl_result = ctk.CTkLabel(self, text="Kết quả: ...", font=("Arial", 18, "bold"))
        self.lbl_result.pack()

    def load_audio(self):
        self.audio_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])
        if self.audio_path:
            self.lbl_audio.configure(text=self.audio_path)

    def load_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if self.image_path:
            self.lbl_image.configure(text=self.image_path)

    def process_diagnosis(self):
        if not self.audio_path or not self.image_path:
            self.lbl_result.configure(text="Lỗi: Vui lòng tải đủ 2 file!", text_color="red")
            return
            
        audio_score = self.audio_predictor.predict(self.audio_path)
        image_score = self.image_predictor.predict(self.image_path)
        
        final_score = self.fusion_engine.fuse(audio_score, image_score)
        
        label = "Abnormal" if final_score > 0.5 else "Normal"
        display_text = f"Kết quả: {label} ({final_score * 100:.1f}%)"
        self.lbl_result.configure(text=display_text, text_color="white")
        
        self.storage_manager.save_files(self.audio_path, self.image_path, label)

if __name__ == "__main__":
    app = DiagnosisApp()
    app.mainloop()