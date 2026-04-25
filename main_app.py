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
        self.geometry("700x700")
        
        self.audio_predictor = AudioPredictor("ai_engines/current_weights/best_global_audio.pth")
        self.image_predictor = ImagePredictor("ai_engines/current_weights/best_global_image.pth")
        self.fusion_engine = LateFusion(audio_weight=0.6, image_weight=0.4)
        self.storage_manager = StorageManager(client_id=1)
        
        self.audio_path = None
        self.image_path = None
        
        # Mode selection
        self.lbl_mode = ctk.CTkLabel(self, text="Chọn chế độ chẩn đoán:", font=("Arial", 14, "bold"))
        self.lbl_mode.pack(pady=10)
        
        self.mode_var = ctk.StringVar(value="fusion")
        self.radio_fusion = ctk.CTkRadioButton(self, text="Chế độ Fusion (Âm thanh + X-quang)", variable=self.mode_var, value="fusion", command=self.update_mode)
        self.radio_fusion.pack(pady=5)
        
        self.radio_audio = ctk.CTkRadioButton(self, text="Chỉ Âm thanh", variable=self.mode_var, value="audio", command=self.update_mode)
        self.radio_audio.pack(pady=5)
        
        self.radio_image = ctk.CTkRadioButton(self, text="Chỉ X-quang", variable=self.mode_var, value="image", command=self.update_mode)
        self.radio_image.pack(pady=5)
        
        # Separators
        self.separator1 = ctk.CTkLabel(self, text="─" * 50)
        self.separator1.pack(pady=15)
        
        # File loading section
        self.btn_load_audio = ctk.CTkButton(self, text="Tải file Âm thanh (.wav)", command=self.load_audio)
        self.btn_load_audio.pack(pady=10)
        
        self.lbl_audio = ctk.CTkLabel(self, text="Chưa có file âm thanh", text_color="gray")
        self.lbl_audio.pack()
        
        self.btn_load_image = ctk.CTkButton(self, text="Tải file X-quang (.png/.jpg)", command=self.load_image)
        self.btn_load_image.pack(pady=10)
        
        self.lbl_image = ctk.CTkLabel(self, text="Chưa có file X-quang", text_color="gray")
        self.lbl_image.pack()
        
        # Separator
        self.separator2 = ctk.CTkLabel(self, text="─" * 50)
        self.separator2.pack(pady=15)
        
        self.btn_predict = ctk.CTkButton(self, text="CHẨN ĐOÁN & LƯU TRỮ", command=self.process_diagnosis, fg_color="green", height=40)
        self.btn_predict.pack(pady=20)
        
        self.lbl_result = ctk.CTkLabel(self, text="Kết quả: ...", font=("Arial", 18, "bold"))
        self.lbl_result.pack(pady=10)

    def load_audio(self):
        self.audio_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])
        if self.audio_path:
            self.lbl_audio.configure(text=self.audio_path, text_color="white")

    def load_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if self.image_path:
            self.lbl_image.configure(text=self.image_path, text_color="white")
    
    def update_mode(self):
        """Update UI based on selected mode"""
        mode = self.mode_var.get()
        if mode == "fusion":
            self.btn_load_audio.configure(state="normal")
            self.btn_load_image.configure(state="normal")
        elif mode == "audio":
            self.btn_load_audio.configure(state="normal")
            self.btn_load_image.configure(state="disabled")
            self.image_path = None
            self.lbl_image.configure(text="Chưa có file X-quang", text_color="gray")
        else:  # image mode
            self.btn_load_audio.configure(state="disabled")
            self.btn_load_image.configure(state="normal")
            self.audio_path = None
            self.lbl_audio.configure(text="Chưa có file âm thanh", text_color="gray")

    def process_diagnosis(self):
        mode = self.mode_var.get()
        
        if mode == "fusion":
            if not self.audio_path or not self.image_path:
                self.lbl_result.configure(text="Lỗi: Vui lòng tải đủ 2 file!", text_color="red")
                return
                
            audio_score = self.audio_predictor.predict(self.audio_path)
            image_score = self.image_predictor.predict(self.image_path)
            
            final_score = self.fusion_engine.fuse(audio_score, image_score)
            
            label = "Abnormal" if final_score > 0.5 else "Normal"
            # Calculate confidence for the predicted label
            if label == "Normal":
                confidence = (1 - final_score) * 100
            else:
                confidence = final_score * 100
            display_text = f"Kết quả (Fusion): {label} ({confidence:.1f}%)"
            self.lbl_result.configure(text=display_text, text_color="white")
            
            self.storage_manager.save_files(self.audio_path, self.image_path, label)
        
        elif mode == "audio":
            if not self.audio_path:
                self.lbl_result.configure(text="Lỗi: Vui lòng tải file âm thanh!", text_color="red")
                return
            
            audio_score = self.audio_predictor.predict(self.audio_path)
            
            label = "Abnormal" if audio_score > 0.5 else "Normal"
            # Calculate confidence for the predicted label
            if label == "Normal":
                confidence = (1 - audio_score) * 100
            else:
                confidence = audio_score * 100
            display_text = f"Kết quả (Âm thanh): {label} ({confidence:.1f}%)"
            self.lbl_result.configure(text=display_text, text_color="white")
            
            self.storage_manager.save_files(self.audio_path, None, label)
        
        else:  # image mode
            if not self.image_path:
                self.lbl_result.configure(text="Lỗi: Vui lòng tải file X-quang!", text_color="red")
                return
            
            image_score = self.image_predictor.predict(self.image_path)
            
            label = "Abnormal" if image_score > 0.5 else "Normal"
            # Calculate confidence for the predicted label
            if label == "Normal":
                confidence = (1 - image_score) * 100
            else:
                confidence = image_score * 100
            display_text = f"Kết quả (X-quang): {label} ({confidence:.1f}%)"
            self.lbl_result.configure(text=display_text, text_color="white")
            
            self.storage_manager.save_files(None, self.image_path, label)

if __name__ == "__main__":
    app = DiagnosisApp()
    app.mainloop()