import numpy as np

class LateFusion:
    def __init__(self, audio_weight=0.4, image_weight=0.6):
        # Chuẩn hóa để đảm bảo tổng trọng số luôn = 1.0 (tránh lỗi do user nhập sai)
        total = audio_weight + image_weight
        self.w_audio = audio_weight / total
        self.w_image = image_weight / total

    def fuse(self, audio_score, image_score):
        """
        Gộp điểm số dựa trên độ tự tin (Adaptive Confidence Fusion).
        Trả về điểm cuối cùng và mức độ đóng góp (trọng số) của từng nhánh.
        """
        audio_conf = abs(audio_score - 0.5)
        image_conf = abs(image_score - 0.5)

        # Nếu cả 2 đều phân vân (score ~ 0.5), dùng trọng số tĩnh ban đầu
        if audio_conf < 0.01 and image_conf < 0.01:
            final_score = (audio_score * self.w_audio) + (image_score * self.w_image)
            return final_score, self.w_audio, self.w_image

        # Tính toán lại trọng số động
        total_conf = audio_conf + image_conf
        dyn_w_audio = audio_conf / total_conf
        dyn_w_image = image_conf / total_conf

        final_score = (audio_score * dyn_w_audio) + (image_score * dyn_w_image)
        
        return final_score, dyn_w_audio, dyn_w_image