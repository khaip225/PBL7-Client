class LateFusion:
    def __init__(self, audio_weight=0.6, image_weight=0.4):
        self.w_audio = audio_weight
        self.w_image = image_weight

    def fuse(self, audio_score, image_score):
        return (audio_score * self.w_audio) + (image_score * self.w_image)