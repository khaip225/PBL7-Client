"""Constants used across the AI engines."""

IMAGE_CLASS_NAMES = ["Normal", "Pneumonia", "COPD_Emphysema", "Fibrosis"]
IMAGE_DISEASE_NAMES = ["Pneumonia", "COPD_Emphysema", "Fibrosis"]
AUDIO_CLASS_NAMES = ["Normal", "Crackle", "Wheeze"]
AUDIO_ATTR_NAMES = ["Crackle", "Wheeze"]

# Prototype name to display name mapping
PROTO_DISPLAY = {
    "p_normal_img": "Phổi bình thường (Ảnh)",
    "p_pneumonia": "Viêm phổi",
    "p_emphysema": "COPD/Khí phế thũng",
    "p_fibrosis": "Xơ phổi",
    "p_normal_aud": "Âm thanh bình thường",
    "p_crackle": "Ran nổ (Crackle)",
    "p_wheeze": "Ran rít (Wheeze)",
}

# Prototype name mapping: checkpoint → internal
# Notebook uses p_emphysema, not p_copd
PROTO_NAME_MAP = {
    "p_normal_img": "p_normal_img",
    "p_normal_aud": "p_normal_aud",
    "p_pneumonia": "p_pneumonia",
    "p_emphysema": "p_emphysema",
    "p_fibrosis": "p_fibrosis",
    "p_crackle": "p_crackle",
    "p_wheeze": "p_wheeze",
}
