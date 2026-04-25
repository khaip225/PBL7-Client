import os
import torch
import torchaudio
import soundfile as sf
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class AudioPneumoniaDataset(Dataset):
    def __init__(self, csv_file, audio_dir, max_length=15, target_sr=16000, n_mels=64, is_train=True):
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"File Metadata missing: {csv_file}")
            
        self.data = pd.read_csv(csv_file)
        self.audio_dir = audio_dir
        self.max_length = max_length
        self.target_sr = target_sr
        self.is_train = is_train

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sr, n_fft=1024, hop_length=512, n_mels=n_mels
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=8)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=20)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        wav_path = self.data.iloc[idx]["path"]
        label = int(self.data.iloc[idx]["label"])

        filename = os.path.basename(wav_path.replace('\\', '/'))
        real_path = os.path.join(self.audio_dir, filename)

        if not os.path.exists(real_path):
             raise FileNotFoundError(f"Audio file missing: {real_path}")

        waveform_np, sr = sf.read(real_path)
        waveform = torch.from_numpy(waveform_np).float()
        
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.ndim == 2:
            waveform = waveform.transpose(0, 1)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
            waveform = resampler(waveform)

        target_samples = self.target_sr * self.max_length
        current_samples = waveform.shape[1]

        if current_samples < target_samples:
            padding = target_samples - current_samples
            waveform = F.pad(waveform, (0, padding))
        elif current_samples > target_samples:
            waveform = waveform[:, :target_samples]

        mel_spec = self.mel_spectrogram(waveform)
        mel_spec_db = self.amplitude_to_db(mel_spec)

        mean = mel_spec_db.mean()
        std = mel_spec_db.std()
        mel_spec_db = (mel_spec_db - mean) / (std + 1e-6)

        if self.is_train:
            mel_spec_db = self.freq_mask(mel_spec_db)
            mel_spec_db = self.time_mask(mel_spec_db)

        return mel_spec_db, torch.tensor(label, dtype=torch.float32)

def load_client_data(client_id, batch_size=16, base_dir="./fl_data"):
    audio_dir = os.path.join(base_dir, "fl_audio")
    
    train_csv = os.path.join(base_dir, "metadata", "audio_fl", f"client_{client_id}_train.csv")
    val_csv = os.path.join(base_dir, "metadata", "audio_fl", f"client_{client_id}_val.csv")

    train_set = AudioPneumoniaDataset(train_csv, audio_dir, is_train=True)
    val_set = AudioPneumoniaDataset(val_csv, audio_dir, is_train=False)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


class ImagePneumoniaDataset(Dataset):
    def __init__(self, data_dir, is_train=True):
        self.data_dir = data_dir
        self.is_train = is_train
        
        # Tự động quét file trong 2 thư mục NORMAL và PNEUMONIA
        self.data = []
        normal_dir = os.path.join(data_dir, "NORMAL")
        pneumonia_dir = os.path.join(data_dir, "PNEUMONIA")
        
        if os.path.exists(normal_dir):
            for f in os.listdir(normal_dir):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.data.append((os.path.join(normal_dir, f), 0))
                    
        if os.path.exists(pneumonia_dir):
            for f in os.listdir(pneumonia_dir):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.data.append((os.path.join(pneumonia_dir, f), 1))
                    
        if len(self.data) == 0:
            print(f"[Cảnh báo] Không tìm thấy ảnh nào trong {normal_dir} hoặc {pneumonia_dir}")

        if self.is_train:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)

def load_client_data_image(client_id, batch_size=16, base_dir="./fl_data"):
    # Cấu trúc thư mục mới: fl_data/fl_image/client_1/train/NORMAL ...
    client_dir = os.path.join(base_dir, "fl_image", f"client_{client_id}")
    
    # Nếu chưa chia thư mục client riêng, lấy xài chung thư mục tổng để demo
    if not os.path.exists(client_dir):
        client_dir = os.path.join(base_dir, "fl_image")
        
    train_dir = os.path.join(client_dir, "train")
    val_dir = os.path.join(client_dir, "test") # Hoặc val
    if not os.path.exists(val_dir): val_dir = os.path.join(client_dir, "val")
    
    # Nếu không có file train/test bên trong, quét thẳng thư mục hiện tại
    if not os.path.exists(train_dir): train_dir = client_dir
    if not os.path.exists(val_dir): val_dir = client_dir

    train_set = ImagePneumoniaDataset(train_dir, is_train=True)
    val_set = ImagePneumoniaDataset(val_dir, is_train=False)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader