import os
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class AudioPneumoniaDataset(Dataset):
    def __init__(self, client_id, base_dir=r"C:\PBL7_Data", max_length=15, target_sr=16000, n_mels=64, is_train=True):
        self.audio_dir = os.path.join(base_dir, f"Client_{client_id}", "audio_files")
        self.files = [f for f in os.listdir(self.audio_dir) if f.endswith('.wav')]
        
        if len(self.files) < 2:
            raise ValueError(f"Client {client_id} không đủ dữ liệu (ít nhất 2 file).")

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
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        file_path = os.path.join(self.audio_dir, file_name)
        
        # Bắt nhãn từ tên file sinh ra ở Giai đoạn 1
        label = 1.0 if file_name.startswith("Abnormal") else 0.0

        waveform, sr = torchaudio.load(file_path)

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

def load_client_data(client_id, batch_size=16):
    # Dùng chung class, chia train/val
    full_dataset = AudioPneumoniaDataset(client_id, is_train=True)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_set, val_set = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # Ép val_set không dùng Augmentation (Masking)
    val_set.dataset.is_train = False 
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    return train_loader, val_loader