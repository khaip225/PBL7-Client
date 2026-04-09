import os
import torch
import torchaudio
import soundfile as sf
import torch.nn.functional as F
import pandas as pd
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

def load_client_data(client_id, batch_size=16, base_dir=r"C:\Users\phant\Downloads\PBL7"):
    audio_dir = os.path.join(base_dir, "fl_lung_data_audio", "lung_data_audio_processed")
    
    train_csv = os.path.join(base_dir, "metadata", "audio_fl", f"client_{client_id}_train.csv")
    val_csv = os.path.join(base_dir, "metadata", "audio_fl", f"client_{client_id}_val.csv")

    train_set = AudioPneumoniaDataset(train_csv, audio_dir, is_train=True)
    val_set = AudioPneumoniaDataset(val_csv, audio_dir, is_train=False)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader