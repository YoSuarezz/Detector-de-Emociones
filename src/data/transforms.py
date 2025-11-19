# src/data/transforms.py
from typing import Tuple
import numpy as np
import torch
import librosa

def wav_to_mel_tensor(y: np.ndarray, sr: int = 16000,
                      n_mels: int = 64, n_fft: int = 1024, hop_length: int = 512) -> torch.Tensor:
    """
    Convierte una señal numpy (mono, float32) a tensor torch normalizado (1, n_mels, time).
    """
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db -= mel_db.mean()
    mel_db /= (mel_db.std() + 1e-9)
    return torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0)
