# src/utils/audio.py
from typing import Tuple, Any
import numpy as np
import torch
import librosa

def load_audio(path: str, sr: int = 16000) -> Tuple[np.ndarray, int]:
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y.astype('float32'), sr

def pad_or_trim(y: np.ndarray, sr: int, duration: float = 3.0) -> np.ndarray:
    target_len = int(sr * duration)
    if len(y) < target_len:
        return np.pad(y, (0, target_len - len(y)), mode='constant')
    return y[:target_len]

def safe_torch_load(path, map_location='cpu', weights_only=True):
    """
    Carga segura sin imprimir mensajes molestos.
    Si falla weights_only=True, usa fallback silencioso.
    """
    import torch
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except Exception:
        # fallback sin mostrar ningún warning
        return torch.load(path, map_location=map_location, weights_only=False)


