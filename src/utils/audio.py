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

def safe_torch_load(path: str, map_location: any = "cpu"):
    """
    Intenta cargar un checkpoint de forma segura:
      1) Primero intenta con weights_only=True (si la versión de torch lo soporta).
      2) Si eso falla (UnpicklingError u otro), vuelve a cargar con weights_only=False.
    Nota de seguridad: volver a cargar con weights_only=False puede ejecutar
    código arbitario contenido en el archivo. SOLO hacerlo si confías en el origen del archivo.
    """
    # Intento 1: weights_only=True (si la versión de torch lo acepta)
    try:
        # Algunas versiones de torch no aceptan el argumento; por eso lo probamos dentro del try.
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        # versión antigua de torch que no soporta weights_only -> fallback simple
        return torch.load(path, map_location=map_location)
    except Exception as e:
        # Si falla por seguridad/unpickling (es el caso que viste), intentamos el fallback.
        msg = str(e)
        print("[WARN] safe_torch_load: weights_only=True falló:", msg)
        print("[WARN] Intentando cargar con weights_only=False (fallback).")
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            # Si torch no acepta weights_only arg, reintentar simple
            return torch.load(path, map_location=map_location)

