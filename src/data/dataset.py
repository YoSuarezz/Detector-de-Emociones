# src/data/dataset.py
from torch.utils.data import Dataset
from typing import List, Tuple, Optional
import json, os
import torch
from pathlib import Path
from src.utils.audio import load_audio, pad_or_trim
from src.data.transforms import wav_to_mel_tensor

EMOTION_MAP = {
    'neutral': 0, 'calm': 1, 'happy': 2, 'sad': 3,
    'angry': 4, 'fearful':5, 'disgust':6, 'surprised':7
}
GENDER_MAP = {'female': 0, 'male': 1, 'unknown': 0}

class VoiceEmotionGenderDataset(Dataset):
    def __init__(self, items: List[dict], mode: str = "pt", preload: bool = False,
                 sr: int = 16000, duration: float = 3.0):
        """
        mode: "pt" (preprocessed .pt files) or "wav" (raw wav)
        preload: si mode=="pt", carga todos los tensors en RAM
        """
        self.items = items
        self.mode = mode
        self.preload = preload and mode == "pt"
        self.sr = sr
        self.duration = duration
        if self.preload:
            self._tensors = []
            self._labels = []
            for it in self.items:
                t = torch.load(it['pt_path'])
                self._tensors.append(t)
                self._labels.append((int(it['emotion']), int(it['gender'])))
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        it = self.items[idx]
        if self.preload:
            t = self._tensors[idx]
            emo, gen = self._labels[idx]
            return t, emo, gen
        if self.mode == "pt":
            t = torch.load(it['pt_path'])
            return t, int(it['emotion']), int(it['gender'])
        else:
            wav, sr = load_audio(it['path'], sr=self.sr)
            wav = pad_or_trim(wav, sr, self.duration)
            t = wav_to_mel_tensor(wav, sr=self.sr)
            return t, EMOTION_MAP.get(it['emotion'].lower(), 0), GENDER_MAP.get(it['gender'].lower(), 0)
