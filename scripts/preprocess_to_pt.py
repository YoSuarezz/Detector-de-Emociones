# scripts/preprocess_to_pt.py
"""
Lee items_all.json (output de prepare_datasets.py) y para cada item:
 - carga wav, recorta/pad a duration segundos
 - calcula mel-spectrogram (mel_db), normaliza
 - guarda tensor torch (.pt) en carpeta preprocessed/
 - escribe un nuevo JSON items_preprocessed.json con ruta al .pt y etiquetas numéricas

Uso:
python scripts/preprocess_to_pt.py --items items_all.json --out_dir data/preprocessed --sr 16000 --duration 3.0
"""
import os
import json
from pathlib import Path
import argparse
import torch
import numpy as np
import librosa

# label maps deben coincidir con las que usa el modelo
EMOTION_TO_IDX = {
    'neutral': 0,
    'calm': 1,
    'happy': 2,
    'sad': 3,
    'angry': 4,
    'fearful': 5,
    'disgust': 6,
    'surprised': 7
}
GENDER_TO_IDX = {'female': 0, 'male': 1, 'unknown': 0}

def load_audio(path, sr=16000):
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y.astype('float32')

def pad_or_trim(y, sr, duration=3.0):
    target_len = int(sr * duration)
    if len(y) < target_len:
        pad = target_len - len(y)
        y = np.pad(y, (0, pad), mode='constant')
    else:
        y = y[:target_len]
    return y

def wav_to_mel_tensor(y, sr=16000, n_mels=64, n_fft=1024, hop_length=512):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db -= mel_db.mean()
    mel_db /= (mel_db.std() + 1e-9)
    t = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0)  # (1, n_mels, time)
    return t

def main(items_json, out_dir, sr=16000, duration=3.0, n_mels=64):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    items = json.load(open(items_json, 'r', encoding='utf-8'))
    out_items = []
    for i, it in enumerate(items):
        wav_path = Path(it['path'])
        if not wav_path.exists():
            print("Missing:", wav_path)
            continue
        try:
            y = load_audio(str(wav_path), sr=sr)
            y = pad_or_trim(y, sr, duration=duration)
            mel_t = wav_to_mel_tensor(y, sr=sr, n_mels=n_mels)
            # save .pt
            base_name = f"item_{i:06d}.pt"
            save_path = out_dir / base_name
            torch.save(mel_t, save_path)
            # labels numeric
            emo_label = EMOTION_TO_IDX.get(it.get('emotion','neutral').lower(), 0)
            gen_label = GENDER_TO_IDX.get(it.get('gender','unknown').lower(), 0)
            out_items.append({
                "pt_path": str(save_path.resolve()),
                "emotion": int(emo_label),
                "gender": int(gen_label),
                "source": it.get("source","")
            })
            if (i+1) % 200 == 0:
                print(f"Processed {i+1} items")
        except Exception as e:
            print("Error processing", wav_path, e)
    out_json = out_dir / "items_preprocessed.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out_items, f, indent=2)
    print("Preprocessing finished. Saved", len(out_items), "items to", out_json)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", required=True, help="items_all.json path")
    parser.add_argument("--out_dir", default="data/preprocessed")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--n_mels", type=int, default=64)
    args = parser.parse_args()
    main(args.items, args.out_dir, sr=args.sr, duration=args.duration, n_mels=args.n_mels)
