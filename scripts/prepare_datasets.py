# scripts/prepare_datasets.py
"""
Prepara items list (items_all.json) a partir de directorios con RAVDESS y CREMA-D.
Salida: items_all.json con lista de dicts:
  { "path": "/abs/path/to.wav", "emotion": "happy", "gender": "male" }

Notas:
- RAVDESS: estructura típica: data/RAVDESS/Actor_01/03-01-01-01-01-01-01.wav
  Filename fields (split by '-') -> emotion code en campo index 2 (0-based).
  emotion_code_map: 1:neutral,2:calm,3:happy,4:sad,5:angry,6:fearful,7:disgust,8:surprised
  Actor folders Actor_01..Actor_24; asumimos Actor 01-12 = male, 13-24 = female.
- CREMA-D: filenames estilo 1001_DFA_ANG_XX.wav o 1025_IEO_HAP_XX.wav
  Se parsea por '_' y se usa el campo emotion abreviado (ANG, HAP, SAD, etc.)
  gender inference: CREMA-D actor ids (1001..): actor id parity -> male/female not fiable,
  por eso intentamos leer actor id (first 4 digits) y usar a mapping minimal:
    - si actor id <= 1100: male else female (heurística simple).
  Si prefieres, edita el items_all.json después para corregir géneros.
"""
import os
import json
from pathlib import Path

EMOTION_MAP_RAV = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

EMOTION_MAP_CREMA = {
    'ANG': 'angry',
    'DIS': 'disgust',
    'FEA': 'fearful',
    'HAP': 'happy',
    'NEU': 'neutral',
    'SAD': 'sad'
}

def parse_ravdess(root_dir):
    """
    root_dir: path to folder containing Actor_* subfolders for RAVDESS
    """
    items = []
    root = Path(root_dir)
    if not root.exists():
        print("RAVDESS root not found:", root_dir)
        return items

    for actor_dir in sorted(root.glob("Actor_*")):
        actor_name = actor_dir.name  # Actor_01
        try:
            actor_num = int(actor_name.split('_')[1])
        except Exception:
            actor_num = None
        # Heuristic: actors 1-12 male, 13-24 female (common RAVDESS layout)
        if actor_num is not None:
            gender = 'male' if actor_num <= 12 else 'female'
        else:
            gender = 'unknown'

        for wav in actor_dir.rglob("*.wav"):
            fname = wav.name  # e.g. 03-01-01-01-01-01-01.wav
            parts = fname.split('-')
            if len(parts) < 3:
                continue
            emo_code = parts[2]
            emo = EMOTION_MAP_RAV.get(emo_code, 'neutral')
            items.append({
                "path": str(wav.resolve()),
                "emotion": emo,
                "gender": gender,
                "source": "RAVDESS"
            })
    print(f"RAVDESS parsed: {len(items)} items")
    return items

def parse_cremad(root_dir):
    """
    root_dir: path to CREMA-D wavs (all wav files in folder)
    CREMA-D filename examples:
      1001_DFA_ANG_XX.wav
      1025_IEO_HAP_XX.wav
    We extract emotion from the 3-letter code (field index 2 when split by '_')
    and actor id from first 4 digits for a heuristic gender.
    """
    items = []
    root = Path(root_dir)
    if not root.exists():
        print("CREMA-D root not found:", root_dir)
        return items

    for wav in root.rglob("*.wav"):
        fname = wav.name
        parts = fname.split('_')
        emo = 'neutral'
        gender = 'unknown'
        if len(parts) >= 3:
            emo_code = parts[2]
            emo = EMOTION_MAP_CREMA.get(emo_code, 'neutral')
        # actor id heuristic
        try:
            aid = int(parts[0][:4])  # first section like '1001'
            # heuristic: many actor ids below 1100 are male in some distributions; not guaranteed
            gender = 'male' if aid < 1100 else 'female'
        except Exception:
            gender = 'unknown'
        items.append({
            "path": str(wav.resolve()),
            "emotion": emo,
            "gender": gender,
            "source": "CREMA-D"
        })
    print(f"CREMA-D parsed: {len(items)} items")
    return items

def main(ravdess_dir, cremad_dir, out_json="items_all.json"):
    all_items = []
    all_items += parse_ravdess(ravdess_dir)
    all_items += parse_cremad(cremad_dir)
    # optionally shuffle
    # import random; random.shuffle(all_items)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_items, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(all_items)} items to {out_json}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ravdess", required=True, help="Path to RAVDESS root (containing Actor_* folders)")
    parser.add_argument("--cremad", required=True, help="Path to CREMA-D wav root")
    parser.add_argument("--out", default="items_all.json")
    args = parser.parse_args()
    main(args.ravdess, args.cremad, args.out)
