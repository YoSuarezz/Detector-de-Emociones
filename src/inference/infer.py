# src/inference/infer.py
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
import torch_directml

from src.utils.audio import load_audio, pad_or_trim, safe_torch_load
from src.data.transforms import wav_to_mel_tensor
from src.models.model import SmallResNetMultiHead, SimpleCNN

EMOTION_INV = {
    0:'neutral',1:'calm',2:'happy',3:'sad',
    4:'angry',5:'fearful',6:'disgust',7:'surprised'
}
GENDER_INV = {0:'female',1:'male'}

def load_model(ckpt_path: str, device):
    state = safe_torch_load(ckpt_path, map_location='cpu')
    # try canonical model first
    for ModelCls in (SmallResNetMultiHead, SimpleCNN):
        model = ModelCls()
        try:
            model.load_state_dict(state)
            model.to(device)
            model.eval()
            return model
        except Exception:
            pass
    # try strip "module."
    if isinstance(state, dict):
        fixed = {k[len('module.'):]: v for k, v in state.items() if k.startswith('module.')}
        for ModelCls in (SmallResNetMultiHead, SimpleCNN):
            model = ModelCls()
            try:
                model.load_state_dict(fixed)
                model.to(device)
                model.eval()
                return model
            except Exception:
                pass
    raise RuntimeError("No se pudo cargar el checkpoint en los modelos conocidos.")

def infer_single(model, wav_path: str, device, sr=16000, duration=3.0):
    y, _ = load_audio(wav_path, sr=sr)
    y = pad_or_trim(y, sr, duration=duration)
    mel = wav_to_mel_tensor(y, sr=sr).unsqueeze(0).to(device)  # (1,1,n_mels,time)

    with torch.no_grad():
        e_logits, g_logits = model(mel)
        e_probs = F.softmax(e_logits, dim=1).cpu().squeeze(0)
        g_probs = F.softmax(g_logits, dim=1).cpu().squeeze(0)

    # top-2 emotions
    topk = torch.topk(e_probs, k=min(2, e_probs.numel()))
    top_emotions = [(EMOTION_INV[int(idx.item())], float(prob.item())) for prob, idx in zip(topk.values, topk.indices)]

    g_idx = int(g_probs.argmax().item())
    g_prob = float(g_probs[g_idx].item())
    gender = (GENDER_INV[g_idx], g_prob)

    return top_emotions, gender

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wav", required=True, help="Ruta al archivo WAV (ej: data/user_audio/prueba1.wav)")
    p.add_argument("--ckpt", default="model_final.pth", help="Checkpoint (state_dict) a cargar")
    args = p.parse_args()

    # device
    try:
        device = torch_directml.device()
    except Exception:
        device = torch.device('cpu')

    model = load_model(args.ckpt, device)
    top_emotions, gender = infer_single(model, args.wav, device)

    # Salida limpia (exactamente estas 3 líneas)
    # EMO1: label (prob)
    # EMO2: label (prob)
    # GENDER: label (prob)
    for i, (lab, pval) in enumerate(top_emotions, start=1):
        print(f"EMO{i}: {lab} ({pval:.3f})")
    print(f"GENDER: {gender[0]} ({gender[1]:.3f})")

if __name__ == "__main__":
    main()
