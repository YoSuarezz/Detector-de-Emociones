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

def _ensure_min_time(mel_tensor, min_time=128):
    """
    mel_tensor: Tensor shape (1, n_mels, time) or (B,1,n_mels,time)
    Returns tensor with time dimension >= min_time (pad with zeros on right).
    """
    if mel_tensor.dim() == 3:
        # (1, n_mels, time)
        mel = mel_tensor
        pad_bs = False
    elif mel_tensor.dim() == 4:
        mel = mel_tensor.squeeze(0) if mel_tensor.size(0) == 1 else mel_tensor
        pad_bs = True
    else:
        raise ValueError("Unexpected mel dim: " + str(mel_tensor.shape))

    if mel.dim() == 3:
        # convert to (1,1,n_mels,time)
        mel = mel.unsqueeze(0)

    # mel now (B,1,n_mels,time)
    B, C, H, W = mel.shape
    if W < min_time:
        pad = min_time - W
        pad_tensor = (0, pad, 0, 0)  # pad last dim (W) on the right
        mel = torch.nn.functional.pad(mel, pad_tensor, mode='constant', value=0.0)
    return mel

def infer_single(model, wav_path, device, sr=16000, duration=3.0):
    # load audio
    y, _ = load_audio(wav_path, sr=sr)
    y = pad_or_trim(y, sr, duration=duration)
    mel = wav_to_mel_tensor(y, sr=sr)    # shape (1, n_mels, time)
    # ensure batch dim and min time for ResNet
    mel = mel.unsqueeze(0) if mel.dim() == 3 else mel  # (1,1,n_mels,time) expected later
    # now pad to safe minimum time to avoid shape mismatch in ResNet skip-connections
    mel = _ensure_min_time(mel, min_time=128)  # 128 is safe; adjust if needed

    # dtype & device
    mel = mel.to(dtype=torch.float32)

    # Try on provided device (DML). If it errors, fallback to CPU and retry.
    model.eval()
    try:
        mel_dev = mel.to(device, non_blocking=True)
        with torch.no_grad():
            e_logits, g_logits = model(mel_dev)
            e_probs = torch.nn.functional.softmax(e_logits, dim=1).cpu().squeeze(0)
            g_probs = torch.nn.functional.softmax(g_logits, dim=1).cpu().squeeze(0)
    except Exception as e:
        # fallback: report and retry on CPU
        print("[WARN] model forward failed on device", device, "->", repr(e))
        print("[WARN] Retrying forward on CPU...")
        cpu = torch.device('cpu')
        model_cpu = model.to(cpu)
        mel_cpu = mel.to(cpu)
        with torch.no_grad():
            e_logits, g_logits = model_cpu(mel_cpu)
            e_probs = torch.nn.functional.softmax(e_logits, dim=1).squeeze(0)
            g_probs = torch.nn.functional.softmax(g_logits, dim=1).squeeze(0)

    # top-k emotions (2)
    topk = torch.topk(e_probs, k=min(2, e_probs.numel()))
    emotions = [(EMOTION_INV[int(i.item())], float(p.item())) for p, i in zip(topk.values, topk.indices)]
    g_idx = int(g_probs.argmax().item())
    return emotions, (GENDER_INV[g_idx], float(g_probs[g_idx].item()))

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
