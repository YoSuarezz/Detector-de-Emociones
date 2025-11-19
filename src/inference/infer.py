# src/inference/infer.py
import argparse
from pathlib import Path
import logging
import os
import matplotlib
matplotlib.use("Agg")  # backend no interactivo, compatible con servidores


import numpy as np
import torch
import torch.nn.functional as F

# si tienes torch_directml instalado, úsalo; si no, caemos a CPU
try:
    import torch_directml
except Exception:
    torch_directml = None

from src.utils.audio import load_audio, pad_or_trim, safe_torch_load
from src.data.transforms import wav_to_mel_tensor
from src.models.model import SmallResNetMultiHead, SimpleCNN

# etiquetas inversas
EMOTION_INV = {
    0: 'neutral', 1: 'calm', 2: 'happy', 3: 'sad',
    4: 'angry', 5: 'fearful', 6: 'disgust', 7: 'surprised'
}
GENDER_INV = {0: 'female', 1: 'male'}

# logger (lo usamos para mensajes internos; la salida al usuario será limpia)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("infer")


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
        # (1, n_mels, time) -> convert to (1,1,n_mels,time)
        mel = mel_tensor.unsqueeze(0)
    elif mel_tensor.dim() == 4:
        mel = mel_tensor
    else:
        raise ValueError("Unexpected mel dim: " + str(mel_tensor.shape))

    # mel now (B,1,n_mels,time)
    B, C, H, W = mel.shape
    if W < min_time:
        pad = min_time - W
        pad_tensor = (0, pad, 0, 0)  # pad last dim (W) on the right
        mel = torch.nn.functional.pad(mel, pad_tensor, mode='constant', value=0.0)
    return mel


def infer_single(model, wav_path, device, sr=16000, duration=3.0):
    """
    Realiza toda la inferencia y devuelve:
      - top 2 emociones (label,prob)
      - gender (label,prob)
      - e_probs_np (numpy vector prob emo)
      - g_probs_np (numpy vector prob gen)
      - mel_cpu_np (numpy 2D mel for plotting) shape (n_mels, time)
      - waveform (numpy 1D)
    """
    # load audio (numpy float32)
    y, _ = load_audio(wav_path, sr=sr)
    y = pad_or_trim(y, sr, duration=duration)

    # mel tensor (1, n_mels, time) on CPU initially
    mel = wav_to_mel_tensor(y, sr=sr)    # shape (1, n_mels, time)
    mel = mel.unsqueeze(0) if mel.dim() == 3 else mel  # -> (1,1,n_mels,time)
    mel = _ensure_min_time(mel, min_time=128)
    mel = mel.to(dtype=torch.float32)

    # keep a CPU copy for plotting
    mel_cpu = mel.detach().cpu().squeeze()  # if shape (1,1,n_mels,time) -> squeeze -> (n_mels,time) or (1,n_mels,time)
    if mel_cpu.ndim == 3:
        # (1,n_mels,time) -> drop batch
        mel_cpu = mel_cpu.squeeze(0)
    mel_cpu_np = mel_cpu.numpy()

    model.eval()
    # Try forward on device, if fails retry on CPU
    try:
        mel_dev = mel.to(device, non_blocking=True)
        with torch.no_grad():
            e_logits, g_logits = model(mel_dev)
            e_probs = F.softmax(e_logits, dim=1).cpu().squeeze(0)
            g_probs = F.softmax(g_logits, dim=1).cpu().squeeze(0)
    except Exception as e:
        log.warning("[WARN] model forward failed on device %s -> %s", device, repr(e))
        cpu = torch.device('cpu')
        model_cpu = model.to(cpu)
        mel_cpu_t = mel.to(cpu)
        with torch.no_grad():
            e_logits, g_logits = model_cpu(mel_cpu_t)
            e_probs = F.softmax(e_logits, dim=1).squeeze(0)
            g_probs = F.softmax(g_logits, dim=1).squeeze(0)

    # top-2 emotions
    topk = torch.topk(e_probs, k=min(2, e_probs.numel()))
    emotions = [(EMOTION_INV[int(i.item())], float(p.item())) for p, i in zip(topk.values, topk.indices)]
    g_idx = int(g_probs.argmax().item())
    gender = (GENDER_INV[g_idx], float(g_probs[g_idx].item()))

    return emotions, gender, e_probs.cpu().numpy(), g_probs.cpu().numpy(), mel_cpu_np, y


def _ensure_output_dir():
    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    return out


def _save_combined_plot(wav_path: str, e_probs_np, g_probs_np, mel_np, waveform, sr=16000):
    """
    Guarda una figura combinada:
    - Waveform (fila 1 - ancho completo)
    - Espectrograma (fila 2, col 1)
    - Barras emociones (fila 2, col 2)
    - Barras género (fila 3, col 2)
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    outdir = _ensure_output_dir()
    stem = Path(wav_path).stem
    fname = outdir / f"infer_{stem}.png"

    # ***** FIGURE LAYOUT *****
    plt.figure(figsize=(14, 10))
    gs = GridSpec(nrows=3, ncols=2, height_ratios=[1.2, 2.5, 1.8], width_ratios=[3, 2])

    # --- (1) WAVEFORM (TOP, FULL WIDTH)
    ax_wave = plt.subplot(gs[0, :])
    times = np.arange(len(waveform)) / float(sr)
    ax_wave.plot(times, waveform, linewidth=0.7)
    ax_wave.set_title(f"Waveform: {Path(wav_path).name}", fontsize=12)
    ax_wave.set_xlabel("Time (s)")
    ax_wave.set_ylabel("Amplitude")
    ax_wave.grid(True, linewidth=0.3, alpha=0.5)

    # --- (2) MEL SPECTROGRAM (BOTTOM-LEFT)
    ax_mel = plt.subplot(gs[1:, 0])
    img = ax_mel.imshow(mel_np, aspect="auto", origin="lower", cmap="magma")
    ax_mel.set_title("Mel Spectrogram", fontsize=12)
    ax_mel.set_xlabel("Frames")
    ax_mel.set_ylabel("Mel bins")
    plt.colorbar(img, ax=ax_mel, shrink=0.7)

    # --- (3) EMOTION BARS (MIDDLE-RIGHT)
    ax_emo = plt.subplot(gs[1, 1])
    emo_labels = [EMOTION_INV[i] for i in range(len(e_probs_np))]
    x = np.arange(len(emo_labels))
    ax_emo.bar(x, e_probs_np, color="#4c72b0")
    ax_emo.set_xticks(x)
    ax_emo.set_xticklabels(emo_labels, rotation=45, ha="right")
    ax_emo.set_ylim(0, 1)
    ax_emo.set_ylabel("Probability")
    ax_emo.set_title("Emotion Probabilities")

    # --- (4) GENDER BARS (BOTTOM-RIGHT)
    ax_gen = plt.subplot(gs[2, 1])
    gen_labels = [GENDER_INV[i] for i in range(len(g_probs_np))]
    x2 = np.arange(len(gen_labels))
    ax_gen.bar(x2, g_probs_np, color="#dd8452")
    ax_gen.set_xticks(x2)
    ax_gen.set_xticklabels(gen_labels)
    ax_gen.set_ylim(0, 1)
    ax_gen.set_ylabel("Probability")
    ax_gen.set_title("Gender Probabilities")

    # Ajuste final sin errores
    plt.subplots_adjust(
        left=0.07, right=0.97, 
        top=0.94, bottom=0.08, 
        wspace=0.18, hspace=0.25
    )

    plt.savefig(str(fname), dpi=150)
    plt.close()
    return str(fname)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wav", required=True, help="Ruta al archivo WAV (ej: data/user_audio/prueba1.wav)")
    p.add_argument("--ckpt", default="model_final.pth", help="Checkpoint (state_dict)")
    p.add_argument("--no_plot", action="store_true", help="No guardar la imagen de salida")
    args = p.parse_args()

    # pick device
    if torch_directml is not None:
        try:
            device = torch_directml.device()
        except Exception as e:
            log.warning("torch_directml.device() failed -> %s. Falling back to CPU.", e)
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')

    # load model
    model = load_model(args.ckpt, device)

    # infer
    top_emotions, gender, e_probs_np, g_probs_np, mel_np, waveform = infer_single(model, args.wav, device)

    # salida limpia (solo estas 3 líneas)
    for i, (lab, pval) in enumerate(top_emotions, start=1):
        print(f"EMO{i}: {lab} ({pval:.3f})")
    print(f"GENDER: {gender[0]} ({gender[1]:.3f})")

    # guardar gráfica (si no se pidió --no_plot)
    if not args.no_plot:
        try:
            out_img = _save_combined_plot(args.wav, e_probs_np, g_probs_np, mel_np, waveform)
            log.info("Saved inference plot to %s", out_img)
        except Exception as e:
            log.warning("Could not save plot: %s", e)


if __name__ == "__main__":
    main()
