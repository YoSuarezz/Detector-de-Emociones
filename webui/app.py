# webui/app.py
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from pathlib import Path
import subprocess
import sys
import time

BASE_DIR = Path(__file__).resolve().parents[1]  # repo root
WEBUI_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "webui_uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
ALLOWED_EXT = {'.wav', '.mp3'}

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, static_folder=str(WEBUI_DIR / 'static'), template_folder=str(WEBUI_DIR / 'templates'))


def allowed_file(name: str):
    return Path(name).suffix.lower() in ALLOWED_EXT


# Intento de import interno de infer para mejor UX (devuelve vectores)
INTERNAL_INFER_AVAILABLE = False
try:
    sys.path.insert(0, str(BASE_DIR))
    # import names only (no side effects expected)
    from src.inference.infer import infer_single, load_model, _save_combined_plot
    INTERNAL_INFER_AVAILABLE = True
except Exception:
    INTERNAL_INFER_AVAILABLE = False


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'wavfile' not in request.files:
        return jsonify({'ok': False, 'error': 'No file part'})
    f = request.files['wavfile']
    if f.filename == '':
        return jsonify({'ok': False, 'error': 'No selected file'})
    if not allowed_file(f.filename):
        return jsonify({'ok': False, 'error': 'Unsupported file type'})

    fname = secure_filename(f.filename)
    save_path = UPLOAD_DIR / fname
    f.save(str(save_path))

    ckpt = request.form.get('ckpt', 'model_final.pth')
    ckpt_path = str((BASE_DIR / ckpt).resolve())

    result = {
        'ok': False,
        'stdout': '',
        'stderr': '',
        'image': None,
        'emo_probs': None,
        'gen_probs': None,
        'top_emotions': None,
        'top_gender': None,
    }

    stem = Path(save_path).stem
    candidate_img = OUTPUT_DIR / f"infer_{stem}.png"

    # --- Try internal inference (returns numpy arrays) ---
    if INTERNAL_INFER_AVAILABLE:
        try:
            import torch
            try:
                import torch_directml
                device = torch_directml.device()
            except Exception:
                device = torch.device('cpu')
        except Exception:
            device = None

        model = None
        try:
            # load model with device (if load_model expects device)
            if device is not None:
                try:
                    model = load_model(ckpt_path, device)
                except Exception:
                    # fallback to cpu
                    try:
                        model = load_model(ckpt_path, torch.device('cpu'))
                    except Exception:
                        model = None
            else:
                # try without device param (in case implementation differs)
                try:
                    model = load_model(ckpt_path, torch.device('cpu'))
                except Exception:
                    model = None
        except Exception:
            model = None

        if model is not None:
            try:
                out = infer_single(model, str(save_path), device)
                # infer_single -> (top_emotions, gender, e_probs_np, g_probs_np, mel_np, waveform)
                top_emotions, gender, e_probs_np, g_probs_np, mel_np, waveform = out

                # try save image via module helper
                try:
                    imgpath = _save_combined_plot(str(save_path), e_probs_np, g_probs_np, mel_np, waveform)
                    candidate_img = Path(imgpath)
                except Exception:
                    candidate_img = OUTPUT_DIR / f"infer_{stem}.png"

                result.update({
                    'ok': True,
                    'stdout': "\n".join([f"EMO{i+1}: {lab} ({p:.3f})" for i, (lab, p) in enumerate(top_emotions)]),
                    'stderr': '',
                    'image': f'/outputs/{candidate_img.name}' if candidate_img.exists() else None,
                    'emo_probs': e_probs_np.tolist() if hasattr(e_probs_np, 'tolist') else None,
                    'gen_probs': g_probs_np.tolist() if hasattr(g_probs_np, 'tolist') else None,
                    'top_emotions': [(t[0], float(t[1])) for t in top_emotions],
                    'top_gender': (gender[0], float(gender[1]))
                })
                return jsonify(result)
            except Exception as e:
                # record error and fallback to subprocess
                result['stderr'] = f'Internal infer failed: {e}'

    # --- Fallback: run original script via subprocess ---
    python_exe = sys.executable
    cmd = [python_exe, '-m', 'src.inference.infer', '--wav', str(save_path), '--ckpt', ckpt_path]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=str(BASE_DIR), timeout=120)
        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()
        result['ok'] = True
        result['stdout'] = stdout
        result['stderr'] = stderr
    except subprocess.TimeoutExpired:
        return jsonify({'ok': False, 'error': 'Timeout during inference'})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})

    if candidate_img.exists():
        result['image'] = f'/outputs/{candidate_img.name}'
    return jsonify(result)


@app.route('/outputs/<path:filename>')
def outputs(filename):
    return send_from_directory(str(OUTPUT_DIR), filename)


if __name__ == '__main__':
    # Evitamos que Flask reinicie el proceso al detectar cambios en archivos
    # (útil porque la carpeta outputs/ cambia cuando se genera la imagen).
    app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=False)
