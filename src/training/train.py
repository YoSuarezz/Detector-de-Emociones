# src/training/train.py
import argparse, json, time, os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch_directml
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.data.dataset import VoiceEmotionGenderDataset
from src.models.model import SmallResNetMultiHead, SimpleCNN

def get_device():
    try:
        d = torch_directml.device()
        print("✔ Using DirectML device:", d)
        return d
    except Exception:
        print("⚠ torch_directml not available, falling back to CPU")
        return torch.device('cpu')

def train(items_json, epochs=3, batch_size=8, lr=1e-2, preload=False, accumulate=1, use_simple=False):
    device = get_device()

    items = json.load(open(items_json, 'r', encoding='utf-8'))
    ds = VoiceEmotionGenderDataset(items, mode="pt", preload=preload)

    n = len(ds)
    train_n = int(n*0.8); val_n = n - train_n
    train_ds, val_ds = random_split(ds, [train_n, val_n])

    # DirectML: use 0 workers to avoid issues; pin_memory False for DML
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    model = SimpleCNN() if use_simple else SmallResNetMultiHead()
    model.to(device)

    # Use SGD recommended for DirectML; but keep option to use Adam if desired
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    acc_steps = max(1, int(accumulate))
    print(f"[INFO] epochs={epochs} batch={batch_size} preload={preload} accumulate={acc_steps}")

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        optimizer.zero_grad()
        running_loss = 0.0
        step = 0
        t0 = time.time()
        for i, (mel, emo, gen) in enumerate(pbar):
            mel = mel.to(device)
            emo = emo.to(device)
            gen = gen.to(device)

            e_logits, g_logits = model(mel)
            loss = criterion(e_logits, emo) + 0.5 * criterion(g_logits, gen)
            loss = loss / acc_steps
            loss.backward()

            if (i + 1) % acc_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                step += 1

            running_loss += loss.item() * acc_steps
            pbar.set_postfix(loss=running_loss / (i+1))

        t1 = time.time()
        print(f"[INFO] Epoch {epoch+1} training time: {t1-t0:.1f}s")

        # validation
        model.eval()
        total = 0; ce = 0; cg = 0
        with torch.no_grad():
            for mel, emo, gen in val_loader:
                mel = mel.to(device)
                emo = emo.to(device)
                gen = gen.to(device)
                e_logits, g_logits = model(mel)
                pe = e_logits.argmax(dim=1); pg = g_logits.argmax(dim=1)
                total += emo.shape[0]
                ce += (pe == emo).sum().item()
                cg += (pg == gen).sum().item()
        print(f"[INFO] Epoch {epoch+1} val emo acc: {ce/total:.3f} gen acc: {cg/total:.3f}")

        torch.save(model.state_dict(), f"model_epoch{epoch+1}.pth")

    torch.save(model.state_dict(), "model_final.pth")
    print("[INFO] saved model_final.pth")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--items", required=True, help="data/preprocessed/items_preprocessed.json")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--preload", action="store_true", help="cargar todos los .pt en RAM")
    p.add_argument("--accumulate", type=int, default=1, help="acumulacion de gradiente")
    p.add_argument("--use_simple", action="store_true", help="usar modelo SimpleCNN (debug)")
    args = p.parse_args()

    train(args.items, epochs=args.epochs, batch_size=args.batch_size,
          lr=args.lr, preload=args.preload, accumulate=args.accumulate, use_simple=args.use_simple)
