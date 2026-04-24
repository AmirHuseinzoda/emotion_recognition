"""
Обучение аудио-модели (HuBERT + MLP head).

Стратегия:
  Фаза 1: трансформерные слои заморожены, обучаем только MLP-голову.
  Фаза 2: размораживаем трансформер, full fine-tuning с lr=5e-5.

Запуск:
  python training/train_audio.py --config configs/config.yaml
"""

import argparse
import sys
import os
from pathlib import Path

import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datasets.ravdess import RAVDESSAudioDataset
from models.audio.transformer import AudioEmotionModel

import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_loaders(cfg: dict):
    splits_dir = Path(cfg['data']['splits_dir'])
    audio_cfg  = cfg['audio']

    train_df = pd.read_csv(splits_dir / 'train.csv')
    val_df   = pd.read_csv(splits_dir / 'val.csv')

    train_df = train_df[train_df['ext'] == '.wav'].reset_index(drop=True)
    val_df   = val_df[val_df['ext'] == '.wav'].reset_index(drop=True)

    train_ds = RAVDESSAudioDataset(
        train_df,
        target_sr=audio_cfg['sample_rate'],
        max_duration_sec=audio_cfg['max_duration_sec'],
        augment=True,
    )
    val_ds = RAVDESSAudioDataset(
        val_df,
        target_sr=audio_cfg['sample_rate'],
        max_duration_sec=audio_cfg['max_duration_sec'],
        augment=False,
    )

    bs = cfg['audio']['train']['batch_size']
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                              num_workers=4, pin_memory=True)
    return train_loader, val_loader


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for waveform, labels in loader:
        waveform, labels = waveform.to(device), labels.to(device)
        logits = model(waveform)
        loss   = criterion(logits, labels)
        total_loss += loss.item() * len(labels)
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    n   = len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    uar = accuracy_score(all_labels, all_preds, normalize=True)  # = UAR для balanced
    return total_loss / n, acc, f1


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for waveform, labels in tqdm(loader, leave=False):
        waveform, labels = waveform.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(waveform)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(labels)
        all_preds.extend(logits.argmax(dim=1).detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    n   = len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return total_loss / n, acc, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml')
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    os.makedirs('checkpoints', exist_ok=True)

    train_loader, val_loader = build_loaders(cfg)

    audio_cfg = cfg['audio']
    model = AudioEmotionModel(
        num_classes=cfg['emotions']['num_classes'],
        model_name=audio_cfg['model_name'],
        dropout=audio_cfg['dropout'],
        freeze_feature_encoder=audio_cfg['train']['freeze_feature_encoder'],
        freeze_transformer=True,   # Фаза 1
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    train_cfg = cfg['audio']['train']
    ckpt_path = cfg['paths']['audio_model_ckpt']
    best_f1   = 0.0

    # ── Фаза 1: только голова ────────────────────────────────────────────
    warmup    = train_cfg['warmup_epochs']
    phase1_ep = warmup + 5   # 5 эпох разогрева головы

    print(f"\n=== Phase 1: head only ({phase1_ep} epochs) ===")
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=train_cfg['lr'] * 10,   # для головы можно lr побольше
        weight_decay=train_cfg['weight_decay'],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=phase1_ep)

    for epoch in range(1, phase1_ep + 1):
        tr_loss, tr_acc, tr_f1 = train_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc, va_f1 = evaluate(model, val_loader, device, criterion)
        scheduler.step()
        print(f"[P1] Ep {epoch:02d} | train f1={tr_f1:.3f} | val f1={va_f1:.3f} acc={va_acc:.3f}")
        if va_f1 > best_f1:
            best_f1 = va_f1
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> Saved (f1={best_f1:.3f})")

    # ── Фаза 2: full fine-tuning ─────────────────────────────────────────
    remaining = train_cfg['epochs'] - phase1_ep
    print(f"\n=== Phase 2: full fine-tune ({remaining} epochs) ===")
    model.unfreeze_transformer()

    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg['lr'],
        weight_decay=train_cfg['weight_decay'],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=remaining)

    for epoch in range(1, remaining + 1):
        tr_loss, tr_acc, tr_f1 = train_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc, va_f1 = evaluate(model, val_loader, device, criterion)
        scheduler.step()
        print(f"[P2] Ep {epoch:02d} | train f1={tr_f1:.3f} | val f1={va_f1:.3f} acc={va_acc:.3f}")
        if va_f1 > best_f1:
            best_f1 = va_f1
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> Saved (f1={best_f1:.3f})")

    print(f"\nBest val F1: {best_f1:.4f}  |  Checkpoint: {ckpt_path}")


if __name__ == '__main__':
    main()
