"""
Обучение видео-модели (EfficientNet-B0 + TCN).

Стратегия двухфазного обучения:
  Фаза 1 (frozen_epochs): backbone заморожен, обучаем только TCN + classifier.
  Фаза 2: разморозка backbone, обучение end-to-end с меньшим lr.

Запуск:
  python training/train_video.py --config configs/config.yaml
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
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datasets.ravdess import RAVDESSVideoDataset, IDX_TO_LABEL
from models.video.backbone import VideoEmotionModel

import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_loaders(cfg: dict):
    splits_dir = Path(cfg['data']['splits_dir'])
    proc_dir   = Path(cfg['data']['processed_dir']) / 'video'
    wf         = cfg['video']['window_frames']

    train_df = pd.read_csv(splits_dir / 'train.csv')
    val_df   = pd.read_csv(splits_dir / 'val.csv')

    # Только видео-файлы
    train_df = train_df[train_df['ext'].isin(['.mp4', '.flv'])].reset_index(drop=True)
    val_df   = val_df[val_df['ext'].isin(['.mp4', '.flv'])].reset_index(drop=True)

    train_ds = RAVDESSVideoDataset(train_df, proc_dir, wf, augment=True)
    val_ds   = RAVDESSVideoDataset(val_df,   proc_dir, wf, augment=False)

    bs = cfg['video']['train']['batch_size']
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

    for frames, labels in loader:
        frames, labels = frames.to(device), labels.to(device)
        logits = model(frames)
        loss = criterion(logits, labels)
        total_loss += loss.item() * len(labels)
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    n = len(all_labels)
    if n == 0:
        return 0.0, 0.0, 0.0
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return total_loss / n, acc, f1


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for frames, labels in tqdm(loader, leave=False):
        frames, labels = frames.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(frames)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(labels)
        all_preds.extend(logits.argmax(dim=1).detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    n = len(all_labels)
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

    tcn_cfg = cfg['video']['tcn']
    model = VideoEmotionModel(
        num_classes=cfg['emotions']['num_classes'],
        num_channels=tcn_cfg['num_channels'],
        num_levels=tcn_cfg['num_levels'],
        kernel_size=tcn_cfg['kernel_size'],
        dropout=tcn_cfg['dropout'],
        pretrained=cfg['video']['backbone_pretrained'],
        frozen_backbone=True,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    train_cfg = cfg['video']['train']
    frozen_epochs = cfg['video']['backbone_frozen_epochs']
    total_epochs  = train_cfg['epochs']

    best_f1 = 0.0
    ckpt_path = cfg['paths']['video_model_ckpt']

    # ── Фаза 1: backbone заморожен ──────────────────────────────────────
    print(f"\n=== Phase 1: backbone frozen ({frozen_epochs} epochs) ===")
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=train_cfg['lr'], weight_decay=train_cfg['weight_decay']
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=frozen_epochs)

    for epoch in range(1, frozen_epochs + 1):
        tr_loss, tr_acc, tr_f1 = train_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc, va_f1 = evaluate(model, val_loader, device, criterion)
        scheduler.step()

        print(f"[P1] Ep {epoch:02d}/{frozen_epochs} | "
              f"train loss={tr_loss:.4f} acc={tr_acc:.3f} f1={tr_f1:.3f} | "
              f"val loss={va_loss:.4f} acc={va_acc:.3f} f1={va_f1:.3f}")

        if va_f1 > best_f1:
            best_f1 = va_f1
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> Saved (f1={best_f1:.3f})")

    # ── Фаза 2: разморозка backbone ──────────────────────────────────────
    remaining = total_epochs - frozen_epochs
    print(f"\n=== Phase 2: full fine-tune ({remaining} epochs) ===")
    model.unfreeze_backbone()

    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg['lr'] * 0.1,   # маленький lr для backbone
        weight_decay=train_cfg['weight_decay']
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=remaining)

    for epoch in range(1, remaining + 1):
        tr_loss, tr_acc, tr_f1 = train_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc, va_f1 = evaluate(model, val_loader, device, criterion)
        scheduler.step()

        print(f"[P2] Ep {epoch:02d}/{remaining} | "
              f"train loss={tr_loss:.4f} acc={tr_acc:.3f} f1={tr_f1:.3f} | "
              f"val loss={va_loss:.4f} acc={va_acc:.3f} f1={va_f1:.3f}")

        if va_f1 > best_f1:
            best_f1 = va_f1
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> Saved (f1={best_f1:.3f})")

    print(f"\nBest val F1: {best_f1:.4f}  |  Checkpoint: {ckpt_path}")


if __name__ == '__main__':
    main()
