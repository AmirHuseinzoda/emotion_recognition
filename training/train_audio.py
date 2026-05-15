"""
Обучение аудио-модели (WavLM-Large + AttentiveMeanPool + MLP head).

Стратегия:
  Фаза 1: трансформерные слои заморожены, обучаем только голову.
  Фаза 2: размораживаем трансформер, full fine-tuning с малым lr.

Данные: CREMA-D .wav + RAVDESS .wav (оба источника, actor-independent split).

Улучшения vs предыдущей версии:
  - WavLM-Large (microsoft/wavlm-large) вместо HuBERT-base
  - AttentiveMeanPool вместо простого среднего
  - Включён RAVDESS аудио (+~18% данных, больше разнообразия спикеров)
  - Mixup augmentation (50% вероятность применения к батчу)
  - AMP (torch.cuda.amp) — экономит память для больших моделей

Запуск:
  python training/train_audio.py --config configs/config.yaml
"""

import argparse
import sys
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, accuracy_score, recall_score, confusion_matrix
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datasets.ravdess import RAVDESSAudioDataset, CLASS_NAMES
from models.audio.transformer import AudioEmotionModel
from training.utils import FocalLoss, EarlyStopping, make_weighted_sampler, Mixup, mixup_loss

import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_loaders(cfg: dict):
    splits_dir  = Path(cfg['data']['splits_dir'])
    audio_cfg   = cfg['audio']
    train_cfg   = audio_cfg['train']
    num_classes = cfg['emotions']['num_classes']

    train_df = pd.read_csv(splits_dir / 'train.csv')
    val_df   = pd.read_csv(splits_dir / 'val.csv')

    # Оба источника: CREMA-D .wav + RAVDESS .wav
    train_df = train_df[train_df['ext'] == '.wav'].reset_index(drop=True)
    val_df   = val_df[val_df['ext'] == '.wav'].reset_index(drop=True)

    for split_name, df in [('train', train_df), ('val', val_df)]:
        total = len(df)
        if 'source' in df.columns:
            src_counts = df['source'].value_counts().to_dict()
            print(f'  [{split_name}] {total} audio samples: {src_counts}')
        else:
            print(f'  [{split_name}] {total} audio samples')

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

    bs = train_cfg['batch_size']

    if train_cfg.get('weighted_sampler', True):
        sampler = make_weighted_sampler(train_df['label'].values, num_classes)
        train_loader = DataLoader(train_ds, batch_size=bs, sampler=sampler,
                                  num_workers=4, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                                  num_workers=4, pin_memory=True)

    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            num_workers=4, pin_memory=True)
    return train_loader, val_loader


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for waveform, labels in loader:
        waveform, labels = waveform.to(device), labels.to(device)
        with torch.cuda.amp.autocast():
            logits = model(waveform)
            loss   = criterion(logits, labels)
        total_loss += loss.item() * len(labels)
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    n   = len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    uar = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    return total_loss / n, acc, f1, uar


def train_epoch(model, loader, optimizer, criterion, device, scaler, mixup: Mixup = None):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    mixup_prob = 0.5  # применяем mixup к 50% батчей

    for waveform, labels in tqdm(loader, leave=False):
        waveform, labels = waveform.to(device), labels.to(device)
        optimizer.zero_grad()

        use_mixup = mixup is not None and torch.rand(1).item() < mixup_prob
        if use_mixup:
            waveform, y_a, y_b, lam = mixup.mix(waveform, labels)

        with torch.cuda.amp.autocast():
            logits = model(waveform)
            if use_mixup:
                loss = mixup_loss(criterion, logits, y_a, y_b, lam)
            else:
                loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * len(labels)
        all_preds.extend(logits.argmax(dim=1).detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    n   = len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    uar = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    return total_loss / n, acc, f1, uar


def print_confusion_matrix(model, loader, device, num_classes):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for waveform, labels in loader:
            waveform = waveform.to(device)
            with torch.cuda.amp.autocast():
                preds = model(waveform).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    names = CLASS_NAMES[:num_classes]
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    df_cm = pd.DataFrame(cm, index=names, columns=names)
    print('\nConfusion matrix (val):')
    print(df_cm.to_string())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml')
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    os.makedirs('checkpoints', exist_ok=True)

    train_loader, val_loader = build_loaders(cfg)

    audio_cfg   = cfg['audio']
    train_cfg   = audio_cfg['train']
    num_classes = cfg['emotions']['num_classes']

    model = AudioEmotionModel(
        num_classes=num_classes,
        model_name=audio_cfg['model_name'],
        dropout=audio_cfg['dropout'],
        freeze_feature_encoder=train_cfg['freeze_feature_encoder'],
        freeze_transformer=True,   # Фаза 1: только голова
    ).to(device)

    criterion = FocalLoss(
        gamma=train_cfg.get('focal_gamma', 2.0),
        label_smoothing=train_cfg.get('label_smoothing', 0.1),
        num_classes=num_classes,
    )

    mixup     = Mixup(alpha=train_cfg.get('mixup_alpha', 0.4))
    scaler    = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    ckpt_path = cfg['paths']['audio_model_ckpt']
    best_f1   = 0.0
    es        = EarlyStopping(patience=train_cfg.get('early_stopping_patience', 8))

    # ── Фаза 1: только голова ────────────────────────────────────────────
    warmup    = train_cfg['warmup_epochs']
    phase1_ep = warmup + 5

    print(f'\n=== Phase 1: head only ({phase1_ep} epochs) ===')
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=train_cfg['lr'] * 10,
        weight_decay=train_cfg['weight_decay'],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=phase1_ep)

    for epoch in range(1, phase1_ep + 1):
        tr_loss, tr_acc, tr_f1, tr_uar = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler, mixup)
        va_loss, va_acc, va_f1, va_uar = evaluate(model, val_loader, device, criterion)
        scheduler.step()
        print(f'[P1] Ep {epoch:02d} | train f1={tr_f1:.3f} uar={tr_uar:.3f} | '
              f'val f1={va_f1:.3f} uar={va_uar:.3f} acc={va_acc:.3f}')
        if va_f1 > best_f1:
            best_f1 = va_f1
            torch.save(model.state_dict(), ckpt_path)
            print(f'  -> Saved (f1={best_f1:.3f})')

    # ── Фаза 2: full fine-tuning ─────────────────────────────────────────
    remaining = train_cfg['epochs'] - phase1_ep
    print(f'\n=== Phase 2: full fine-tune ({remaining} epochs) ===')
    model.unfreeze_transformer()

    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg['lr'],
        weight_decay=train_cfg['weight_decay'],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=remaining)
    es        = EarlyStopping(patience=train_cfg.get('early_stopping_patience', 8))

    for epoch in range(1, remaining + 1):
        tr_loss, tr_acc, tr_f1, tr_uar = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler, mixup)
        va_loss, va_acc, va_f1, va_uar = evaluate(model, val_loader, device, criterion)
        scheduler.step()
        print(f'[P2] Ep {epoch:02d} | train f1={tr_f1:.3f} uar={tr_uar:.3f} | '
              f'val f1={va_f1:.3f} uar={va_uar:.3f} acc={va_acc:.3f}')
        if va_f1 > best_f1:
            best_f1 = va_f1
            torch.save(model.state_dict(), ckpt_path)
            print(f'  -> Saved (f1={best_f1:.3f})')
        if es.step(va_f1):
            print(f'  Early stopping at epoch {epoch}')
            break

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print_confusion_matrix(model, val_loader, device, num_classes)
    print(f'\nBest val F1: {best_f1:.4f}  |  Checkpoint: {ckpt_path}')


if __name__ == '__main__':
    main()
