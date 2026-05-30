"""
Обучение видео-модели (EfficientNet-B0 + Bi-LSTM с attention pooling).

Стратегия двухфазного обучения:
  Фаза 1 (frozen_epochs): backbone заморожен, обучаем только LSTM + classifier.
  Фаза 2: разморозка backbone, обучение end-to-end с меньшим lr.

Данные: только CREMA-D .flv (91 актёр, actor-independent split).

Улучшения vs предыдущей версии:
  - Mixup augmentation (50% вероятность применения к батчу)
  - AMP (torch.cuda.amp) для экономии памяти при window_frames=32

Запуск:
  python training/train_video.py --config configs/config.yaml
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
from datasets.ravdess import RAVDESSVideoDataset, CLASS_NAMES
from models.video.backbone import VideoEmotionModel
from training.utils import FocalLoss, EarlyStopping, make_weighted_sampler, Mixup, mixup_loss

import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_loaders(cfg: dict):
    splits_dir  = Path(cfg['data']['splits_dir'])
    proc_dir    = Path(cfg['data']['processed_dir']) / 'video'
    wf          = cfg['video']['window_frames']
    train_cfg   = cfg['video']['train']
    num_classes = cfg['emotions']['num_classes']

    train_df = pd.read_csv(splits_dir / 'train.csv')
    val_df   = pd.read_csv(splits_dir / 'val.csv')

    # Только CREMA-D: 91 актёр, чистое студийное качество, нет domain shift
    train_df = train_df[train_df['ext'] == '.flv'].reset_index(drop=True)
    val_df   = val_df[val_df['ext'] == '.flv'].reset_index(drop=True)

    # Обрезаем классы с переобучением (sad, angry, fearful имеют низкий val recall
    # при высоком train recall => модель запоминает актёро-специфичные паттерны).
    # happy и disgust оставляем полностью — они обобщаются хорошо.
    # Cap = количество neutral (наименьший класс) чтобы не терять много данных.
    seed = cfg['data']['seed']
    counts = train_df['label'].value_counts()
    neutral_count = int(counts.get(0, counts.min()))  # label 0 = neutral
    overfit_labels = [2, 3, 4]  # sad, angry, fearful
    label_names = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgust']
    parts = []
    for label in range(num_classes):
        subset = train_df[train_df['label'] == label]
        if label in overfit_labels and len(subset) > neutral_count:
            subset = subset.sample(neutral_count, random_state=seed)
            print(f'  Capped {label_names[label]} ({label}): {counts[label]} -> {neutral_count}')
        parts.append(subset)
    train_df = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)

    # Статистика
    print(f'  [train] {len(train_df)} samples after cap | class dist: {train_df["label"].value_counts().sort_index().to_dict()}')
    print(f'  [val]   {len(val_df)} samples')

    train_ds = RAVDESSVideoDataset(train_df, proc_dir, wf, augment=True)
    val_ds   = RAVDESSVideoDataset(val_df,   proc_dir, wf, augment=False)

    bs = train_cfg['batch_size']

    if train_cfg.get('weighted_sampler', True):
        train_labels = train_ds.df['label'].values
        sampler = make_weighted_sampler(train_labels, num_classes)
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

    for frames, labels in loader:
        frames, labels = frames.to(device), labels.to(device)
        with torch.cuda.amp.autocast():
            logits = model(frames)
            loss   = criterion(logits, labels)
        total_loss += loss.item() * len(labels)
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    n = len(all_labels)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    uar = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    return total_loss / n, acc, f1, uar


def train_epoch(model, loader, optimizer, criterion, device, scaler, mixup: Mixup = None):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    mixup_prob = 0.5

    for frames, labels in tqdm(loader, leave=False):
        frames, labels = frames.to(device), labels.to(device)
        optimizer.zero_grad()

        use_mixup = mixup is not None and torch.rand(1).item() < mixup_prob
        if use_mixup:
            frames, y_a, y_b, lam = mixup.mix(frames, labels)

        with torch.cuda.amp.autocast():
            logits = model(frames)
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

    n = len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    uar = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    return total_loss / n, acc, f1, uar


def print_confusion_matrix(model, loader, device, num_classes):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for frames, labels in loader:
            frames = frames.to(device)
            preds = model(frames).argmax(dim=1).cpu().numpy()
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

    tcn_cfg     = cfg['video']['tcn']
    train_cfg   = cfg['video']['train']
    num_classes = cfg['emotions']['num_classes']

    model = VideoEmotionModel(
        num_classes=num_classes,
        num_channels=tcn_cfg['num_channels'],
        num_levels=tcn_cfg['num_levels'],
        kernel_size=tcn_cfg['kernel_size'],
        dropout=tcn_cfg['dropout'],
        pretrained=cfg['video']['backbone_pretrained'],
        frozen_backbone=True,
    ).to(device)

    # neutral=0, happy=1, sad=2, angry=3, fearful=4, disgust=5
    # Данные sad/angry/fearful уже обрезаны выше — используем умеренные веса.
    # angry дополнительно буcтим весом (низкий recall + данные обрезаны).
    # happy/disgust полные данные → слегка снижаем вес.
    class_weight = torch.tensor([1.2, 0.8, 1.1, 1.4, 1.1, 0.8], device=device)
    criterion = FocalLoss(
        gamma=train_cfg.get('focal_gamma', 2.0),
        label_smoothing=train_cfg.get('label_smoothing', 0.1),
        num_classes=num_classes,
        class_weight=class_weight,
    )

    mixup         = Mixup(alpha=train_cfg.get('mixup_alpha', 0.4))
    scaler        = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    frozen_epochs = cfg['video']['backbone_frozen_epochs']
    total_epochs  = train_cfg['epochs']
    ckpt_path     = cfg['paths']['video_model_ckpt']
    best_f1       = 0.0
    es            = EarlyStopping(patience=train_cfg.get('early_stopping_patience', 8))

    # ── Фаза 1: backbone заморожен ──────────────────────────────────────
    print(f'\n=== Phase 1: backbone frozen ({frozen_epochs} epochs) ===')
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=train_cfg['lr'], weight_decay=train_cfg['weight_decay']
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=frozen_epochs)

    for epoch in range(1, frozen_epochs + 1):
        tr_loss, tr_acc, tr_f1, tr_uar = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler, mixup)
        va_loss, va_acc, va_f1, va_uar = evaluate(model, val_loader, device, criterion)
        scheduler.step()

        print(f'[P1] Ep {epoch:02d}/{frozen_epochs} | '
              f'train f1={tr_f1:.3f} uar={tr_uar:.3f} | '
              f'val f1={va_f1:.3f} uar={va_uar:.3f} acc={va_acc:.3f}')

        if va_f1 > best_f1:
            best_f1 = va_f1
            torch.save(model.state_dict(), ckpt_path)
            print(f'  -> Saved (f1={best_f1:.3f})')

    # ── Фаза 2: разморозка backbone ──────────────────────────────────────
    remaining = total_epochs - frozen_epochs
    print(f'\n=== Phase 2: full fine-tune ({remaining} epochs) ===')
    model.unfreeze_backbone()

    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg['lr'] * 0.1,
        weight_decay=train_cfg['weight_decay']
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=remaining)

    for epoch in range(1, remaining + 1):
        tr_loss, tr_acc, tr_f1, tr_uar = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler, mixup)
        va_loss, va_acc, va_f1, va_uar = evaluate(model, val_loader, device, criterion)
        scheduler.step()

        print(f'[P2] Ep {epoch:02d}/{remaining} | '
              f'train f1={tr_f1:.3f} uar={tr_uar:.3f} | '
              f'val f1={va_f1:.3f} uar={va_uar:.3f} acc={va_acc:.3f}')

        if va_f1 > best_f1:
            best_f1 = va_f1
            torch.save(model.state_dict(), ckpt_path)
            print(f'  -> Saved (f1={best_f1:.3f})')
        if es.step(va_f1):
            print(f'  Early stopping at epoch {epoch}')
            break

    # Загружаем лучшую модель и печатаем confusion matrix
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print_confusion_matrix(model, val_loader, device, num_classes)
    print(f'\nBest val F1: {best_f1:.4f}  |  Checkpoint: {ckpt_path}')


if __name__ == '__main__':
    main()
