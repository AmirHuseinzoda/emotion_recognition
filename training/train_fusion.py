"""
Обучение fusion-модуля поверх замороженных видео и аудио моделей.

Загружает лучшие чекпоинты обеих модалей, обучает только fusion-слой.
Сравниваются два метода: 'weighted' и 'attention'.

Запуск:
  python training/train_fusion.py --config configs/config.yaml --fusion_type attention
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
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datasets.ravdess import (RAVDESSAudioDataset, RAVDESSVideoDataset,
                               RAVDESSMultimodalDataset, IDX_TO_LABEL)
from models.video.backbone import VideoEmotionModel
from models.audio.transformer import AudioEmotionModel
from models.fusion.fusion import FusionModel

import yaml


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_loaders(cfg, splits_dir, proc_video_dir):
    train_df_all = pd.read_csv(splits_dir / 'train.csv')
    val_df_all   = pd.read_csv(splits_dir / 'val.csv')

    audio_cfg = cfg['audio']

    for split_name, df_all in [('train', train_df_all), ('val', val_df_all)]:
        # Только речевые видео (01-...) и речевое аудио (03-...)
        df_vid = df_all[df_all['ext'] == '.mp4'].reset_index(drop=True)
        df_vid = df_vid[df_vid['path'].apply(lambda p: Path(p).name.startswith('01-'))].reset_index(drop=True)
        df_aud = df_all[df_all['ext'] == '.wav'].reset_index(drop=True)
        df_aud = df_aud[df_aud['path'].apply(lambda p: Path(p).name.startswith('03-'))].reset_index(drop=True)

        # RAVDESS: видео "01-XX-...", аудио "03-XX-..." — первый компонент (модальность) отличается.
        # Матчим по компонентам 2–7 (всё кроме модальности).
        def content_key(p):
            return '-'.join(Path(p).stem.split('-')[1:])

        keys_vid = df_vid['path'].apply(content_key)
        keys_aud = df_aud['path'].apply(content_key)
        common = sorted(set(keys_vid) & set(keys_aud))

        df_vid = df_vid[keys_vid.isin(common)].reset_index(drop=True)
        df_aud = df_aud[keys_aud.isin(common)].reset_index(drop=True)

        # Сортируем одинаково, чтобы индексы совпадали
        df_vid = df_vid.assign(_key=df_vid['path'].apply(content_key)).sort_values('_key').drop(columns='_key').reset_index(drop=True)
        df_aud = df_aud.assign(_key=df_aud['path'].apply(content_key)).sort_values('_key').drop(columns='_key').reset_index(drop=True)

        # Дополнительно: убираем видео без .npy файла
        npy_exists = df_vid['path'].apply(
            lambda p: (proc_video_dir / (Path(p).stem + '.npy')).exists()
        )
        missing_keys = set(df_vid[~npy_exists]['path'].apply(content_key))
        df_vid = df_vid[npy_exists].reset_index(drop=True)
        df_aud = df_aud[~df_aud['path'].apply(content_key).isin(missing_keys)].reset_index(drop=True)

        print(f"  [{split_name}] paired samples: {len(df_vid)} video / {len(df_aud)} audio")

        vid_ds = RAVDESSVideoDataset(
            df_vid, proc_video_dir,
            window_frames=cfg['video']['window_frames'],
            augment=(split_name == 'train'),
        )
        aud_ds = RAVDESSAudioDataset(
            df_aud,
            target_sr=audio_cfg['sample_rate'],
            max_duration_sec=audio_cfg['max_duration_sec'],
            augment=(split_name == 'train'),
        )
        mm_ds = RAVDESSMultimodalDataset(aud_ds, vid_ds)

        loader = DataLoader(
            mm_ds,
            batch_size=cfg['fusion']['train']['batch_size'],
            shuffle=(split_name == 'train'),
            num_workers=4,
            pin_memory=True,
        )
        if split_name == 'train':
            train_loader = loader
        else:
            val_loader = loader

    return train_loader, val_loader


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for frames, waveform, labels in loader:
        frames   = frames.to(device)
        waveform = waveform.to(device)
        labels   = labels.to(device)

        out  = model(frames, waveform)
        loss = criterion(out['logits_fusion'], labels)
        total_loss += loss.item() * len(labels)

        all_preds.extend(out['logits_fusion'].argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    cm  = confusion_matrix(all_labels, all_preds)
    return total_loss / len(all_labels), acc, f1, cm


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for frames, waveform, labels in tqdm(loader, leave=False):
        frames   = frames.to(device)
        waveform = waveform.to(device)
        labels   = labels.to(device)

        optimizer.zero_grad()
        out  = model(frames, waveform)
        loss = criterion(out['logits_fusion'], labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.fusion.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(labels)
        all_preds.extend(out['logits_fusion'].argmax(1).detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return total_loss / len(all_labels), acc, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',      default='configs/config.yaml')
    parser.add_argument('--fusion_type', default='attention', choices=['weighted', 'attention'])
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  |  Fusion: {args.fusion_type}")

    os.makedirs('checkpoints', exist_ok=True)

    splits_dir    = Path(cfg['data']['splits_dir'])
    proc_video_dir = Path(cfg['data']['processed_dir']) / 'video'

    # ── Загрузка обученных моделей ────────────────────────────────────────
    tcn_cfg = cfg['video']['tcn']
    video_model = VideoEmotionModel(
        num_classes=cfg['emotions']['num_classes'],
        num_channels=tcn_cfg['num_channels'],
        num_levels=tcn_cfg['num_levels'],
        kernel_size=tcn_cfg['kernel_size'],
        dropout=tcn_cfg['dropout'],
    )
    video_model.load_state_dict(torch.load(cfg['paths']['video_model_ckpt'], map_location='cpu'))

    audio_model = AudioEmotionModel(
        num_classes=cfg['emotions']['num_classes'],
        model_name=cfg['audio']['model_name'],
        dropout=cfg['audio']['dropout'],
    )
    audio_model.load_state_dict(torch.load(cfg['paths']['audio_model_ckpt'], map_location='cpu'))

    fusion_model = FusionModel(
        video_model=video_model,
        audio_model=audio_model,
        fusion_type=args.fusion_type,
        num_classes=cfg['emotions']['num_classes'],
        video_embed_dim=tcn_cfg['num_channels'],
        audio_embed_dim=cfg['audio']['hidden_size'],
        hidden_dim=cfg['fusion']['hidden_dim'],
    ).to(device)

    train_loader, val_loader = build_loaders(cfg, splits_dir, proc_video_dir)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    train_cfg = cfg['fusion']['train']

    optimizer = AdamW(
        fusion_model.fusion.parameters(),
        lr=train_cfg['lr'],
        weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=train_cfg['epochs'])
    ckpt_path = cfg['paths']['fusion_model_ckpt']
    best_f1   = 0.0

    for epoch in range(1, train_cfg['epochs'] + 1):
        tr_loss, tr_acc, tr_f1 = train_epoch(fusion_model, train_loader, optimizer, criterion, device)
        va_loss, va_acc, va_f1, cm = evaluate(fusion_model, val_loader, device, criterion)
        scheduler.step()

        print(f"Ep {epoch:02d}/{train_cfg['epochs']} | "
              f"train f1={tr_f1:.3f} | val f1={va_f1:.3f} acc={va_acc:.3f}")

        if va_f1 > best_f1:
            best_f1 = va_f1
            torch.save(fusion_model.state_dict(), ckpt_path)
            print(f"  -> Saved (f1={best_f1:.3f})")

    # Финальная матрица ошибок
    _, _, _, cm = evaluate(fusion_model, val_loader, device, criterion)
    print("\nConfusion matrix (val):")
    labels_names = [IDX_TO_LABEL[i] for i in range(cfg['emotions']['num_classes'])]
    print(pd.DataFrame(cm, index=labels_names, columns=labels_names).to_string())
    print(f"\nBest val F1: {best_f1:.4f}")


if __name__ == '__main__':
    main()
