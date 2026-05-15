"""
Обучение fusion-модуля поверх видео и аудио моделей.

Стратегия двухфазного обучения:
  Фаза 1: базовые модели заморожены, обучается только fusion-слой.
  Фаза 2 (joint fine-tuning): размораживаем всё, дифференциальный LR.
            fusion:       lr  (1e-4)
            base models:  joint_finetune_lr (5e-6)
          + auxiliary loss из видео и аудио ветвей (предотвращает забывание).

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
from sklearn.metrics import f1_score, accuracy_score, recall_score, confusion_matrix
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datasets.ravdess import (RAVDESSAudioDataset, RAVDESSVideoDataset,
                               RAVDESSMultimodalDataset, IDX_TO_LABEL_6 as IDX_TO_LABEL)
from models.video.backbone import VideoEmotionModel
from models.audio.transformer import AudioEmotionModel
from models.fusion.fusion import FusionModel
from training.utils import FocalLoss, EarlyStopping, make_weighted_sampler

import yaml


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _pair_ravdess(df_all: pd.DataFrame, proc_video_dir: Path):
    def content_key(p):
        return '-'.join(Path(p).stem.split('-')[1:])

    df_vid = df_all[(df_all['ext'] == '.mp4') &
                     df_all['path'].apply(lambda p: Path(p).name.startswith('01-'))].copy()
    df_aud = df_all[(df_all['ext'] == '.wav') &
                     df_all['path'].apply(lambda p: Path(p).name.startswith('03-'))].copy()

    keys_vid = df_vid['path'].apply(content_key)
    keys_aud = df_aud['path'].apply(content_key)
    common = sorted(set(keys_vid) & set(keys_aud))

    df_vid = (df_vid[keys_vid.isin(common)]
              .assign(_k=lambda d: d['path'].apply(content_key))
              .sort_values('_k').drop(columns='_k').reset_index(drop=True))
    df_aud = (df_aud[keys_aud.isin(common)]
              .assign(_k=lambda d: d['path'].apply(content_key))
              .sort_values('_k').drop(columns='_k').reset_index(drop=True))

    npy_mask = df_vid['path'].apply(lambda p: (proc_video_dir / (Path(p).stem + '.npy')).exists())
    missing  = set(df_vid[~npy_mask]['path'].apply(content_key))
    df_vid   = df_vid[npy_mask].reset_index(drop=True)
    df_aud   = df_aud[~df_aud['path'].apply(content_key).isin(missing)].reset_index(drop=True)

    return df_vid, df_aud


def _pair_cremad(df_all: pd.DataFrame, proc_video_dir: Path):
    if 'source' not in df_all.columns:
        return pd.DataFrame(), pd.DataFrame()

    c_vid = df_all[(df_all['ext'] == '.flv') & (df_all['source'] == 'cremad')].copy()
    c_aud = df_all[(df_all['ext'] == '.wav') & (df_all['source'] == 'cremad')].copy()
    if c_vid.empty or c_aud.empty:
        return pd.DataFrame(), pd.DataFrame()

    v_stems = c_vid['path'].apply(lambda p: Path(p).stem)
    a_stems = c_aud['path'].apply(lambda p: Path(p).stem)
    common  = sorted(set(v_stems) & set(a_stems))
    if not common:
        return pd.DataFrame(), pd.DataFrame()

    c_vid = (c_vid[v_stems.isin(common)]
             .assign(_s=lambda d: d['path'].apply(lambda p: Path(p).stem))
             .sort_values('_s').drop(columns='_s').reset_index(drop=True))
    c_aud = (c_aud[a_stems.isin(common)]
             .assign(_s=lambda d: d['path'].apply(lambda p: Path(p).stem))
             .sort_values('_s').drop(columns='_s').reset_index(drop=True))

    npy_mask = c_vid['path'].apply(lambda p: (proc_video_dir / (Path(p).stem + '.npy')).exists())
    missing  = set(c_vid[~npy_mask]['path'].apply(lambda p: Path(p).stem))
    c_vid    = c_vid[npy_mask].reset_index(drop=True)
    c_aud    = c_aud[~c_aud['path'].apply(lambda p: Path(p).stem).isin(missing)].reset_index(drop=True)

    return c_vid, c_aud


def build_loaders(cfg, splits_dir, proc_video_dir):
    train_df_all = pd.read_csv(splits_dir / 'train.csv')
    val_df_all   = pd.read_csv(splits_dir / 'val.csv')
    audio_cfg    = cfg['audio']
    train_loader = val_loader = None

    for split_name, df_all in [('train', train_df_all), ('val', val_df_all)]:
        cre_vid, cre_aud = _pair_cremad(df_all, proc_video_dir)

        df_vid = cre_vid
        df_aud = cre_aud

        print(f'  [{split_name}] CREMA-D: {len(cre_vid)} pairs')

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

        if split_name == 'train':
            labels = np.array([vid_ds.df.iloc[i]['label'] for i in range(len(vid_ds))])
            num_classes = cfg['emotions']['num_classes']
            sampler = make_weighted_sampler(labels, num_classes)
            loader = DataLoader(mm_ds, batch_size=cfg['fusion']['train']['batch_size'],
                                sampler=sampler, num_workers=4, pin_memory=True)
            train_loader = loader
        else:
            loader = DataLoader(mm_ds, batch_size=cfg['fusion']['train']['batch_size'],
                                shuffle=False, num_workers=4, pin_memory=True)
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

        with torch.cuda.amp.autocast():
            out  = model(frames, waveform)
            loss = criterion(out['logits_fusion'], labels)
        total_loss += loss.item() * len(labels)

        all_preds.extend(out['logits_fusion'].argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    uar = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    cm  = confusion_matrix(all_labels, all_preds)
    return total_loss / len(all_labels), acc, f1, uar, cm


def train_epoch(model, loader, optimizer, criterion, device, scaler,
                aux_weight: float = 0.0, train_fusion_only: bool = True):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for frames, waveform, labels in tqdm(loader, leave=False):
        frames   = frames.to(device)
        waveform = waveform.to(device)
        labels   = labels.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            out  = model(frames, waveform)
            loss = criterion(out['logits_fusion'], labels)

            # Auxiliary losses от базовых моделей — предотвращает забывание
            if aux_weight > 0 and not train_fusion_only:
                loss_v = criterion(out['logits_video'], labels)
                loss_a = criterion(out['logits_audio'], labels)
                loss = loss + aux_weight * (loss_v + loss_a)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        if train_fusion_only:
            nn.utils.clip_grad_norm_(model.fusion.parameters(), max_norm=1.0)
        else:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

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

    splits_dir     = Path(cfg['data']['splits_dir'])
    proc_video_dir = Path(cfg['data']['processed_dir']) / 'video'

    # ── Загрузка обученных базовых моделей ───────────────────────────────
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

    train_cfg = cfg['fusion']['train']
    criterion = FocalLoss(
        gamma=train_cfg.get('focal_gamma', 2.0),
        label_smoothing=train_cfg.get('label_smoothing', 0.1),
        num_classes=cfg['emotions']['num_classes'],
    )

    scaler    = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    ckpt_path = cfg['paths']['fusion_model_ckpt']
    best_f1   = 0.0
    es        = EarlyStopping(patience=train_cfg.get('early_stopping_patience', 6))

    # ── Фаза 1: только fusion-слой ───────────────────────────────────────
    phase1_epochs = train_cfg['epochs']
    print(f'\n=== Phase 1: fusion head only ({phase1_epochs} epochs) ===')

    optimizer = AdamW(
        fusion_model.fusion.parameters(),
        lr=train_cfg['lr'],
        weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=phase1_epochs)

    for epoch in range(1, phase1_epochs + 1):
        tr_loss, tr_acc, tr_f1 = train_epoch(
            fusion_model, train_loader, optimizer, criterion, device, scaler,
            aux_weight=0.0, train_fusion_only=True)
        va_loss, va_acc, va_f1, va_uar, cm = evaluate(fusion_model, val_loader, device, criterion)
        scheduler.step()

        print(f"Ep {epoch:02d}/{phase1_epochs} | "
              f"train f1={tr_f1:.3f} | val f1={va_f1:.3f} uar={va_uar:.3f} acc={va_acc:.3f}")

        if va_f1 > best_f1:
            best_f1 = va_f1
            torch.save(fusion_model.state_dict(), ckpt_path)
            print(f"  -> Saved (f1={best_f1:.3f})")
        if es.step(va_f1):
            print(f"  Early stopping at epoch {epoch}")
            break

    # ── Фаза 2: joint fine-tuning (опционально) ──────────────────────────
    joint_epochs = train_cfg.get('joint_finetune_epochs', 0)
    if joint_epochs > 0 and train_cfg.get('joint_finetune', False):
        print(f'\n=== Phase 2: joint fine-tuning ({joint_epochs} epochs) ===')
        fusion_model.enable_joint_finetune()
        jft_lr    = train_cfg.get('joint_finetune_lr', 5e-6)
        aux_w     = train_cfg.get('aux_loss_weight', 0.3)
        es_joint  = EarlyStopping(patience=train_cfg.get('early_stopping_patience', 6))

        optimizer = AdamW([
            {'params': fusion_model.fusion.parameters(),       'lr': train_cfg['lr']},
            {'params': fusion_model.video_model.parameters(),  'lr': jft_lr},
            {'params': fusion_model.audio_model.parameters(),  'lr': jft_lr},
        ], weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=joint_epochs)

        for epoch in range(1, joint_epochs + 1):
            tr_loss, tr_acc, tr_f1 = train_epoch(
                fusion_model, train_loader, optimizer, criterion, device, scaler,
                aux_weight=aux_w, train_fusion_only=False)
            va_loss, va_acc, va_f1, va_uar, cm = evaluate(
                fusion_model, val_loader, device, criterion)
            scheduler.step()

            print(f"[JFT] Ep {epoch:02d}/{joint_epochs} | "
                  f"train f1={tr_f1:.3f} | val f1={va_f1:.3f} uar={va_uar:.3f}")

            if va_f1 > best_f1:
                best_f1 = va_f1
                torch.save(fusion_model.state_dict(), ckpt_path)
                print(f"  -> Saved (f1={best_f1:.3f})")
            if es_joint.step(va_f1):
                print(f"  Early stopping at epoch {epoch}")
                break

    # Финальная матрица ошибок
    fusion_model.load_state_dict(torch.load(ckpt_path, map_location=device))
    _, _, _, _, cm = evaluate(fusion_model, val_loader, device, criterion)
    num_classes  = cfg['emotions']['num_classes']
    labels_names = [IDX_TO_LABEL[i] for i in range(num_classes)]
    print("\nConfusion matrix (val):")
    print(pd.DataFrame(cm, index=labels_names, columns=labels_names).to_string())
    print(f"\nBest val F1: {best_f1:.4f}")


if __name__ == '__main__':
    main()
