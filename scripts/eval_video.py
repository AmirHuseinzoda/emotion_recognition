"""
Оценка видео-модели на тестовой выборке.

Запуск:
  python scripts/eval_video.py --config configs/config.yaml
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, classification_report
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datasets.ravdess import RAVDESSVideoDataset, IDX_TO_LABEL
from models.video.backbone import VideoEmotionModel

import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []

    for frames, labels in loader:
        frames, labels = frames.to(device), labels.to(device)
        logits = model(frames)
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml')
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    splits_dir = Path(cfg['data']['splits_dir'])
    proc_dir   = Path(cfg['data']['processed_dir']) / 'video'
    wf         = cfg['video']['window_frames']

    test_df = pd.read_csv(splits_dir / 'test.csv')
    test_df = test_df[test_df['ext'].isin(['.mp4', '.flv'])].reset_index(drop=True)
    print(f"Test samples: {len(test_df)}")

    test_ds = RAVDESSVideoDataset(test_df, proc_dir, wf, augment=False)
    test_loader = DataLoader(test_ds, batch_size=cfg['video']['train']['batch_size'],
                             shuffle=False, num_workers=4, pin_memory=True)

    tcn_cfg = cfg['video']['tcn']
    model = VideoEmotionModel(
        num_classes=cfg['emotions']['num_classes'],
        num_channels=tcn_cfg['num_channels'],
        num_levels=tcn_cfg['num_levels'],
        kernel_size=tcn_cfg['kernel_size'],
        dropout=tcn_cfg['dropout'],
        pretrained=False,
        frozen_backbone=False,
    ).to(device)

    ckpt_path = cfg['paths']['video_model_ckpt']
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"Loaded checkpoint: {ckpt_path}\n")

    labels, preds = evaluate(model, test_loader, device)

    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average='macro', zero_division=0)

    print(f"Test Accuracy : {acc:.4f}")
    print(f"Test Macro F1 : {f1:.4f}\n")

    label_names = [IDX_TO_LABEL[i] for i in range(cfg['emotions']['num_classes'])]
    print(classification_report(labels, preds, target_names=label_names, zero_division=0))


if __name__ == '__main__':
    main()
