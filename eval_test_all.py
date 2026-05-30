"""
Оценка всех трёх модальностей на тестовой или валидационной выборке.
Выводит метрики по классам и сохраняет матрицы ошибок.

Запуск:
    python eval_test_all.py              # test (по умолчанию)
    python eval_test_all.py --split val  # val
"""

import sys; sys.path.insert(0, '.')
import argparse
import torch

_parser = argparse.ArgumentParser()
_parser.add_argument('--split', default='test', choices=['test', 'val'])
SPLIT = _parser.parse_args().split
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score,
    confusion_matrix, classification_report,
)

from datasets.ravdess import (
    RAVDESSAudioDataset,
    RAVDESSVideoDataset,
    RAVDESSMultimodalDataset,
    IDX_TO_LABEL_6 as IDX_TO_LABEL,
)
from models.audio.transformer import AudioEmotionModel
from models.video.backbone import VideoEmotionModel
from models.fusion.fusion import FusionModel

# ──────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────
cfg = yaml.safe_load(open('configs/config.yaml'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}\n')

NUM_CLASSES = cfg['emotions']['num_classes']
CLASS_NAMES = [IDX_TO_LABEL[i] for i in range(NUM_CLASSES)]
SPLITS_DIR  = Path(cfg['data']['splits_dir'])
PROC_DIR    = Path(cfg['data']['processed_dir']) / 'video'
AUDIO_CFG   = cfg['audio']
TCN_CFG     = cfg['video']['tcn']

OUT_DIR = Path('eval_results')
OUT_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────
def plot_confusion_matrix(cm: np.ndarray, title: str, out_path: Path):
    """Красивая нормированная матрица ошибок (проценты по строкам)."""
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, data, fmt, label in zip(
        axes,
        [cm, cm_norm],
        ['d', '.1%'],
        ['Counts', 'Row-normalised (recall per class)'],
    ):
        sns.heatmap(
            data,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            linewidths=0.5,
            linecolor='#e0e0e0',
            ax=ax,
            cbar=True,
            annot_kws={'size': 10},
        )
        ax.set_title(label, fontsize=11, pad=8)
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('True', fontsize=10)
        ax.tick_params(axis='x', rotation=30)
        ax.tick_params(axis='y', rotation=0)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved → {out_path}')


def print_report(labels, preds, title: str):
    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average='macro', zero_division=0)
    uar = recall_score(labels, preds, average='macro', zero_division=0)
    print(f'\n{"─"*55}')
    print(f'  {title}')
    print(f'{"─"*55}')
    print(f'  Accuracy : {acc:.4f}')
    print(f'  Macro F1 : {f1:.4f}')
    print(f'  UAR      : {uar:.4f}')
    print()
    print(classification_report(
        labels, preds,
        target_names=CLASS_NAMES,
        zero_division=0,
        digits=4,
    ))
    return acc, f1, uar


def pair_cremad(df_all, proc_video_dir):
    """Возвращает выровненные (df_vid, df_aud) только для CREMA-D."""
    c_vid = df_all[(df_all['ext'] == '.flv') & (df_all['source'] == 'cremad')].copy()
    c_aud = df_all[(df_all['ext'] == '.wav') & (df_all['source'] == 'cremad')].copy()
    if c_vid.empty or c_aud.empty:
        return pd.DataFrame(), pd.DataFrame()
    v_stems = c_vid['path'].apply(lambda p: Path(p).stem)
    a_stems = c_aud['path'].apply(lambda p: Path(p).stem)
    common = sorted(set(v_stems) & set(a_stems))
    c_vid = (c_vid[v_stems.isin(common)]
             .assign(_s=lambda d: d['path'].apply(lambda p: Path(p).stem))
             .sort_values('_s').drop(columns='_s').reset_index(drop=True))
    c_aud = (c_aud[a_stems.isin(common)]
             .assign(_s=lambda d: d['path'].apply(lambda p: Path(p).stem))
             .sort_values('_s').drop(columns='_s').reset_index(drop=True))
    npy_mask = c_vid['path'].apply(
        lambda p: (proc_video_dir / (Path(p).stem + '.npy')).exists()
    )
    missing = set(c_vid[~npy_mask]['path'].apply(lambda p: Path(p).stem))
    c_vid = c_vid[npy_mask].reset_index(drop=True)
    c_aud = c_aud[~c_aud['path'].apply(lambda p: Path(p).stem).isin(missing)].reset_index(drop=True)
    return c_vid, c_aud


# ──────────────────────────────────────────────────────
# Load models
# ──────────────────────────────────────────────────────
print('Loading models...')

audio_model = AudioEmotionModel(
    num_classes=NUM_CLASSES,
    model_name=AUDIO_CFG['model_name'],
    dropout=0.0,
    layer_aggregation=AUDIO_CFG.get('layer_aggregation', True),
)
audio_model.load_state_dict(
    torch.load('checkpoints/audio_best.pt', map_location='cpu', weights_only=False)
)
audio_model.eval().to(device)
print('  Audio model OK')

video_model = VideoEmotionModel(
    num_classes=NUM_CLASSES,
    num_channels=TCN_CFG['num_channels'],
    num_levels=TCN_CFG['num_levels'],
    kernel_size=TCN_CFG['kernel_size'],
    dropout=0.0,
    pretrained=False,
    temporal_type=cfg['video'].get('temporal_type', 'transformer'),
)
video_model.load_state_dict(
    torch.load('checkpoints/video_best.pt', map_location='cpu', weights_only=False)
)
video_model.eval().to(device)
print('  Video model OK')

fusion_model = FusionModel(
    video_model=video_model,
    audio_model=audio_model,
    fusion_type='attention',
    num_classes=NUM_CLASSES,
    video_embed_dim=TCN_CFG['num_channels'],
    audio_embed_dim=AUDIO_CFG['hidden_size'],
    hidden_dim=cfg['fusion']['hidden_dim'],
)
fusion_model.load_state_dict(
    torch.load('checkpoints/fusion_best.pt', map_location='cpu', weights_only=False)
)
fusion_model.eval().to(device)
print('  Fusion model OK\n')


# ──────────────────────────────────────────────────────
# TEST SET
# ──────────────────────────────────────────────────────
test_df_all = pd.read_csv(SPLITS_DIR / f'{SPLIT}.csv')

# ── 1. AUDIO ──────────────────────────────────────────
print('=== AUDIO — test set ===')
df_aud = test_df_all[
    test_df_all['ext'] == '.wav'
].reset_index(drop=True)
print(f'  Samples: {len(df_aud)}')

aud_ds = RAVDESSAudioDataset(
    df_aud,
    target_sr=AUDIO_CFG['sample_rate'],
    max_duration_sec=AUDIO_CFG['max_duration_sec'],
    augment=False,
)
aud_loader = DataLoader(aud_ds, batch_size=16, shuffle=False, num_workers=0)

aud_preds, aud_labels = [], []
with torch.no_grad():
    for waveform, labels in aud_loader:
        out = audio_model(waveform.to(device))
        aud_preds.extend(out.argmax(1).cpu().numpy())
        aud_labels.extend(labels.numpy())

aud_preds  = np.array(aud_preds)
aud_labels = np.array(aud_labels)

print_report(aud_labels, aud_preds, f'Audio (WavLM-Large) — {SPLIT.capitalize()}')
cm_aud = confusion_matrix(aud_labels, aud_preds)
plot_confusion_matrix(cm_aud, f'Audio model — {SPLIT.capitalize()} Confusion Matrix', OUT_DIR / f'cm_audio_{SPLIT}.png')

# ── 2. VIDEO ──────────────────────────────────────────
print('\n=== VIDEO — test set ===')
df_vid = test_df_all[
    test_df_all['ext'].isin(['.flv', '.mp4'])
].reset_index(drop=True)
print(f'  Samples (before .npy filter): {len(df_vid)}')

vid_ds = RAVDESSVideoDataset(
    df_vid, PROC_DIR,
    window_frames=cfg['video']['window_frames'],
    augment=False,
)
print(f'  Samples (after .npy filter): {len(vid_ds)}')
vid_loader = DataLoader(vid_ds, batch_size=12, shuffle=False, num_workers=0)

vid_preds, vid_labels = [], []
with torch.no_grad():
    for frames, labels in vid_loader:
        out = video_model(frames.to(device))
        vid_preds.extend(out.argmax(1).cpu().numpy())
        vid_labels.extend(labels.numpy())

vid_preds  = np.array(vid_preds)
vid_labels = np.array(vid_labels)

print_report(vid_labels, vid_preds, f'Video (EfficientNet-B0 + Transformer) — {SPLIT.capitalize()}')
cm_vid = confusion_matrix(vid_labels, vid_preds)
plot_confusion_matrix(cm_vid, f'Video model — {SPLIT.capitalize()} Confusion Matrix', OUT_DIR / f'cm_video_{SPLIT}.png')

# ── 3. FUSION ─────────────────────────────────────────
print('\n=== FUSION — test set ===')
df_fvid, df_faud = pair_cremad(test_df_all, PROC_DIR)
print(f'  Paired samples: {len(df_fvid)}')

fvid_ds = RAVDESSVideoDataset(df_fvid, PROC_DIR, window_frames=cfg['video']['window_frames'], augment=False)
faud_ds = RAVDESSAudioDataset(df_faud, target_sr=AUDIO_CFG['sample_rate'],
                               max_duration_sec=AUDIO_CFG['max_duration_sec'], augment=False)
mm_ds   = RAVDESSMultimodalDataset(faud_ds, fvid_ds)
mm_loader = DataLoader(mm_ds, batch_size=16, shuffle=False, num_workers=0)

fus_preds, fus_labels = [], []
with torch.no_grad():
    for frames, waveform, labels in mm_loader:
        out = fusion_model(frames.to(device), waveform.to(device))
        fus_preds.extend(out['logits_fusion'].argmax(1).cpu().numpy())
        fus_labels.extend(labels.numpy())

fus_preds  = np.array(fus_preds)
fus_labels = np.array(fus_labels)

print_report(fus_labels, fus_preds, f'Fusion (Attention) — {SPLIT.capitalize()}')
cm_fus = confusion_matrix(fus_labels, fus_preds)
plot_confusion_matrix(cm_fus, f'Fusion model — {SPLIT.capitalize()} Confusion Matrix', OUT_DIR / f'cm_fusion_{SPLIT}.png')

# ── 4. SUMMARY TABLE ──────────────────────────────────
print('\n' + '═'*55)
print(f'  SUMMARY — {SPLIT.capitalize()} set')
print('═'*55)
summary = []
for name, labels, preds in [
    ('Audio',  aud_labels, aud_preds),
    ('Video',  vid_labels, vid_preds),
    ('Fusion', fus_labels, fus_preds),
]:
    summary.append({
        'Model':    name,
        'Accuracy': f'{accuracy_score(labels, preds):.4f}',
        'Macro F1': f'{f1_score(labels, preds, average="macro", zero_division=0):.4f}',
        'UAR':      f'{recall_score(labels, preds, average="macro", zero_division=0):.4f}',
    })
print(pd.DataFrame(summary).to_string(index=False))
print(f'\nConfusion matrices saved to: {OUT_DIR.resolve()}')
