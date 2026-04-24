"""
RAVDESS dataset parser and PyTorch Dataset classes.

Filename format: MM-VV-EE-II-SS-RR-AA.ext
  MM - modality:   01=audio-only, 02=video-only, 03=audio-video
  VV - vocal:      01=speech, 02=song
  EE - emotion:    01=neutral, 02=calm, 03=happy, 04=sad,
                   05=angry, 06=fearful, 07=disgust, 08=surprised
  II - intensity:  01=normal, 02=strong
  SS - statement:  01="Kids are talking by the door"
                   02="Dogs are sitting by the door"
  RR - repetition: 01, 02
  AA - actor:      01–24 (odd=male, even=female)
"""

import os
import re
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import soundfile as sf
import librosa
from torch.utils.data import Dataset
import cv2
from PIL import Image
from torchvision import transforms

EMOTION_MAP = {
    '01': ('neutral',   0),
    '02': ('calm',      1),
    '03': ('happy',     2),
    '04': ('sad',       3),
    '05': ('angry',     4),
    '06': ('fearful',   5),
    '07': ('disgust',   6),
    '08': ('surprised', 7),
}

LABEL_TO_IDX = {name: idx for _, (name, idx) in EMOTION_MAP.items()}
IDX_TO_LABEL = {idx: name for name, idx in LABEL_TO_IDX.items()}

# 6-классовый маппинг (после объединения calm→neutral, удаления surprised/neutral_ravdess)
IDX_TO_LABEL_6 = {0: 'neutral', 1: 'happy', 2: 'sad', 3: 'angry', 4: 'fearful', 5: 'disgust'}


def parse_ravdess_filename(filepath: str) -> Optional[dict]:
    """Извлекает метаданные из имени файла RAVDESS."""
    name = Path(filepath).stem
    parts = name.split('-')
    if len(parts) != 7:
        return None
    modality, vocal, emotion, intensity, statement, repetition, actor = parts
    if emotion not in EMOTION_MAP:
        return None
    emotion_name, emotion_idx = EMOTION_MAP[emotion]
    return {
        'path': filepath,
        'modality': int(modality),
        'vocal': int(vocal),
        'emotion': emotion_name,
        'label': emotion_idx,
        'intensity': int(intensity),
        'statement': int(statement),
        'repetition': int(repetition),
        'actor': int(actor),
        'gender': 'male' if int(actor) % 2 == 1 else 'female',
    }


def build_ravdess_index(root_dir: str, modality: str = 'video') -> pd.DataFrame:
    """
    Сканирует директорию RAVDESS и строит DataFrame с метаданными.
    modality: 'video' (.mp4), 'audio' (.wav), 'both'
    """
    root = Path(root_dir)
    extensions = []
    if modality in ('video', 'both'):
        extensions.append('.mp4')
    if modality in ('audio', 'both'):
        extensions.append('.wav')

    records = []
    for ext in extensions:
        for filepath in sorted(root.rglob(f'*{ext}')):
            meta = parse_ravdess_filename(str(filepath))
            if meta is not None:
                meta['ext'] = ext
                records.append(meta)

    df = pd.DataFrame(records)
    return df


CREMAD_EMOTION_MAP = {
    'ANG': ('angry',   4),
    'DIS': ('disgust', 6),
    'FEA': ('fearful', 5),
    'HAP': ('happy',   2),
    'NEU': ('neutral', 0),
    'SAD': ('sad',     3),
}


def build_cremad_index(root_dir: str) -> pd.DataFrame:
    """
    Сканирует директорию CREMA-D и строит DataFrame.
    Формат файла: {actorID}_{sentence}_{emotion}_{intensity}.flv
    Все актёры идут в train (actor=-1 означает CREMA-D актёра).
    """
    root = Path(root_dir)
    records = []
    for filepath in sorted(root.rglob('*.flv')):
        parts = filepath.stem.split('_')
        if len(parts) < 3:
            continue
        emotion_code = parts[2].upper()
        if emotion_code not in CREMAD_EMOTION_MAP:
            continue
        emotion_name, emotion_idx = CREMAD_EMOTION_MAP[emotion_code]
        records.append({
            'path':   str(filepath),
            'emotion': emotion_name,
            'label':   emotion_idx,
            'actor':   -1,   # CREMA-D актёры всегда в train
            'ext':     '.flv',
            'source':  'cremad',
        })

    df = pd.DataFrame(records)
    return df


# ──────────────────────────────────────────────
# Аудио датасет
# ──────────────────────────────────────────────

class RAVDESSAudioDataset(Dataset):
    """
    Возвращает (waveform [1, T], label) для HuBERT/wav2vec fine-tuning.
    Waveform нормализован, ресэмплирован до target_sr, паддирован/обрезан до max_len сэмплов.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target_sr: int = 16000,
        max_duration_sec: float = 5.0,
        augment: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.target_sr = target_sr
        self.max_len = int(max_duration_sec * target_sr)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        waveform, sr = sf.read(row['path'], dtype='float32', always_2d=False)

        # Моно
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)

        # Ресэмплинг
        if sr != self.target_sr:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.target_sr)

        # Нормализация
        peak = np.abs(waveform).max()
        if peak > 0:
            waveform = waveform / peak

        # Pad / trim
        if len(waveform) < self.max_len:
            waveform = np.pad(waveform, (0, self.max_len - len(waveform)))
        else:
            waveform = waveform[:self.max_len]

        waveform = torch.tensor(waveform)

        if self.augment:
            waveform = self._augment(waveform)

        return waveform, int(row['label'])  # (T,), int

    def _augment(self, waveform: torch.Tensor) -> torch.Tensor:
        # Случайный гауссовый шум
        if torch.rand(1).item() < 0.3:
            waveform = waveform + 0.005 * torch.randn_like(waveform)
        # Случайный сдвиг по времени
        if torch.rand(1).item() < 0.3:
            shift = int(torch.randint(-1600, 1600, (1,)).item())
            waveform = torch.roll(waveform, shift, dims=-1)
        return waveform


# ──────────────────────────────────────────────
# Видео датасет (предобработанные последовательности)
# ──────────────────────────────────────────────

class RAVDESSVideoDataset(Dataset):
    """
    Загружает предобработанные последовательности кадров лиц из processed_dir.
    Каждый файл: .npy формата (T, C, H, W), float32, нормализован.

    Предполагает, что скрипт scripts/preprocess_video.py уже запущен.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        processed_dir: str,
        window_frames: int = 24,
        augment: bool = False,
    ):
        self.processed_dir = Path(processed_dir)
        self.window_frames = window_frames
        self.augment = augment

        # Оставляем только файлы у которых есть .npy
        mask = df['path'].apply(
            lambda p: (self.processed_dir / (Path(p).stem + '.npy')).exists()
        )
        dropped = (~mask).sum()
        if dropped > 0:
            print(f"[VideoDataset] Skipped {dropped} samples without .npy")
        self.df = df[mask].reset_index(drop=True)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        npy_path = self.processed_dir / (Path(row['path']).stem + '.npy')

        frames = np.load(str(npy_path))  # (T, H, W, C) uint8

        # Случайное окно или центральное
        T = len(frames)
        if T >= self.window_frames:
            if self.augment:
                start = np.random.randint(0, T - self.window_frames + 1)
            else:
                start = (T - self.window_frames) // 2
            frames = frames[start: start + self.window_frames]
        else:
            # Паддинг повторением последнего кадра
            pad = self.window_frames - T
            frames = np.concatenate([frames, np.tile(frames[-1:], (pad, 1, 1, 1))], axis=0)

        # (T, C, H, W) float tensor
        tensor_frames = torch.stack([
            self.transform(Image.fromarray(f)) for f in frames
        ])  # (T, 3, H, W)

        if self.augment:
            tensor_frames = self._augment(tensor_frames)

        return tensor_frames, int(row['label'])

    def _augment(self, frames: torch.Tensor) -> torch.Tensor:
        # Горизонтальный флип
        if torch.rand(1).item() < 0.5:
            frames = torch.flip(frames, dims=[-1])

        # Яркость / контраст / насыщенность — одни параметры для всей последовательности
        if torch.rand(1).item() < 0.5:
            brightness = 1.0 + (torch.rand(1).item() - 0.5) * 0.4   # [0.8, 1.2]
            contrast   = 1.0 + (torch.rand(1).item() - 0.5) * 0.4
            frames = torch.clamp(frames * brightness, -3, 3)
            mean = frames.mean(dim=[-2, -1], keepdim=True)
            frames = torch.clamp(mean + (frames - mean) * contrast, -3, 3)

        # Небольшой поворот (±15°) — одинаковый для всех кадров
        if torch.rand(1).item() < 0.4:
            import torchvision.transforms.functional as TF
            angle = (torch.rand(1).item() - 0.5) * 30.0
            frames = torch.stack([TF.rotate(f, angle) for f in frames])

        # Гауссовский шум
        if torch.rand(1).item() < 0.3:
            frames = frames + torch.randn_like(frames) * 0.05

        return frames


# ──────────────────────────────────────────────
# Мультимодальный датасет
# ──────────────────────────────────────────────

class RAVDESSMultimodalDataset(Dataset):
    """Объединяет аудио и видео для обучения fusion-модуля."""

    def __init__(self, audio_ds: RAVDESSAudioDataset, video_ds: RAVDESSVideoDataset):
        assert len(audio_ds) == len(video_ds), "Audio and video datasets must be aligned"
        self.audio_ds = audio_ds
        self.video_ds = video_ds

    def __len__(self):
        return len(self.audio_ds)

    def __getitem__(self, idx):
        waveform, label_a = self.audio_ds[idx]
        frames, label_v = self.video_ds[idx]
        assert label_a == label_v
        return frames, waveform, label_a
