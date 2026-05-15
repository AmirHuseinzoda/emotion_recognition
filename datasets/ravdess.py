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
# Это основной маппинг — используйте его везде при num_classes=6
IDX_TO_LABEL_6 = {0: 'neutral', 1: 'happy', 2: 'sad', 3: 'angry', 4: 'fearful', 5: 'disgust'}
CLASS_NAMES = [IDX_TO_LABEL_6[i] for i in range(6)]  # ['neutral','happy','sad','angry','fearful','disgust']


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
    'ANG': ('angry',   3),
    'DIS': ('disgust', 5),
    'FEA': ('fearful', 4),
    'HAP': ('happy',   1),
    'NEU': ('neutral', 0),
    'SAD': ('sad',     2),
}


def build_cremad_index(root_dir: str, audio_dir: str = None) -> pd.DataFrame:
    """
    Сканирует директорию CREMA-D и строит DataFrame.
    Формат файла: {actorID}_{sentence}_{emotion}_{intensity}.flv/.wav

    Args:
        root_dir:  директория с .flv видеофайлами CREMA-D
        audio_dir: директория с извлечёнными .wav файлами (опционально).
                   Создаётся скриптом scripts/extract_cremad_audio.py.
    """
    records = []

    def _parse_file(filepath: Path, ext: str):
        parts = filepath.stem.split('_')
        if len(parts) < 3:
            return None
        emotion_code = parts[2].upper()
        if emotion_code not in CREMAD_EMOTION_MAP:
            return None
        emotion_name, emotion_idx = CREMAD_EMOTION_MAP[emotion_code]
        try:
            actor_id = int(parts[0])
        except ValueError:
            actor_id = -1
        return {
            'path':    str(filepath),
            'emotion': emotion_name,
            'label':   emotion_idx,
            'actor':   actor_id,
            'ext':     ext,
            'source':  'cremad',
        }

    # Видео: .flv файлы
    for filepath in sorted(Path(root_dir).rglob('*.flv')):
        rec = _parse_file(filepath, '.flv')
        if rec is not None:
            records.append(rec)

    # Аудио: извлечённые .wav файлы (если директория указана и существует)
    if audio_dir:
        audio_path = Path(audio_dir)
        if audio_path.exists():
            for filepath in sorted(audio_path.rglob('*.wav')):
                rec = _parse_file(filepath, '.wav')
                if rec is not None:
                    records.append(rec)

    return pd.DataFrame(records)


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
        try:
            waveform, sr = sf.read(row['path'], dtype='float32', always_2d=False)
        except Exception:
            return torch.zeros(self.max_len), int(row['label'])

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
        T = waveform.shape[-1]

        # Volume jitter — имитирует разные расстояния до микрофона
        if torch.rand(1).item() < 0.5:
            gain = 0.5 + torch.rand(1).item()   # [0.5, 1.5]
            waveform = waveform * gain

        # Gaussian noise с переменной интенсивностью
        if torch.rand(1).item() < 0.4:
            sigma = 0.002 + torch.rand(1).item() * 0.012
            waveform = waveform + sigma * torch.randn_like(waveform)

        # Time shift ±0.3 сек
        if torch.rand(1).item() < 0.4:
            shift = int(torch.randint(-4800, 4800, (1,)).item())
            waveform = torch.roll(waveform, shift, dims=-1)

        # SpecAugment: time masking (до 20% сигнала, 1-2 маски)
        if torch.rand(1).item() < 0.5:
            waveform = waveform.clone()
            max_mask = max(1, int(T * 0.20))
            for _ in range(torch.randint(1, 3, (1,)).item()):
                mask_len = torch.randint(1, max_mask + 1, (1,)).item()
                if T - mask_len > 0:
                    start = torch.randint(0, T - mask_len, (1,)).item()
                    waveform[start: start + mask_len] = 0.0

        # Clamp чтобы не выйти за [-1, 1]
        waveform = waveform.clamp(-1.0, 1.0)
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
        import torchvision.transforms.functional as TF
        frames = frames.contiguous()  # flip/rotate возвращают view — нужна копия для in-place ops

        # Горизонтальный флип
        if torch.rand(1).item() < 0.5:
            frames = torch.flip(frames, dims=[-1])

        # Яркость / контраст — одни параметры для всей последовательности
        if torch.rand(1).item() < 0.5:
            brightness = 1.0 + (torch.rand(1).item() - 0.5) * 0.4
            contrast   = 1.0 + (torch.rand(1).item() - 0.5) * 0.4
            frames = torch.clamp(frames * brightness, -3, 3)
            mean = frames.mean(dim=[-2, -1], keepdim=True)
            frames = torch.clamp(mean + (frames - mean) * contrast, -3, 3)

        # Поворот ±15°
        if torch.rand(1).item() < 0.4:
            angle = (torch.rand(1).item() - 0.5) * 30.0
            frames = torch.stack([TF.rotate(f, angle) for f in frames])

        # Grayscale — снижает зависимость от цвета кожи и освещения
        if torch.rand(1).item() < 0.2:
            gray = frames.mean(dim=1, keepdim=True).expand_as(frames)
            frames = gray

        # RandomErasing — заставляет модель не полагаться на один регион лица
        if torch.rand(1).item() < 0.3:
            frames = frames.clone()
            T, C, H, W = frames.shape
            erase_h = int(H * (0.1 + torch.rand(1).item() * 0.2))
            erase_w = int(W * (0.1 + torch.rand(1).item() * 0.2))
            top  = torch.randint(0, H - erase_h + 1, (1,)).item()
            left = torch.randint(0, W - erase_w + 1, (1,)).item()
            frames[:, :, top:top + erase_h, left:left + erase_w] = 0.0

        # Temporal dropout — случайно обнуляет кадры, имитирует потери в realtime
        if torch.rand(1).item() < 0.2:
            frames = frames.clone()
            T = frames.shape[0]
            n_drop = max(1, int(T * 0.1))
            drop_idx = torch.randperm(T)[:n_drop]
            frames[drop_idx] = 0.0

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
