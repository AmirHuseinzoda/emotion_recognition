"""
Предобработка аудио RAVDESS:
  1. Ресэмплинг до 16 кГц
  2. Нормализация
  3. Обрезка/паддинг до max_duration_sec
  4. Сохранение как .npy (T,) float32

Запуск:
  python scripts/preprocess_audio.py --raw_dir data/raw/RAVDESS \
                                     --out_dir data/processed/audio
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datasets.ravdess import build_ravdess_index


def process_audio(path: str, target_sr: int, max_len: int) -> np.ndarray:
    # soundfile + librosa — не зависит от TorchCodec
    waveform, sr = sf.read(path, dtype='float32', always_2d=False)

    # Моно
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    # Ресэмплинг
    if sr != target_sr:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)

    # Нормализация
    peak = np.abs(waveform).max()
    if peak > 0:
        waveform = waveform / peak

    # Pad / trim
    if len(waveform) < max_len:
        waveform = np.pad(waveform, (0, max_len - len(waveform)))
    else:
        waveform = waveform[:max_len]

    return waveform.astype(np.float32)  # (T,)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir',         default='data/raw/RAVDESS')
    parser.add_argument('--out_dir',         default='data/processed/audio')
    parser.add_argument('--target_sr',       type=int,   default=16000)
    parser.add_argument('--max_duration_sec', type=float, default=5.0)
    parser.add_argument('--skip_existing',   action='store_true', default=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    max_len = int(args.max_duration_sec * args.target_sr)

    df = build_ravdess_index(args.raw_dir, modality='audio')
    if df.empty:
        print(f"[ERROR] No .wav files found in {args.raw_dir}")
        return

    print(f"Found {len(df)} audio files. Processing...")
    failed = 0

    for _, row in tqdm(df.iterrows(), total=len(df)):
        out_path = out_dir / (Path(row['path']).stem + '.npy')
        if args.skip_existing and out_path.exists():
            continue
        try:
            arr = process_audio(row['path'], args.target_sr, max_len)
            np.save(str(out_path), arr)
        except Exception as e:
            print(f"[WARN] {row['path']}: {e}")
            failed += 1

    print(f"\nDone. Failed: {failed}")


if __name__ == '__main__':
    main()
