"""
Предобработка видео RAVDESS:
  1. Обнаружение лица (MTCNN)
  2. Выравнивание и кроп до face_size × face_size
  3. Сохранение последовательности кадров как .npy (T, H, W, C) uint8

Запуск:
  python scripts/preprocess_video.py --raw_dir data/raw/RAVDESS \
                                     --out_dir data/processed/video \
                                     --face_size 112
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datasets.ravdess import build_ravdess_index, build_cremad_index

try:
    from facenet_pytorch import MTCNN
    import torch
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    detector = MTCNN(
        image_size=112,
        margin=20,
        keep_all=False,
        device=DEVICE,
        post_process=False,
    )
except ImportError:
    print("[ERROR] facenet-pytorch not installed. Run: pip install facenet-pytorch")
    sys.exit(1)


def extract_face_sequence(video_path: str, face_size: int, max_frames: int = 300,
                          batch_size: int = 32) -> np.ndarray:
    """
    Извлекает последовательность кропов лица из видео.
    Возвращает массив (T, face_size, face_size, 3) uint8.
    Возвращает None если лицо не найдено ни в одном кадре.
    """
    cap = cv2.VideoCapture(video_path)
    all_frames = []
    frame_idx = 0

    while cap.isOpened() and frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_idx += 1
    cap.release()

    if not all_frames:
        return None

    # Уменьшаем кадры перед детекцией — MTCNN обрабатывает 4x меньше пикселей
    small_frames = [cv2.resize(f, (640, 360)) for f in all_frames]

    faces = []
    for i in range(0, len(small_frames), batch_size):
        batch = small_frames[i:i + batch_size]
        try:
            results = detector(batch)
        except (ValueError, Exception):
            # Fallback: обрабатываем кадры по одному если батч не сработал
            results = [detector(f) for f in batch]

        for face_tensor in results:
            if face_tensor is not None:
                face_np = face_tensor.permute(1, 2, 0).byte().cpu().numpy()
                if face_np.shape[:2] != (face_size, face_size):
                    face_np = cv2.resize(face_np, (face_size, face_size))
                faces.append(face_np)

    if len(faces) == 0:
        return None
    return np.stack(faces, axis=0)  # (T, H, W, C)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir',    default='data/raw/RAVDESS')
    parser.add_argument('--cremad_dir', default='data/raw/CREMA_D')
    parser.add_argument('--out_dir',    default='data/processed/video')
    parser.add_argument('--face_size',  type=int, default=112)
    parser.add_argument('--skip_existing', action='store_true', default=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    ravdess_df = build_ravdess_index(args.raw_dir, modality='video')
    cremad_df  = pd.DataFrame()
    if Path(args.cremad_dir).exists():
        cremad_df = build_cremad_index(args.cremad_dir)

    df = pd.concat([ravdess_df, cremad_df], ignore_index=True)
    if df.empty:
        print("[ERROR] No video files found")
        return

    print(f"Found {len(df)} video files (RAVDESS: {len(ravdess_df)}, CREMA-D: {len(cremad_df)}). Processing...")

    skipped = 0
    failed = 0

    for _, row in tqdm(df.iterrows(), total=len(df)):
        out_path = out_dir / (Path(row['path']).stem + '.npy')

        if args.skip_existing and out_path.exists():
            skipped += 1
            continue

        seq = extract_face_sequence(row['path'], args.face_size)
        if seq is None:
            print(f"[WARN] No face detected: {row['path']}")
            failed += 1
            continue

        np.save(str(out_path), seq)

    print(f"\nDone. Skipped: {skipped}, Failed: {failed}, Saved: {len(df) - skipped - failed}")


if __name__ == '__main__':
    main()
