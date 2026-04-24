"""
Разбивает RAVDESS + CREMA-D на train/val/test.

RAVDESS: актёры 1–18 → train, 19–21 → val, 22–24 → test (actor-independent).
CREMA-D: все актёры → train (другой датасет, нет пересечения с val/test).

Запуск:
  python scripts/make_splits.py --raw_dir data/raw/RAVDESS \
                                --cremad_dir data/raw/CREMA_D \
                                --splits_dir data/splits
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datasets.ravdess import build_ravdess_index, build_cremad_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir',    default='data/raw/RAVDESS')
    parser.add_argument('--cremad_dir', default='data/raw/CREMA_D')
    parser.add_argument('--splits_dir', default='data/splits')
    args = parser.parse_args()

    Path(args.splits_dir).mkdir(parents=True, exist_ok=True)

    # Новый маппинг эмоций: 6 классов
    # neutral(0)=RAVDESS calm + CREMA-D neutral
    # happy(1), sad(2), angry(3), fearful(4), disgust(5)
    # Удаляем: RAVDESS neutral, surprised
    EMOTION_REMAP = {
        'calm':     ('neutral', 0),
        'happy':    ('happy',   1),
        'sad':      ('sad',     2),
        'angry':    ('angry',   3),
        'fearful':  ('fearful', 4),
        'disgust':  ('disgust', 5),
        # neutral и surprised из RAVDESS удаляем (не включаем в KEEP)
    }
    KEEP_RAVDESS = set(EMOTION_REMAP.keys())

    def remap(df):
        df = df[df['emotion'].isin(KEEP_RAVDESS)].copy()
        df['emotion'] = df['emotion'].map(lambda e: EMOTION_REMAP[e][0])
        df['label']   = df['emotion'].map(lambda e: dict(EMOTION_REMAP.values())[e])
        return df

    # RAVDESS
    ravdess_df = build_ravdess_index(args.raw_dir, modality='both')
    ravdess_df = remap(ravdess_df)
    train_rav = ravdess_df[ravdess_df['actor'] <= 18]
    val_df    = ravdess_df[(ravdess_df['actor'] >= 19) & (ravdess_df['actor'] <= 21)]
    test_df   = ravdess_df[ravdess_df['actor'] >= 22]

    # CREMA-D (только видео, только train) — neutral уже label=0, остальные перемапим
    cremad_df = pd.DataFrame()
    cremad_path = Path(args.cremad_dir)
    if cremad_path.exists():
        cremad_df = build_cremad_index(args.cremad_dir)
        # Обновляем label по новому маппингу
        cremad_remap = {'neutral':0,'happy':1,'sad':2,'angry':3,'fearful':4,'disgust':5}
        cremad_df['label'] = cremad_df['emotion'].map(cremad_remap)
        print(f"CREMA-D: {len(cremad_df)} video files found")
    else:
        print(f"[WARN] CREMA-D not found at {args.cremad_dir}, skipping")

    train_df = pd.concat([train_rav, cremad_df], ignore_index=True)

    for name, split in [('train', train_df), ('val', val_df), ('test', test_df)]:
        out = Path(args.splits_dir) / f'{name}.csv'
        split.to_csv(out, index=False)
        print(f"{name}: {len(split)} samples -> {out}")

    print("\nClass distribution (train):")
    print(train_df['emotion'].value_counts().sort_index().to_string())


if __name__ == '__main__':
    main()
