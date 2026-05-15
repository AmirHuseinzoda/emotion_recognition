"""
Разбивает RAVDESS + CREMA-D на train/val/test.

RAVDESS:  актёры 1–18  → train, 19–21 → val, 22–24 → test (actor-independent).
CREMA-D:  91 актёр (1001–1091), разбивка по актёрам:
            1001–1072 → train  (~79%)
            1073–1082 → val    (~11%)
            1083–1091 → test   (~10%)

CREMA-D аудио (.wav) включается автоматически, если указан --cremad_audio_dir
и скрипт extract_cremad_audio.py уже запущен.

Запуск:
  python scripts/make_splits.py \
      --raw_dir data/raw/RAVDESS \
      --cremad_dir data/raw/CREMA_D \
      --cremad_audio_dir data/raw/CREMA_D_audio \
      --splits_dir data/splits
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datasets.ravdess import build_ravdess_index, build_cremad_index


# 6-классовый ремаппинг
EMOTION_REMAP = {
    'calm':    ('neutral', 0),
    'happy':   ('happy',   1),
    'sad':     ('sad',     2),
    'angry':   ('angry',   3),
    'fearful': ('fearful', 4),
    'disgust': ('disgust', 5),
    # RAVDESS 'neutral' (01) и 'surprised' (08) удаляем
}
CREMAD_REMAP = {'neutral': 0, 'happy': 1, 'sad': 2, 'angry': 3, 'fearful': 4, 'disgust': 5}
KEEP_RAVDESS = set(EMOTION_REMAP.keys())


def remap_ravdess(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df['emotion'].isin(KEEP_RAVDESS)].copy()
    df['emotion'] = df['emotion'].map(lambda e: EMOTION_REMAP[e][0])
    df['label'] = df['emotion'].map(lambda e: dict(EMOTION_REMAP.values())[e])
    df['source'] = 'ravdess'
    return df


def print_stats(name: str, df: pd.DataFrame):
    print(f"\n{name}: {len(df)} samples")
    if df.empty:
        return
    by_source = df.groupby('source').size() if 'source' in df.columns else pd.Series()
    if not by_source.empty:
        for src, cnt in by_source.items():
            print(f"  {src}: {cnt}")
    print("  Class distribution:")
    dist = df.groupby('emotion').size().sort_values(ascending=False)
    for em, cnt in dist.items():
        print(f"    {em}: {cnt}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir',          default='data/raw/RAVDESS')
    parser.add_argument('--cremad_dir',        default='data/raw/CREMA_D')
    parser.add_argument('--cremad_audio_dir',  default='data/raw/CREMA_D_audio',
                        help='Директория с извлечёнными .wav CREMA-D (extract_cremad_audio.py)')
    parser.add_argument('--splits_dir',        default='data/splits')
    args = parser.parse_args()

    Path(args.splits_dir).mkdir(parents=True, exist_ok=True)

    # ── RAVDESS ──────────────────────────────────────────────────────────
    ravdess_df = build_ravdess_index(args.raw_dir, modality='both')
    ravdess_df = remap_ravdess(ravdess_df)

    train_rav = ravdess_df[ravdess_df['actor'] <= 18]
    val_rav   = ravdess_df[(ravdess_df['actor'] >= 19) & (ravdess_df['actor'] <= 21)]
    test_rav  = ravdess_df[ravdess_df['actor'] >= 22]

    # ── CREMA-D ──────────────────────────────────────────────────────────
    cremad_path = Path(args.cremad_dir)
    audio_dir   = args.cremad_audio_dir if Path(args.cremad_audio_dir).exists() else None

    if not cremad_path.exists():
        print(f'[WARN] CREMA-D not found at {cremad_path}, skipping')
        cremad_train = cremad_val = cremad_test = pd.DataFrame()
    else:
        cremad_df = build_cremad_index(args.cremad_dir, audio_dir=audio_dir)
        cremad_df['label'] = cremad_df['emotion'].map(CREMAD_REMAP)
        cremad_df = cremad_df.dropna(subset=['label'])
        cremad_df['label'] = cremad_df['label'].astype(int)

        wav_count = (cremad_df['ext'] == '.wav').sum()
        flv_count = (cremad_df['ext'] == '.flv').sum()
        print(f'CREMA-D: {flv_count} video (.flv), {wav_count} audio (.wav extracted), '
              f'{cremad_df["actor"].nunique()} actors')

        cremad_train = cremad_df[cremad_df['actor'] <= 1072]
        cremad_val   = cremad_df[(cremad_df['actor'] >= 1073) & (cremad_df['actor'] <= 1082)]
        cremad_test  = cremad_df[cremad_df['actor'] >= 1083]

    # ── Объединение ───────────────────────────────────────────────────────
    train_df = pd.concat([train_rav, cremad_train], ignore_index=True)
    val_df   = pd.concat([val_rav,   cremad_val],   ignore_index=True)
    test_df  = pd.concat([test_rav,  cremad_test],  ignore_index=True)

    for name, split in [('train', train_df), ('val', val_df), ('test', test_df)]:
        out = Path(args.splits_dir) / f'{name}.csv'
        split.to_csv(out, index=False)
        print_stats(name.upper(), split)
        print(f'  -> saved to {out}')


if __name__ == '__main__':
    main()
