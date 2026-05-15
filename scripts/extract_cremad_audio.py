"""
Извлекает аудио из видеофайлов CREMA-D (.flv) в .wav через ffmpeg.

Запуск:
  python scripts/extract_cremad_audio.py --flv_dir data/raw/CREMA_D \
                                          --wav_dir data/raw/CREMA_D_audio

Требования: pip install imageio-ffmpeg
  (статически слинкованный ffmpeg, не требует системной установки)
"""

import argparse
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

try:
    import imageio_ffmpeg
    _FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    import shutil
    _FFMPEG_PATH = shutil.which('ffmpeg') or 'ffmpeg'


def extract_one(flv_path: Path, wav_path: Path, sr: int) -> tuple[str, bool, str]:
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        _FFMPEG_PATH, '-y', '-loglevel', 'warning',
        '-i', str(flv_path.resolve()),
        '-ar', str(sr),
        '-ac', '1',
        '-vn',
        str(wav_path.resolve()),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    ok = result.returncode == 0
    err = result.stdout.decode('utf-8', errors='replace').strip() if not ok else ''
    return str(flv_path), ok, err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--flv_dir',       default='data/raw/CREMA_D')
    parser.add_argument('--wav_dir',       default='data/raw/CREMA_D_audio')
    parser.add_argument('--sample_rate',   type=int, default=16000)
    parser.add_argument('--workers',       type=int, default=4)
    parser.add_argument('--skip_existing', action='store_true', default=True)
    args = parser.parse_args()

    print(f'ffmpeg: {_FFMPEG_PATH}')
    check = subprocess.run([_FFMPEG_PATH, '-version'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if check.returncode != 0:
        print('[ERROR] ffmpeg не запускается. Установи: pip install imageio-ffmpeg')
        sys.exit(1)
    print(check.stdout.decode('utf-8', errors='replace').splitlines()[0])

    flv_dir = Path(args.flv_dir)
    wav_dir = Path(args.wav_dir)

    if not flv_dir.exists():
        print(f'[ERROR] CREMA-D directory not found: {flv_dir}')
        sys.exit(1)

    flv_files = sorted(flv_dir.rglob('*.flv'))
    if not flv_files:
        print(f'[ERROR] No .flv files found in {flv_dir}')
        sys.exit(1)

    print(f'Found {len(flv_files)} .flv files. Output -> {wav_dir}')

    tasks = []
    for flv in flv_files:
        wav = wav_dir / (flv.stem + '.wav')
        if args.skip_existing and wav.exists():
            continue
        tasks.append((flv, wav))

    if not tasks:
        print('All files already extracted.')
        return

    print(f'Extracting {len(tasks)} files with {args.workers} workers...')
    failed = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(extract_one, flv, wav, args.sample_rate): flv
                   for flv, wav in tasks}
        for fut in tqdm(as_completed(futures), total=len(futures)):
            path, ok, err = fut.result()
            if not ok:
                failed.append((path, err))

    print(f'\nDone. Extracted: {len(tasks) - len(failed)}, Failed: {len(failed)}')
    for path, err in failed[:10]:
        print(f'  FAIL {path}: {err}')


if __name__ == '__main__':
    main()
