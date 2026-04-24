"""
Оптимизация моделей для инференса в реальном времени.

Применяет:
  1. Dynamic quantization к аудио-модели (HuBERT) → ~2x быстрее на CPU, -75% RAM
  2. TorchScript к видео-модели → ~1.3x быстрее
  3. Сохраняет оптимизированные версии в checkpoints/optimized/

Запуск:
  python inference/optimize.py --config configs/config.yaml
"""

import sys
import os
import yaml
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.video.backbone import VideoEmotionModel
from models.audio.transformer import AudioEmotionModel


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def quantize_audio_model(model: AudioEmotionModel) -> nn.Module:
    """
    Dynamic quantization линейных слоёв HuBERT.
    Уменьшает модель с ~360MB до ~90MB, ускоряет CPU-инференс в 2x.
    Качество теряется незначительно (<1% F1).
    """
    quantized = torch.quantization.quantize_dynamic(
        model,
        qconfig_spec={nn.Linear},
        dtype=torch.qint8,
    )
    return quantized


def script_video_model(model: VideoEmotionModel) -> torch.jit.ScriptModule:
    """
    Компилирует видео-модель через TorchScript.
    Убирает overhead Python интерпретатора, ускоряет на ~30%.
    """
    model.eval()
    # Пример входа для трассировки
    dummy = torch.zeros(1, 24, 3, 112, 112)
    scripted = torch.jit.trace(model, dummy)
    return scripted


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml')
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path('checkpoints/optimized')
    out_dir.mkdir(parents=True, exist_ok=True)

    tcn_cfg = cfg['video']['tcn']

    # ── Видео-модель → TorchScript ────────────────────────────────────────
    print("Optimizing video model (TorchScript)...")
    video_model = VideoEmotionModel(
        num_classes=cfg['emotions']['num_classes'],
        num_channels=tcn_cfg['num_channels'],
        num_levels=tcn_cfg['num_levels'],
        kernel_size=tcn_cfg['kernel_size'],
        dropout=0.0,
    )
    video_model.load_state_dict(
        torch.load(cfg['paths']['video_model_ckpt'], map_location='cpu')
    )
    video_model.eval()
    scripted_video = script_video_model(video_model)
    scripted_video.save(str(out_dir / 'video_scripted.pt'))
    print(f"  → Saved: {out_dir / 'video_scripted.pt'}")

    # ── Аудио-модель → Dynamic quantization ──────────────────────────────
    print("Optimizing audio model (INT8 quantization)...")
    audio_model = AudioEmotionModel(
        num_classes=cfg['emotions']['num_classes'],
        model_name=cfg['audio']['model_name'],
        dropout=0.0,
    )
    audio_model.load_state_dict(
        torch.load(cfg['paths']['audio_model_ckpt'], map_location='cpu')
    )
    audio_model.eval()
    quantized_audio = quantize_audio_model(audio_model)
    torch.save(quantized_audio.state_dict(), str(out_dir / 'audio_quantized.pt'))

    # Размер до/после
    orig_size = Path(cfg['paths']['audio_model_ckpt']).stat().st_size / 1e6
    print(f"  Original:   {orig_size:.0f} MB")
    print(f"  Quantized:  ~{orig_size / 4:.0f} MB (оценка)")
    print(f"  → Saved: {out_dir / 'audio_quantized.pt'}")

    print("\nDone. Use checkpoints/optimized/ for deployment.")


if __name__ == '__main__':
    main()
