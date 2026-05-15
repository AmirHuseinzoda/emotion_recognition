"""
FastAPI REST API для распознавания эмоций.

Эндпоинты:
  POST /analyze  — анализ видео + аудио файлов
  GET  /health   — статус сервера

Запуск:
  uvicorn inference.api:app --host 0.0.0.0 --port 8000 --reload
"""

import io
import sys
import time
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import cv2
from PIL import Image
from torchvision import transforms
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.video.backbone import VideoEmotionModel
from models.audio.transformer import AudioEmotionModel
from models.fusion.fusion import FusionModel
from datasets.ravdess import IDX_TO_LABEL_6 as IDX_TO_LABEL

try:
    from facenet_pytorch import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Загрузка конфига и моделей ─────────────────────────────────────────────

with open('configs/config.yaml') as f:
    CFG = yaml.safe_load(f)

NUM_CLASSES   = CFG['emotions']['num_classes']
WINDOW_FRAMES = CFG['video']['window_frames']
SAMPLE_RATE   = CFG['audio']['sample_rate']
MAX_LEN       = int(CFG['audio']['max_duration_sec'] * SAMPLE_RATE)
TCN_CFG       = CFG['video']['tcn']

TRANSFORM = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

detector = MTCNN(image_size=112, margin=20, keep_all=False,
                 post_process=False, device=str(DEVICE)) if MTCNN_AVAILABLE else None

video_model = VideoEmotionModel(
    num_classes=NUM_CLASSES,
    num_channels=TCN_CFG['num_channels'],
    num_levels=TCN_CFG['num_levels'],
    kernel_size=TCN_CFG['kernel_size'],
    dropout=0.0,
)
video_model.load_state_dict(
    torch.load(CFG['paths']['video_model_ckpt'], map_location='cpu')
)
video_model.eval().to(DEVICE)

audio_model = AudioEmotionModel(
    num_classes=NUM_CLASSES,
    model_name=CFG['audio']['model_name'],
    dropout=0.0,
)
audio_model.load_state_dict(
    torch.load(CFG['paths']['audio_model_ckpt'], map_location='cpu')
)
audio_model.eval().to(DEVICE)

fusion_model = FusionModel(
    video_model=video_model,
    audio_model=audio_model,
    fusion_type=CFG['fusion']['method'],
    num_classes=NUM_CLASSES,
    video_embed_dim=TCN_CFG['num_channels'],
    audio_embed_dim=CFG['audio']['hidden_size'],
    hidden_dim=CFG['fusion']['hidden_dim'],
)
fusion_model.load_state_dict(
    torch.load(CFG['paths']['fusion_model_ckpt'], map_location='cpu')
)
fusion_model.eval().to(DEVICE)

# ── Вспомогательные функции ────────────────────────────────────────────────

def extract_face_frames(video_path: str) -> Optional[torch.Tensor]:
    """Возвращает (T, C, H, W) тензор или None."""
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened() and len(frames) < WINDOW_FRAMES * 4:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        if detector is not None:
            face = detector(pil)
            if face is None:
                continue
            face_np = face.permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()
            face_pil = Image.fromarray(face_np)
        else:
            face_pil = pil

        frames.append(TRANSFORM(face_pil))

    cap.release()
    if not frames:
        return None

    # Выбираем равномерно WINDOW_FRAMES кадров
    indices = np.linspace(0, len(frames) - 1, WINDOW_FRAMES, dtype=int)
    seq = torch.stack([frames[i] for i in indices])  # (T, C, H, W)
    return seq


def load_audio(audio_path: str) -> torch.Tensor:
    """Возвращает нормализованный waveform (T,)."""
    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    waveform = waveform / (waveform.abs().max() + 1e-8)
    length = waveform.shape[1]
    if length < MAX_LEN:
        waveform = F.pad(waveform, (0, MAX_LEN - length))
    else:
        waveform = waveform[:, :MAX_LEN]
    return waveform.squeeze(0)


def probs_dict(logits: torch.Tensor) -> dict:
    probs = F.softmax(logits.squeeze(0), dim=0).cpu().numpy()
    return {IDX_TO_LABEL[i]: round(float(probs[i]), 4) for i in range(NUM_CLASSES)}


# ── FastAPI app ────────────────────────────────────────────────────────────

app = FastAPI(
    title="Emotion Recognition API",
    description="Мультимодальное распознавание эмоций (видео + аудио)",
    version="1.0.0",
)


@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE)}


@app.post("/analyze")
async def analyze(
    video: UploadFile = File(..., description="Видеофайл (.mp4, .avi, .mov)"),
    audio: Optional[UploadFile] = File(None, description="Аудиофайл (.wav, .mp3) — если не указан, извлекается из видео"),
):
    t0 = time.time()
    result = {}

    # Сохраняем файлы во временные директории
    with tempfile.NamedTemporaryFile(suffix=Path(video.filename).suffix, delete=False) as tmp_vid:
        tmp_vid.write(await video.read())
        video_path = tmp_vid.name

    audio_path = None
    if audio is not None:
        with tempfile.NamedTemporaryFile(suffix=Path(audio.filename).suffix, delete=False) as tmp_aud:
            tmp_aud.write(await audio.read())
            audio_path = tmp_aud.name

    try:
        # ── Видео ────────────────────────────────────────────────────────
        seq = extract_face_frames(video_path)
        if seq is None:
            raise HTTPException(status_code=422, detail="Лицо не обнаружено в видео.")

        frames_t = seq.unsqueeze(0).to(DEVICE)   # (1, T, C, H, W)
        with torch.no_grad():
            logits_v = video_model(frames_t)
        result['emotion_video'] = probs_dict(logits_v)

        # ── Аудио ────────────────────────────────────────────────────────
        aud_src = audio_path if audio_path else video_path
        try:
            wav = load_audio(aud_src).unsqueeze(0).to(DEVICE)  # (1, T)
            with torch.no_grad():
                logits_a = audio_model(wav)
            result['emotion_audio'] = probs_dict(logits_a)
        except Exception:
            logits_a = logits_v   # fallback — используем видео
            result['emotion_audio'] = result['emotion_video']

        # ── Fusion ───────────────────────────────────────────────────────
        with torch.no_grad():
            out = fusion_model(frames_t, wav)
        result['emotion_combined'] = probs_dict(out['logits_fusion'])
        result['predicted_emotion'] = max(result['emotion_combined'],
                                          key=result['emotion_combined'].get)
        result['latency_ms'] = round((time.time() - t0) * 1000, 1)

    finally:
        Path(video_path).unlink(missing_ok=True)
        if audio_path:
            Path(audio_path).unlink(missing_ok=True)

    return JSONResponse(content=result)
