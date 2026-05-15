"""
Распознавание эмоций в реальном времени с веб-камеры и микрофона.

Запуск:
  python inference/realtime.py --video_ckpt checkpoints/video_best.pt \
                                --audio_ckpt checkpoints/audio_best.pt \
                                --fusion_ckpt checkpoints/fusion_best.pt
"""

import argparse
import sys
import time
import threading
import queue
from pathlib import Path
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.video.backbone import VideoEmotionModel
from models.audio.transformer import AudioEmotionModel
from models.fusion.fusion import FusionModel
from datasets.ravdess import IDX_TO_LABEL_6 as IDX_TO_LABEL

import yaml

try:
    from facenet_pytorch import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False
    print("[WARN] facenet-pytorch not installed. Face detection disabled.")

try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("[WARN] sounddevice not installed. Audio disabled.")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

COLORS = {
    'neutral':   (200, 200, 200),
    'calm':      (150, 200, 150),
    'happy':     (0,   200,   0),
    'sad':       (200, 100,   0),
    'angry':     (0,     0, 255),
    'fearful':   (200,   0, 200),
    'disgust':   (0,   150, 150),
    'surprised': (0,   200, 255),
}

TRANSFORM = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class EmotionInference:
    def __init__(self, cfg, video_ckpt, audio_ckpt, fusion_ckpt, fusion_type='attention'):
        self.cfg = cfg
        self.window_frames = cfg['video']['window_frames']
        self.sample_rate   = cfg['audio']['sample_rate']
        self.num_classes   = cfg['emotions']['num_classes']

        tcn_cfg = cfg['video']['tcn']

        # Видео-модель
        self.video_model = VideoEmotionModel(
            num_classes=self.num_classes,
            num_channels=tcn_cfg['num_channels'],
            num_levels=tcn_cfg['num_levels'],
            kernel_size=tcn_cfg['kernel_size'],
            dropout=0.0,
        )
        self.video_model.load_state_dict(torch.load(video_ckpt, map_location='cpu'))
        self.video_model.eval().to(DEVICE)

        # Аудио-модель
        self.audio_model = AudioEmotionModel(
            num_classes=self.num_classes,
            model_name=cfg['audio']['model_name'],
            dropout=0.0,
        )
        self.audio_model.load_state_dict(torch.load(audio_ckpt, map_location='cpu'))
        self.audio_model.eval().to(DEVICE)

        # Fusion
        self.fusion_model = FusionModel(
            video_model=self.video_model,
            audio_model=self.audio_model,
            fusion_type=fusion_type,
            num_classes=self.num_classes,
            video_embed_dim=tcn_cfg['num_channels'],
            audio_embed_dim=cfg['audio']['hidden_size'],
            hidden_dim=cfg['fusion']['hidden_dim'],
        )
        self.fusion_model.load_state_dict(torch.load(fusion_ckpt, map_location='cpu'))
        self.fusion_model.eval().to(DEVICE)

        # Детектор лиц
        # image_size + margin воспроизводят выравнивание лица как в preprocess_video.py
        self.detector = MTCNN(image_size=112, margin=20, keep_all=False,
                              post_process=False, device=str(DEVICE)) if MTCNN_AVAILABLE else None

        # Буферы
        self.frame_buffer = deque(maxlen=self.window_frames)
        self.audio_buffer  = deque(maxlen=self.sample_rate * 3)  # 3 секунды

        self.last_result = {'emotion': 'neutral', 'probs': {}}

    @torch.no_grad()
    def predict_video(self) -> dict:
        if len(self.frame_buffer) < self.window_frames:
            return {}

        frames = torch.stack(list(self.frame_buffer)).unsqueeze(0).to(DEVICE)  # (1, T, C, H, W)
        logits = self.video_model(frames)
        probs  = F.softmax(logits[0], dim=0).cpu().numpy()
        return {IDX_TO_LABEL[i]: float(probs[i]) for i in range(self.num_classes)}

    @torch.no_grad()
    def predict_audio(self) -> dict:
        if not AUDIO_AVAILABLE or len(self.audio_buffer) < self.sample_rate:
            return {}

        audio_np = np.array(list(self.audio_buffer), dtype=np.float32)
        max_len  = int(self.cfg['audio']['max_duration_sec'] * self.sample_rate)
        if len(audio_np) > max_len:
            audio_np = audio_np[-max_len:]
        else:
            audio_np = np.pad(audio_np, (0, max_len - len(audio_np)))

        waveform = torch.tensor(audio_np).unsqueeze(0).to(DEVICE)  # (1, T)
        logits   = self.audio_model(waveform)
        probs    = F.softmax(logits[0], dim=0).cpu().numpy()
        return {IDX_TO_LABEL[i]: float(probs[i]) for i in range(self.num_classes)}

    def process_frame(self, frame_bgr: np.ndarray):
        """Добавляет кадр в буфер после детекции лица."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        if self.detector is not None:
            face = self.detector(pil)
            if face is None:
                return None, None
            # face: (C, H, W) float32 [0,255] tensor из MTCNN (post_process=False)
            face_tensor = face.permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()
            face_pil = Image.fromarray(face_tensor)
            tensor = TRANSFORM(face_pil)
        else:
            tensor = TRANSFORM(pil)
            face_tensor = None

        self.frame_buffer.append(tensor)
        return face_tensor, tensor

    def add_audio_chunk(self, chunk: np.ndarray):
        self.audio_buffer.extend(chunk.flatten().tolist())

    def draw_overlay(self, frame: np.ndarray, probs: dict) -> np.ndarray:
        if not probs:
            return frame

        top_emotion = max(probs, key=probs.get)
        color = COLORS.get(top_emotion, (255, 255, 255))

        # Полупрозрачная панель
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (260, 200), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        cv2.putText(frame, f"Emotion: {top_emotion.upper()}",
                    (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        y = 55
        for emotion, prob in sorted(probs.items(), key=lambda x: -x[1]):
            bar_w = int(prob * 150)
            bar_color = COLORS.get(emotion, (128, 128, 128))
            cv2.rectangle(frame, (15, y), (15 + bar_w, y + 12), bar_color, -1)
            cv2.putText(frame, f"{emotion[:7]:7s} {prob:.2f}",
                        (170, y + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)
            y += 16

        return frame


def run(cfg, video_ckpt, audio_ckpt, fusion_ckpt, camera_idx=0):
    engine = EmotionInference(cfg, video_ckpt, audio_ckpt, fusion_ckpt)

    # Аудио поток
    if AUDIO_AVAILABLE:
        def audio_callback(indata, frames, time_info, status):
            engine.add_audio_chunk(indata[:, 0])

        stream = sd.InputStream(
            samplerate=cfg['audio']['sample_rate'],
            channels=1,
            callback=audio_callback,
            blocksize=1024,
        )
        stream.start()

    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print(f"[ERROR] Camera not found. Trying indices 0-3...")
        for idx in range(4):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                print(f"  Found camera at index {idx}")
                break
        else:
            print("[ERROR] No camera found. Exiting.")
            return
    print("Running... Press 'q' to quit.")

    probs_video: dict = {}
    probs_audio: dict = {}
    lock = threading.Lock()
    stop_event = threading.Event()

    # Видео-инференс в отдельном потоке — не блокирует отображение
    def video_worker():
        while not stop_event.is_set():
            result = engine.predict_video()
            if result:
                with lock:
                    probs_video.clear()
                    probs_video.update(result)
            time.sleep(0.05)  # ~20 предсказаний/сек максимум

    # Аудио-инференс в отдельном потоке
    def audio_worker():
        while not stop_event.is_set():
            result = engine.predict_audio()
            if result:
                with lock:
                    probs_audio.clear()
                    probs_audio.update(result)
            time.sleep(0.5)  # аудио обновляется каждые 500 мс

    threading.Thread(target=video_worker, daemon=True).start()
    threading.Thread(target=audio_worker, daemon=True).start()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        face_np, _ = engine.process_frame(frame)

        # Главный поток только рисует — никогда не блокируется
        with lock:
            display_probs = dict(probs_video) if probs_video else dict(probs_audio)

        frame = engine.draw_overlay(frame, display_probs)

        if face_np is not None:
            face_small = cv2.resize(face_np, (80, 80))
            face_bgr   = cv2.cvtColor(face_small, cv2.COLOR_RGB2BGR)
            frame[10:90, -90:-10] = face_bgr

        cv2.imshow('Emotion Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stop_event.set()
    cap.release()
    cv2.destroyAllWindows()
    if AUDIO_AVAILABLE:
        stream.stop()


def run_audio_only(cfg, audio_ckpt):
    """Режим без камеры: только аудио-модель, вывод в терминал."""
    if not AUDIO_AVAILABLE:
        print("[ERROR] sounddevice не установлен.")
        return

    audio_cfg = cfg['audio']
    sample_rate = audio_cfg['sample_rate']
    max_len = int(audio_cfg['max_duration_sec'] * sample_rate)

    model = AudioEmotionModel(
        num_classes=cfg['emotions']['num_classes'],
        model_name=audio_cfg['model_name'],
        dropout=0.0,
    )
    model.load_state_dict(torch.load(audio_ckpt, map_location='cpu'))
    model.eval().to(DEVICE)
    print(f"Audio model loaded: {audio_ckpt}")

    audio_buffer = deque(maxlen=max_len)
    stop_event = threading.Event()

    def audio_callback(indata, frames, time_info, status):
        audio_buffer.extend(indata[:, 0].tolist())

    stream = sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        callback=audio_callback,
        blocksize=1024,
    )
    stream.start()
    print("Listening... Press Ctrl+C to stop.\n")

    BAR = 20
    EMOTION_ICONS = {
        'neutral': '😐', 'calm': '😌', 'happy': '😄',
        'sad': '😢', 'angry': '😠', 'fearful': '😨',
    }

    try:
        while not stop_event.is_set():
            time.sleep(0.5)
            if len(audio_buffer) < sample_rate:
                print("  Listening...", end='\r')
                continue

            audio_np = np.array(list(audio_buffer), dtype=np.float32)
            if len(audio_np) < max_len:
                audio_np = np.pad(audio_np, (0, max_len - len(audio_np)))
            else:
                audio_np = audio_np[-max_len:]

            waveform = torch.tensor(audio_np).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                probs = F.softmax(model(waveform)[0], dim=0).cpu().numpy()

            num_classes = cfg['emotions']['num_classes']
            sorted_idx = probs.argsort()[::-1]
            top = IDX_TO_LABEL[sorted_idx[0]]

            print(f"\n{'─'*35}")
            print(f"  {EMOTION_ICONS.get(top, '?')}  {top.upper():<10}  ({probs[sorted_idx[0]]:.0%})")
            print(f"{'─'*35}")
            for i in sorted_idx:
                name = IDX_TO_LABEL[i]
                p = probs[i]
                bar = '█' * int(p * BAR) + '░' * (BAR - int(p * BAR))
                print(f"  {name:<8} {bar} {p:.2f}")

    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
        print("\nStopped.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',      default='configs/config.yaml')
    parser.add_argument('--video_ckpt',  default='checkpoints/video_best.pt')
    parser.add_argument('--audio_ckpt',  default='checkpoints/audio_best.pt')
    parser.add_argument('--fusion_ckpt', default='checkpoints/fusion_best.pt')
    parser.add_argument('--camera',      type=int, default=0, help='Camera index')
    parser.add_argument('--audio_only',  action='store_true', help='Run without camera')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.audio_only:
        run_audio_only(cfg, args.audio_ckpt)
    else:
        run(cfg, args.video_ckpt, args.audio_ckpt, args.fusion_ckpt)


if __name__ == '__main__':
    main()
