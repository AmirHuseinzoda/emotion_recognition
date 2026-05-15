# Мультимодальное распознавание эмоций

Выпускная квалификационная работа — СПбПУ Политехнический университет Петра Великого, ИКНиК.

Система распознаёт эмоции человека в реальном времени по видеопотоку с камеры и аудиопотоку с микрофона. Использует три модели: видео (EfficientNet-B0 + Bi-LSTM), аудио (WavLM-Large) и их слияние через cross-modal attention.

---

## Архитектура

```
Видеопоток  ──► EfficientNet-B0 ──► Bi-LSTM ─────────────┐
                (HSEmotion /        (attention             │
                 AffectNet веса)     pooling)              │
                                                   Cross-Modal
                                                   Attention ──► 6 эмоций
Аудиопоток  ──► WavLM-Large ──► AttentiveMeanPool ────────┘
                (microsoft/
                 wavlm-large)
```

**6 классов эмоций:** `neutral` · `happy` · `sad` · `angry` · `fearful` · `disgust`

---

## Результаты

| Модель | Val F1 | Val UAR |
|---|---|---|
| Видео (EfficientNet-B0 + Bi-LSTM) | 0.62 | — |
| Аудио (WavLM-Large) | 0.78 | — |
| **Fusion (Cross-Modal Attention)** | **0.85** | **—** |

Сплит: actor-independent (RAVDESS актёры 1–18 train, 19–21 val, 22–24 test).

---

## Датасеты

### RAVDESS
- **Ссылка:** https://zenodo.org/record/1188976
- Скачать все 24 архива `Video_Speech_Actor_XX.zip`
- Распаковать в `data/raw/RAVDESS/`:
```
data/raw/RAVDESS/
    Actor_01/
        03-01-01-01-01-01-01.mp4
        03-01-01-01-01-01-01.wav
        ...
    Actor_02/
    ...
    Actor_24/
```

### CREMA-D
- **Ссылка:** https://github.com/CheyneyComputerScience/CREMA-D
- Скачать видеофайлы `.flv` из папки `VideoFlash/` (Google Drive в README репозитория)
- Распаковать в `data/raw/CREMA_D/`:
```
data/raw/CREMA_D/
    1001_DFA_ANG_XX.flv
    1001_DFA_DIS_XX.flv
    ...
    1091_WSI_SAD_XX.flv
```

---

## Структура проекта

```
emotion_recognition/
├── configs/
│   └── config.yaml              # все гиперпараметры
├── datasets/
│   └── ravdess.py               # Dataset классы, парсинг RAVDESS и CREMA-D
├── models/
│   ├── video/
│   │   └── backbone.py          # EfficientNet-B0 + Bi-LSTM (TemporalLSTM)
│   ├── audio/
│   │   └── transformer.py       # WavLM-Large + AttentiveMeanPool + MLP
│   └── fusion/
│       └── fusion.py            # CrossModalAttentionFusion / WeightedFusion
├── training/
│   ├── train_video.py           # двухфазное обучение видео-модели
│   ├── train_audio.py           # двухфазное обучение аудио-модели
│   ├── train_fusion.py          # обучение fusion (+ joint fine-tuning)
│   └── utils.py                 # FocalLoss, EarlyStopping, Mixup, Sampler
├── inference/
│   ├── realtime.py              # инференс с камеры и микрофона в реальном времени
│   ├── api.py                   # FastAPI REST API
│   └── optimize.py              # квантизация / экспорт
├── scripts/
│   ├── make_splits.py           # формирование train/val/test CSV по актёрам
│   ├── preprocess_video.py      # детекция лиц MTCNN → .npy последовательности
│   ├── extract_cremad_audio.py  # извлечение аудио из .flv видео CREMA-D
│   └── eval_video.py            # оценка видео-модели на test-сете
├── eval_audio.py                # оценка аудио-модели на test-сете
├── eval_fusion.py               # оценка fusion-модели на test-сете
├── checkpoints/                 # веса моделей (не в репозитории)
├── data/                        # датасеты (не в репозитории)
└── requirements.txt
```

---

## Установка

```bash
# 1. Клонировать репозиторий
git clone https://github.com/AmirHuseinzoda/emotion_recognition.git
cd emotion_recognition

# 2. Создать conda-окружение
conda create -n diplomenv python=3.12 -y
conda activate diplomenv

# 3. Установить PyTorch с CUDA 12.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 4. Установить остальные зависимости
pip install -r requirements.txt
```

> Требования: GPU с 8+ GB VRAM для обучения (WavLM-Large). Для инференса достаточно 4 GB.

---

## Полный пайплайн

### Шаг 1 — Подготовить данные

```bash
# Сформировать train/val/test сплиты по актёрам
python scripts/make_splits.py \
    --raw_dir data/raw/RAVDESS \
    --cremad_dir data/raw/CREMA_D \
    --cremad_audio_dir data/raw/CREMA_D_audio \
    --splits_dir data/splits

# Извлечь аудио из CREMA-D видео (.flv → .wav) через ffmpeg (~30 мин)
python scripts/extract_cremad_audio.py

# Пересоздать сплиты с учётом извлечённого аудио
python scripts/make_splits.py \
    --cremad_audio_dir data/raw/CREMA_D_audio

# Предобработать видео: детекция лиц (MTCNN) → сохранить .npy (~1-2 ч)
python scripts/preprocess_video.py
```

> **ffmpeg** должен быть установлен и доступен в PATH для извлечения аудио CREMA-D.

### Шаг 2 — Обучение

```bash
# Видео-модель (EfficientNet-B0 + Bi-LSTM, ~50 эпох, ~2 ч)
python training/train_video.py --config configs/config.yaml

# Аудио-модель (WavLM-Large, ~30 эпох, ~4-6 ч)
# WavLM-Large (~1.2 GB) скачается автоматически с HuggingFace при первом запуске
python training/train_audio.py --config configs/config.yaml

# Fusion-модель (cross-modal attention, ~20 эпох, ~30 мин)
python training/train_fusion.py --config configs/config.yaml --fusion_type attention
```

Чекпоинты сохраняются в `checkpoints/` автоматически при улучшении val F1.

### Шаг 3 — Оценка на тест-сете

```bash
python scripts/eval_video.py
python eval_audio.py
python eval_fusion.py
```

---

## Инференс в реальном времени

```bash
# Камера + микрофон (fusion-модель, основной режим)
python inference/realtime.py

# Только аудио (без камеры, терминальный вывод)
python inference/realtime.py --audio_only

# Другой индекс камеры
python inference/realtime.py --camera 1
```

Нажмите `q` в окне камеры для выхода.

---

## REST API

```bash
# Запуск сервера
uvicorn inference.api:app --host 0.0.0.0 --port 8000

# Проверка
curl http://localhost:8000/health
```

**Эндпоинты:**
- `POST /analyze` — загрузить видеофайл (`.mp4`, `.avi`, `.mov`), получить эмоцию и вероятности по каждому классу
- `GET /health` — статус сервера и устройство (CPU/GPU)

---

## Стек

| Компонент | Библиотека/модель |
|---|---|
| Видео-бэкбон | EfficientNet-B0 (timm) + HSEmotion/AffectNet веса |
| Видео-темпорал | Bidirectional LSTM с attention pooling |
| Аудио-бэкбон | WavLM-Large (`microsoft/wavlm-large`) |
| Детекция лиц | MTCNN (facenet-pytorch) |
| Фреймворк | PyTorch 2.x + HuggingFace Transformers |
| Инференс API | FastAPI + Uvicorn |
| Аугментация | Mixup, FocalLoss, SpecAugment, WeightedRandomSampler |

---

## Требования к оборудованию

| Задача | GPU VRAM |
|---|---|
| Обучение аудио (WavLM-Large, batch=8) | 10+ GB |
| Обучение видео (EfficientNet, batch=12) | 6+ GB |
| Инференс (все три модели) | 4+ GB |
