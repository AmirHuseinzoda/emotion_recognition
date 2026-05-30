# Мультимодальное распознавание эмоций

Выпускная квалификационная работа — СПбПУ Политехнический университет Петра Великого, ИКНиК.

Система распознаёт эмоции человека в реальном времени по видеопотоку с камеры и аудиопотоку с микрофона. Использует три модели: видео (EfficientNet-B0 + Temporal Transformer), аудио (WavLM-Large + Weighted Layer Aggregation) и их слияние через multi-layer cross-modal attention.

---

## Архитектура

```
Видеопоток  ──► EfficientNet-B0 ──► Temporal Transformer ────┐
                (HSEmotion/VGAF      (2-layer self-attention,│
                 pretrained)          CLS token pooling)     │
                                                      Multi-Layer
                                                      Cross-Modal
                                                      Attention ──► 6 эмоций
Аудиопоток  ──► WavLM-Large ──► Weighted Layer Agg. ─────────┘
                (microsoft/      (25 слоёв → обучаемое
                 wavlm-large)     взвешивание)
                              ──► AttentiveMeanPool
```

**6 классов эмоций:** `neutral` · `happy` · `sad` · `angry` · `fearful` · `disgust`

---

## Результаты

### Validation set (CREMA-D, 818 сэмплов)

| Модель | Accuracy | Macro F1 | UAR |
|---|---|---|---|
| Видео (EfficientNet-B0 + Temporal Transformer) | 0.6834 | 0.6799 | 0.6805 |
| Аудио (WavLM-Large + Layer Aggregation) | 0.8144 | 0.8113 | 0.8178 |
| **Fusion (Multi-Layer Cross-Modal Attention)** | **0.8778** | **0.8755** | **0.8794** |

### Test set (CREMA-D, 738 сэмплов)

| Модель | Accuracy | Macro F1 | UAR |
|---|---|---|---|
| Видео (EfficientNet-B0 + Temporal Transformer) | 0.6125 | 0.6143 | 0.6124 |
| Аудио (WavLM-Large + Layer Aggregation) | 0.7425 | 0.7480 | 0.7460 |
| **Fusion (Multi-Layer Cross-Modal Attention)** | **0.8089** | **0.8132** | **0.8095** |

Сплит: actor-independent (CREMA-D: актёры по группам; RAVDESS: актёры 1–18 train, 19–21 val, 22–24 test).

---

## Датасеты

Система обучается на двух датасетах: CREMA-D (основной) и RAVDESS (дополнительный).

### CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset)
- **Ссылка:** https://github.com/CheyneyComputerScience/CREMA-D
- 7442 видеозаписи (.flv), 91 актёр, 6 эмоций
- Аудио извлекается из видео скриптом `scripts/extract_cremad_audio.py`
- Скачать видеофайлы `.flv` из папки `VideoFlash/` (Google Drive в README репозитория)
- Распаковать в `data/raw/CREMA_D/`:
```
data/raw/CREMA_D/
    1001_DFA_ANG_XX.flv
    1001_DFA_DIS_XX.flv
    ...
    1091_WSI_SAD_XX.flv
```

### RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
- **Ссылка:** https://zenodo.org/record/1188976
- 1440 синхронных аудио (.wav) и видео (.mp4) записей, 24 актёра, 8 эмоций
- Маппинг 8 → 6 классов: calm → neutral, surprised исключён
- Скачать все 24 архива `Video_Speech_Actor_XX.zip`
- Распаковать в `data/raw/RAVDESS/`:
```
data/raw/RAVDESS/
    Actor_01/
        01-01-01-01-01-01-01.mp4
        03-01-01-01-01-01-01.wav
        ...
    Actor_02/
    ...
    Actor_24/
```

### Предобработка данных

**Видео:**
1. Детекция лиц алгоритмом MTCNN (facenet-pytorch) на каждом кадре
2. Аффинное выравнивание и кроп до 112x112 пикселей
3. Сохранение последовательности кадров в формате NumPy (T, 112, 112, 3) uint8
4. При обучении — скользящее окно 32 кадра (случайное начало), при eval — центральное

**Аудио:**
1. Загрузка .wav, преобразование в моно
2. Ресэмплинг до 16 кГц (librosa)
3. Нормализация по пиковому значению
4. Обрезка / паддинг до 5 секунд (80000 сэмплов)

**Аугментация (обучение):**
- Видео: горизонтальный flip, яркость/контраст, поворот ±15°, grayscale, random erasing, temporal dropout, гауссовский шум
- Аудио: volume jitter, гауссовский шум, temporal shift ±0.3с, time masking (SpecAugment)
- Mixup (alpha=0.4, применяется к 50% батчей) для обеих модальностей

---

## Архитектура моделей (детально)

### Видео-модель: EfficientNet-B0 + Temporal Transformer

| Компонент | Описание |
|---|---|
| Backbone | EfficientNet-B0 с предобученными весами HSEmotion/VGAF (VGGFace2 + AffectNet + AFEW) |
| Проекция | Linear(1280, 256) + LayerNorm + GELU |
| Temporal | 2-layer TransformerEncoder (8 голов, dim=256, FFN=1024, GELU, norm_first) |
| Pooling | CLS token |
| Classifier | Dropout → Linear(256, 6) |
| Обучение | Фаза 1: backbone заморожен (5 эпох, lr=1e-4). Фаза 2: full fine-tune (45 эпох, lr=1e-5) |
| Loss | FocalLoss (gamma=2.0, label_smoothing=0.1) + class weights [1.0, 0.7, 1.6, 1.3, 1.6, 0.7] |
| Данные | CREMA-D (.flv) + RAVDESS (.mp4), ~7600 train сэмплов |

### Аудио-модель: WavLM-Large + Weighted Layer Aggregation

| Компонент | Описание |
|---|---|
| Encoder | WavLM-Large (microsoft/wavlm-large, 317M параметров, 24 transformer-слоя, H=1024) |
| Layer Aggregation | Обучаемое взвешивание 25 скрытых состояний (embedding + 24 слоя) через softmax |
| Pooling | AttentiveMeanPool — обучаемые веса внимания по временной оси |
| Classifier | LayerNorm → Linear(1024, 256) → GELU → Dropout → Linear(256, 6) |
| Обучение | Фаза 1: transformer заморожен (7 эпох, lr=5e-4). Фаза 2: full fine-tune (23 эпохи, lr=5e-5) |
| Loss | FocalLoss (gamma=2.0, label_smoothing=0.1) + class weights [1.0, 0.8, 0.9, 0.9, 1.3, 1.3] |
| Данные | CREMA-D (.wav) + RAVDESS (.wav), ~6750 train сэмплов |

### Fusion: Multi-Layer Cross-Modal Attention

| Компонент | Описание |
|---|---|
| Проекция | Linear(256, 256) для видео, Linear(1024, 256) для аудио |
| Cross-Attention | 2 слоя CrossModalAttentionLayer: v←a и a←v (8 голов), FFN (256→1024→256), LayerNorm, residual |
| Modality Dropout | p=0.15 — случайное обнуление одной модальности при обучении |
| Classifier | Linear(512, 256) → GELU → Dropout → Linear(256, 6) |
| Обучение | Фаза 1: базовые модели заморожены, обучается только fusion (20 эпох, lr=1e-4) |
| Loss | FocalLoss (gamma=2.0, label_smoothing=0.1) |
| Auxiliary | При joint fine-tune: aux_loss_weight=0.3 для видео и аудио ветвей |

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
│   │   └── backbone.py          # EfficientNet-B0 + Temporal Transformer
│   ├── audio/
│   │   └── transformer.py       # WavLM-Large + Weighted Layer Aggregation
│   └── fusion/
│       └── fusion.py            # Multi-Layer CrossModalAttentionFusion
├── training/
│   ├── train_video.py           # двухфазное обучение видео-модели
│   ├── train_audio.py           # двухфазное обучение аудио-модели
│   ├── train_fusion.py          # обучение fusion (+ joint fine-tuning)
│   └── utils.py                 # FocalLoss, EarlyStopping, Mixup, WeightedRandomSampler
├── inference/
│   ├── realtime.py              # инференс с камеры и микрофона в реальном времени
│   ├── api.py                   # FastAPI REST API
│   └── optimize.py              # квантизация / экспорт
├── scripts/
│   ├── make_splits.py           # формирование train/val/test CSV по актёрам
│   ├── preprocess_video.py      # детекция лиц MTCNN → .npy последовательности
│   ├── extract_cremad_audio.py  # извлечение аудио из .flv видео CREMA-D
│   └── eval_video.py            # оценка видео-модели на test-сете
├── eval_test_all.py             # оценка всех трёх моделей (аудио, видео, fusion)
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

> Требования: GPU с 10+ GB VRAM для обучения (WavLM-Large). Для инференса достаточно 4 GB.

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

# Извлечь аудио из CREMA-D видео (.flv → .wav) через ffmpeg
python scripts/extract_cremad_audio.py

# Пересоздать сплиты с учётом извлечённого аудио
python scripts/make_splits.py \
    --cremad_audio_dir data/raw/CREMA_D_audio

# Предобработать видео: детекция лиц (MTCNN) → сохранить .npy
python scripts/preprocess_video.py
```

> **ffmpeg** должен быть установлен и доступен в PATH для извлечения аудио CREMA-D.

### Шаг 2 — Обучение

```bash
# Видео-модель (EfficientNet-B0 + Temporal Transformer, ~50 эпох)
python training/train_video.py --config configs/config.yaml

# Аудио-модель (WavLM-Large + Layer Aggregation, ~30 эпох)
# WavLM-Large (~1.2 GB) скачается автоматически с HuggingFace при первом запуске
python training/train_audio.py --config configs/config.yaml

# Fusion-модель (Multi-Layer Cross-Modal Attention, ~20 эпох)
python training/train_fusion.py --config configs/config.yaml --fusion_type attention
```

Чекпоинты сохраняются в `checkpoints/` автоматически при улучшении val F1.

### Шаг 3 — Оценка

```bash
# Все три модели на val или test
python eval_test_all.py --split test
python eval_test_all.py --split val
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
| Видео backbone | EfficientNet-B0 (timm) + HSEmotion/VGAF pretrained weights |
| Видео temporal | Temporal Transformer (2-layer self-attention, CLS token) |
| Аудио encoder | WavLM-Large (`microsoft/wavlm-large`, 317M params) |
| Аудио aggregation | Weighted Layer Aggregation (25 слоёв) + AttentiveMeanPool |
| Fusion | Multi-Layer Cross-Modal Attention (2 слоя, 8 голов) + Modality Dropout |
| Детекция лиц | MTCNN (facenet-pytorch) |
| Фреймворк | PyTorch 2.x + HuggingFace Transformers |
| Инференс API | FastAPI + Uvicorn |
| Loss / аугментация | FocalLoss, Mixup, SpecAugment, WeightedRandomSampler |

---

## Требования к оборудованию

| Задача | GPU VRAM |
|---|---|
| Обучение аудио (WavLM-Large, batch=8, layer_agg) | 12+ GB |
| Обучение видео (EfficientNet + Transformer, batch=12) | 6+ GB |
| Обучение fusion (оба backbone заморожены) | 12+ GB |
| Инференс (все три модели) | 4+ GB |
