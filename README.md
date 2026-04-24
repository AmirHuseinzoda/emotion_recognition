# Распознавание эмоций человека средствами глубокого обучения

Выпускная квалификационная работа — СПбПУ Политех, ИКНиК.

Мультимодальная система распознавания эмоций по видео и аудио в реальном времени. Видео-ветвь использует EfficientNet-B0 + TCN, аудио-ветвь — HuBERT, слияние модальностей — cross-modal attention fusion.

---

## Архитектура

```
Видеопоток  ──► EfficientNet-B0 ──► TCN ──────────────┐
                (AffectNet веса)   (temporal)           │
                                                  Cross-modal
                                                  Attention ──► 6 классов
Аудиопоток  ──► HuBERT base ──► Mean Pool ────────────┘
                (HuggingFace)
```

**6 классов эмоций:** neutral · calm · happy · sad · angry · fearful

---

## Результаты

| Модель | Val F1 | Test F1 | Test Acc |
|---|---|---|---|
| Видео (EfficientNet-B0 + TCN) | 0.636 | **0.694** | 0.701 |
| Fusion (Video + HuBERT + Attention) | **0.824** | — | — |

Confusion matrix fusion (val, 144 сэмпла):

```
          neutral  calm  happy  sad  angry  fearful
neutral        21     0      3    0      0        0
calm            0    21      1    1      1        0
happy           6     0     17    0      1        0
sad             0     1      0   22      1        0
angry           0     2      6    1     14        1
fearful         0     0      0    0      0       24
```

---

## Датасеты

| Датасет | Описание | Использование |
|---|---|---|
| [RAVDESS](https://zenodo.org/record/1188976) | 24 актёра, 1440 аудиовизуальных записей | train / val / test (по актёрам) |
| [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) | 91 актёр, 7442 видеозаписи | train |

Разбивка RAVDESS: актёры 1–18 → train, 19–21 → val, 22–24 → test.

---

## Структура проекта

```
emotion_recognition/
├── configs/
│   └── config.yaml          # все гиперпараметры
├── datasets/
│   └── ravdess.py           # Dataset классы для RAVDESS и CREMA-D
├── models/
│   ├── video/
│   │   ├── backbone.py      # EfficientNet-B0 + TCN обёртка
│   │   └── tcn.py           # Temporal Convolutional Network
│   ├── audio/
│   │   └── transformer.py   # HuBERT + MLP-голова
│   └── fusion/
│       └── fusion.py        # WeightedFusion / CrossModalAttentionFusion
├── training/
│   ├── train_video.py
│   ├── train_audio.py
│   └── train_fusion.py
├── inference/
│   ├── realtime.py          # инференс с камеры и микрофона
│   ├── api.py               # FastAPI REST + WebSocket
│   └── optimize.py          # экспорт / квантизация
├── scripts/
│   ├── make_splits.py       # формирование train/val/test CSV
│   ├── preprocess_video.py  # извлечение лиц → .npy
│   ├── preprocess_audio.py  # ресэмплинг аудио → .npy
│   └── eval_video.py        # оценка видео-модели на тесте
├── checkpoints/             # сохранённые веса (не в репозитории)
├── data/                    # датасеты (не в репозитории)
└── requirements.txt
```

---

## Установка

```bash
# 1. Клонировать репозиторий
git clone https://github.com/<your-username>/emotion_recognition.git
cd emotion_recognition

# 2. Создать окружение
conda create -n diplomenv python=3.12 -y
conda activate diplomenv

# 3. Установить PyTorch с CUDA 12.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 4. Установить зависимости
pip install -r requirements.txt
```

---

## Подготовка данных

```bash
# Сформировать сплиты (после скачивания RAVDESS и CREMA-D в data/raw/)
python scripts/make_splits.py

# Предобработка видео (детекция лиц, сохранение .npy)
python scripts/preprocess_video.py --raw_dir data/raw/RAVDESS \
                                    --out_dir data/processed/video

# Предобработка аудио
python scripts/preprocess_audio.py --raw_dir data/raw/RAVDESS \
                                    --out_dir data/processed/audio
```

---

## Обучение

```bash
# Видео-модель (EfficientNet-B0 + TCN, 50 эпох)
python training/train_video.py --config configs/config.yaml

# Аудио-модель (HuBERT, 30 эпох)
python training/train_audio.py --config configs/config.yaml

# Fusion (cross-modal attention, 20 эпох)
python training/train_fusion.py --config configs/config.yaml --fusion_type attention
```

---

## Оценка

```bash
# Тестовые метрики видео-модели
python scripts/eval_video.py --config configs/config.yaml
```

---

## Инференс

### Реальное время (камера + микрофон)

```bash
python inference/realtime.py --config configs/config.yaml
```

### Только аудио (без камеры)

```bash
python inference/realtime.py --audio_only
```

### REST API

```bash
uvicorn inference.api:app --host 0.0.0.0 --port 8000
```

`POST /analyze` — загрузить видео/аудио файл, получить эмоцию и вероятности.  
`GET /health` — статус сервера.

---

## Стек

- Python 3.12 · PyTorch 2.11.0 · CUDA 13.1
- HuggingFace Transformers (HuBERT)
- torchvision · OpenCV · librosa · scikit-learn
- FastAPI · Uvicorn
- NVIDIA RTX 5070 12 GB
