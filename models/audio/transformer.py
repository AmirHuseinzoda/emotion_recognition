"""
Аудио-модель: WavLM-Large (или HuBERT) + классификационная голова.

Pipeline:
  raw waveform (T,) → WavLM encoder → (T', H) → AttentiveMeanPool → (H,)
                    → LayerNorm → Linear(H, 256) → GELU → Linear(256, C)

Улучшения vs HuBERT-base:
  - WavLM-Large (H=1024, 24 layers) → +5-8% SER по литературе
  - AttentiveMeanPool: обучаемые веса внимания вместо простого среднего
  - GELU + LayerNorm в голове для устойчивости обучения
  - AutoModel: совместим с любым wav2vec/HuBERT/WavLM чекпоинтом

ВАЖНО: checkpoint HuBERT (state_dict с ключами hubert.*) несовместим
       с этой версией (ключи model.*). Требует переобучения.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoFeatureExtractor


class AttentiveMeanPool(nn.Module):
    """
    Обучаемое взвешенное усреднение по временной оси.
    Фокусируется на наиболее эмоционально насыщенных фреймах.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x:    (B, T, H) — последовательность скрытых состояний
            mask: (B, T)    — маска паддинга (1=реальный токен, 0=паддинг)
        Returns:
            pooled: (B, H)
        """
        w = self.attn(x).squeeze(-1)                    # (B, T)
        if mask is not None:
            w = w.masked_fill(mask == 0, float('-inf'))
        w = F.softmax(w, dim=-1).unsqueeze(-1)          # (B, T, 1)
        return (x * w).sum(dim=1)                       # (B, H)


class AudioEmotionModel(nn.Module):
    """
    WavLM-Large (или любой AutoModel) + AttentiveMeanPool + MLP-голова.

    Args:
        num_classes:            количество классов
        model_name:             HuggingFace checkpoint (WavLM, HuBERT, wav2vec2)
        dropout:                dropout в MLP-голове
        freeze_feature_encoder: заморозить CNN-часть (рекомендуется)
        freeze_transformer:     заморозить трансформерные слои (для разогрева головы)
    """

    def __init__(
        self,
        num_classes: int = 8,
        model_name: str = "microsoft/wavlm-large",
        dropout: float = 0.3,
        freeze_feature_encoder: bool = True,
        freeze_transformer: bool = False,
    ):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.model.config.hidden_size  # 1024 for WavLM-Large

        if freeze_feature_encoder:
            self.model.feature_extractor._freeze_parameters()

        if freeze_transformer:
            for p in self.model.encoder.parameters():
                p.requires_grad = False

        self.pool = AttentiveMeanPool(self.hidden_size)

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def embed(self, waveform: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """(B, T) -> (B, hidden_size) — пул-эмбеддинг перед классификатором."""
        outputs = self.model(input_values=waveform, attention_mask=attention_mask)
        return self.pool(outputs.last_hidden_state, attention_mask)

    def forward(self, waveform: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            waveform:       (B, T) — нормализованный raw waveform 16kHz
            attention_mask: (B, T) — маска паддинга (опционально)
        Returns:
            logits: (B, num_classes)
        """
        return self.classifier(self.embed(waveform, attention_mask))

    def unfreeze_transformer(self):
        """Размораживает трансформерные слои для полного fine-tuning."""
        for p in self.model.encoder.parameters():
            p.requires_grad = True


def get_feature_extractor(model_name: str = "microsoft/wavlm-large"):
    return AutoFeatureExtractor.from_pretrained(model_name)
