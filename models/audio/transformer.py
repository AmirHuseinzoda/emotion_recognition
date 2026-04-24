"""
Аудио-модель: HuBERT base (или wav2vec 2.0) + классификационная голова.

Pipeline:
  raw waveform (T,) → HuBERT → (T', 768) → mean pool → (768,)
                   → MLP head → (num_classes,)

Стратегия fine-tuning:
  1. Заморозить feature_extractor (CNN-часть HuBERT) — всегда.
  2. На первых эпохах обучать только head.
  3. После unfreeze() разморозить трансформерные слои.
"""

import torch
import torch.nn as nn
from transformers import HubertModel, AutoFeatureExtractor


class AudioEmotionModel(nn.Module):
    """
    HuBERT-base с классификационной головой для распознавания эмоций.

    Args:
        num_classes:            количество классов
        model_name:             HuggingFace checkpoint
        dropout:                dropout в MLP-голове
        freeze_feature_encoder: заморозить CNN-часть HuBERT (рекомендуется)
        freeze_transformer:     заморозить трансформерные слои (для разогрева)
    """

    def __init__(
        self,
        num_classes: int = 8,
        model_name: str = "facebook/hubert-base-ls960",
        dropout: float = 0.3,
        freeze_feature_encoder: bool = True,
        freeze_transformer: bool = False,
    ):
        super().__init__()
        self.hubert = HubertModel.from_pretrained(model_name)
        self.hidden_size = self.hubert.config.hidden_size  # 768

        if freeze_feature_encoder:
            self.hubert.feature_extractor._freeze_parameters()

        if freeze_transformer:
            for p in self.hubert.encoder.parameters():
                p.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, waveform: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            waveform:       (B, T) — нормализованный raw waveform 16kHz
            attention_mask: (B, T) — маска паддинга (опционально)
        Returns:
            logits: (B, num_classes)
        """
        outputs = self.hubert(
            input_values=waveform,
            attention_mask=attention_mask,
        )
        # (B, T', hidden_size) → mean pool → (B, hidden_size)
        hidden = outputs.last_hidden_state
        if attention_mask is not None:
            # Маскированное среднее
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = hidden.mean(dim=1)

        return self.classifier(pooled)

    def unfreeze_transformer(self):
        """Размораживает трансформерные слои для полного fine-tuning."""
        for p in self.hubert.encoder.parameters():
            p.requires_grad = True


def get_feature_extractor(model_name: str = "facebook/hubert-base-ls960"):
    """Возвращает HuggingFace feature extractor для нормализации waveform."""
    return AutoFeatureExtractor.from_pretrained(model_name)
