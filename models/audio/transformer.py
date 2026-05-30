"""
Аудио-модель: WavLM-Large + Weighted Layer Aggregation + классификационная голова.

Pipeline:
  raw waveform (B,T) → WavLM-Large encoder → 25 hidden states (embedding + 24 layers)
  → Weighted Layer Aggregation (обучаемые веса по слоям) → (B,T',1024)
  → AttentiveMeanPool → (B,1024) → LayerNorm → Linear(1024,256) → GELU → Linear(256,C)

Weighted Layer Aggregation извлекает разноуровневую информацию:
нижние слои — акустика (тон, громкость), верхние — семантика/эмоции.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoFeatureExtractor


class AttentiveMeanPool(nn.Module):
    """Обучаемое взвешенное усреднение по временной оси."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        w = self.attn(x).squeeze(-1)
        if mask is not None:
            w = w.masked_fill(mask == 0, float('-inf'))
        w = F.softmax(w, dim=-1).unsqueeze(-1)
        return (x * w).sum(dim=1)


class AudioEmotionModel(nn.Module):
    """
    WavLM-Large + Weighted Layer Aggregation + AttentiveMeanPool + MLP head.

    layer_aggregation=True (default, v2): обучаемое взвешивание всех слоёв.
    layer_aggregation=False: только последний слой (v1, обратная совместимость).
    """

    def __init__(
        self,
        num_classes: int = 8,
        model_name: str = "microsoft/wavlm-large",
        dropout: float = 0.3,
        freeze_feature_encoder: bool = True,
        freeze_transformer: bool = False,
        layer_aggregation: bool = True,
    ):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.model.config.hidden_size
        self.layer_aggregation = layer_aggregation

        if freeze_feature_encoder:
            self.model.feature_extractor._freeze_parameters()

        if freeze_transformer:
            for p in self.model.encoder.parameters():
                p.requires_grad = False

        if layer_aggregation:
            num_layers = self.model.config.num_hidden_layers + 1  # +1 for embedding
            self.layer_weights = nn.Parameter(torch.zeros(num_layers))

        self.pool = AttentiveMeanPool(self.hidden_size)

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def _aggregate_layers(self, hidden_states: tuple) -> torch.Tensor:
        """Weighted sum of all hidden states: (num_layers, B, T, H) -> (B, T, H)."""
        stacked = torch.stack(hidden_states, dim=0)        # (L, B, T, H)
        weights = F.softmax(self.layer_weights, dim=0)      # (L,)
        weights = weights.view(-1, 1, 1, 1)                 # (L, 1, 1, 1)
        return (stacked * weights).sum(dim=0)                # (B, T, H)

    def embed(self, waveform: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """(B, T) -> (B, hidden_size)"""
        outputs = self.model(
            input_values=waveform,
            attention_mask=attention_mask,
            output_hidden_states=self.layer_aggregation,
        )

        if self.layer_aggregation:
            hidden = self._aggregate_layers(outputs.hidden_states)
        else:
            hidden = outputs.last_hidden_state

        return self.pool(hidden, attention_mask)

    def forward(self, waveform: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        return self.classifier(self.embed(waveform, attention_mask))

    def unfreeze_transformer(self):
        for p in self.model.encoder.parameters():
            p.requires_grad = True


def get_feature_extractor(model_name: str = "microsoft/wavlm-large"):
    return AutoFeatureExtractor.from_pretrained(model_name)
