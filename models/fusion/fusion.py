"""
Модули слияния (Fusion) видео и аудио предсказаний.

Реализованы два варианта:
  1. WeightedFusion               — взвешенное среднее вероятностей (baseline)
  2. CrossModalAttentionFusion    — cross-modal attention + FFN (основной)

Улучшения CrossModalAttentionFusion vs предыдущей версии:
  - 8 голов вместо 4 (выше ёмкость при hidden_dim=256)
  - Post-attention FFN (×4 expansion) — как в стандартном Transformer блоке
  - GELU вместо ReLU в классификаторе
  - Единый проход через backbone (embed() вместо двух вызовов)
  - Поддержка joint fine-tuning базовых моделей
"""

import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedFusion(nn.Module):
    """Взвешенное среднее вероятностей двух модальностей."""

    def __init__(self, num_classes: int = 8, learnable_alpha: bool = True):
        super().__init__()
        if learnable_alpha:
            self.alpha_logit = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer('alpha_logit', torch.tensor([0.0]))

    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha_logit)

    def forward(self, logits_video: torch.Tensor, logits_audio: torch.Tensor) -> torch.Tensor:
        prob_v = F.softmax(logits_video, dim=-1)
        prob_a = F.softmax(logits_audio, dim=-1)
        fused = self.alpha * prob_v + (1 - self.alpha) * prob_a
        return torch.log(fused + 1e-8)


class CrossModalAttentionFusion(nn.Module):
    """
    Cross-modal attention fusion с полноценным Transformer-блоком.

    Каждая модальность обращается к другой через cross-attention,
    затем FFN дополнительно преобразует результат.
    Конкатенация → классификатор.

    Источник: Tsai et al. "Multimodal Transformer for Unaligned
    Multimodal Language Sequences" (ACL 2019) + стандартный FFN блок.
    """

    def __init__(
        self,
        video_dim: int,
        audio_dim: int,
        num_classes: int = 8,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.2,
    ):
        super().__init__()
        # Проекция в общее пространство
        self.proj_v = nn.Linear(video_dim, hidden_dim)
        self.proj_a = nn.Linear(audio_dim, hidden_dim)

        # Cross-attention: v ← a и a ← v
        self.attn_v2a = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.attn_a2v = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )

        self.norm_v1 = nn.LayerNorm(hidden_dim)
        self.norm_a1 = nn.LayerNorm(hidden_dim)

        # FFN для видео-ветки (после cross-attention)
        self.ffn_v = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        # FFN для аудио-ветки
        self.ffn_a = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        self.norm_v2 = nn.LayerNorm(hidden_dim)
        self.norm_a2 = nn.LayerNorm(hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, embed_video: torch.Tensor, embed_audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embed_video: (B, video_dim)
            embed_audio: (B, audio_dim)
        Returns:
            logits: (B, num_classes)
        """
        v = self.proj_v(embed_video).unsqueeze(1)   # (B, 1, H)
        a = self.proj_a(embed_audio).unsqueeze(1)   # (B, 1, H)

        # Cross-attention + residual
        v_att, _ = self.attn_v2a(query=v, key=a, value=a)
        v = self.norm_v1(v + v_att)                 # (B, 1, H)

        a_att, _ = self.attn_a2v(query=a, key=v, value=v)
        a = self.norm_a1(a + a_att)                 # (B, 1, H)

        # FFN + residual
        v = self.norm_v2(v + self.ffn_v(v)).squeeze(1)   # (B, H)
        a = self.norm_a2(a + self.ffn_a(a)).squeeze(1)   # (B, H)

        return self.classifier(torch.cat([v, a], dim=-1))


class FusionModel(nn.Module):
    """
    Обёртка: замороженные видео- и аудио-модели + обучаемый fusion-модуль.

    Поддерживает два режима:
      - joint_finetune=False (default): базовые модели заморожены,
        обучается только fusion. Нет двойного прохода через backbone.
      - joint_finetune=True: все параметры обучаются. Использовать
        enable_joint_finetune() и дифференциальный LR в оптимайзере.
    """

    def __init__(
        self,
        video_model: nn.Module,
        audio_model: nn.Module,
        fusion_type: str = 'attention',
        num_classes: int = 8,
        video_embed_dim: int = 256,
        audio_embed_dim: int = 1024,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.video_model = video_model
        self.audio_model = audio_model
        self.joint_finetune = False

        # Заморозка базовых моделей по умолчанию
        for p in self.video_model.parameters():
            p.requires_grad = False
        for p in self.audio_model.parameters():
            p.requires_grad = False

        if fusion_type == 'weighted':
            self.fusion = WeightedFusion(num_classes=num_classes)
            self.use_logits = True
        elif fusion_type == 'attention':
            self.fusion = CrossModalAttentionFusion(
                video_dim=video_embed_dim,
                audio_dim=audio_embed_dim,
                num_classes=num_classes,
                hidden_dim=hidden_dim,
            )
            self.use_logits = False
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

    def enable_joint_finetune(self):
        """Размораживает базовые модели для совместного обучения."""
        self.joint_finetune = True
        for p in self.video_model.parameters():
            p.requires_grad = True
        for p in self.audio_model.parameters():
            p.requires_grad = True

    def forward(self, frames: torch.Tensor, waveform: torch.Tensor) -> dict:
        """
        Единый проход через оба backbone (без дублирования вычислений).

        Returns dict с logits всех ветвей для логирования и auxiliary loss.
        """
        ctx = contextlib.nullcontext() if self.joint_finetune else torch.no_grad()

        with ctx:
            embed_v = self.video_model.embed(frames)      # (B, video_embed_dim)
            embed_a = self.audio_model.embed(waveform)    # (B, audio_embed_dim)

        # Logits базовых моделей (для логирования и aux loss)
        logits_v = self.video_model.classifier(embed_v)
        logits_a = self.audio_model.classifier(embed_a)

        if self.use_logits:
            logits_f = self.fusion(logits_v, logits_a)
        else:
            logits_f = self.fusion(embed_v, embed_a)

        return {
            'logits_video':  logits_v,
            'logits_audio':  logits_a,
            'logits_fusion': logits_f,
        }
