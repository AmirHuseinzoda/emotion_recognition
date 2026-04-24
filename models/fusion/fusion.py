"""
Модули слияния (Fusion) видео и аудио предсказаний.

Реализованы два варианта:
  1. WeightedFusion   — взвешенное среднее вероятностей (простой baseline)
  2. AttentionFusion  — cross-modal attention над эмбеддингами
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedFusion(nn.Module):
    """
    Взвешенное среднее вероятностей двух модальностей.
    alpha — вес видео, (1 - alpha) — вес аудио.
    alpha может быть фиксированным или обучаемым параметром.
    """

    def __init__(self, num_classes: int = 8, learnable_alpha: bool = True):
        super().__init__()
        if learnable_alpha:
            # Логит, из которого sigmoid даёт alpha ∈ (0, 1)
            self.alpha_logit = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer('alpha_logit', torch.tensor([0.0]))

    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha_logit)

    def forward(
        self,
        logits_video: torch.Tensor,
        logits_audio: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits_video: (B, num_classes)
            logits_audio: (B, num_classes)
        Returns:
            logits_fused: (B, num_classes)
        """
        prob_v = F.softmax(logits_video, dim=-1)
        prob_a = F.softmax(logits_audio, dim=-1)
        alpha = self.alpha
        fused = alpha * prob_v + (1 - alpha) * prob_a
        # Возвращаем логиты (log для числовой стабильности)
        return torch.log(fused + 1e-8)


class CrossModalAttentionFusion(nn.Module):
    """
    Cross-modal attention fusion.
    Видео-эмбеддинг обращается к аудио-эмбеддингу через attention
    и наоборот, затем конкатенируются и проецируются в num_classes.

    Источник: Tsai et al. "Multimodal Transformer for Unaligned
    Multimodal Language Sequences" (ACL 2019).
    """

    def __init__(
        self,
        video_dim: int,
        audio_dim: int,
        num_classes: int = 8,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        # Проекция в общее пространство
        self.proj_v = nn.Linear(video_dim, hidden_dim)
        self.proj_a = nn.Linear(audio_dim, hidden_dim)

        # v ← a: видео смотрит на аудио
        self.attn_v2a = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )
        # a ← v: аудио смотрит на видео
        self.attn_a2v = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )

        self.norm_v = nn.LayerNorm(hidden_dim)
        self.norm_a = nn.LayerNorm(hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        embed_video: torch.Tensor,
        embed_audio: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            embed_video: (B, video_dim) — эмбеддинг из видео-модели
            embed_audio: (B, audio_dim) — эмбеддинг из аудио-модели
        Returns:
            logits: (B, num_classes)
        """
        # Добавляем seq_len=1 для MultiheadAttention
        v = self.proj_v(embed_video).unsqueeze(1)   # (B, 1, H)
        a = self.proj_a(embed_audio).unsqueeze(1)   # (B, 1, H)

        v_attended, _ = self.attn_v2a(query=v, key=a, value=a)
        a_attended, _ = self.attn_a2v(query=a, key=v, value=v)

        v_out = self.norm_v(v + v_attended).squeeze(1)   # (B, H)
        a_out = self.norm_a(a + a_attended).squeeze(1)   # (B, H)

        fused = torch.cat([v_out, a_out], dim=-1)         # (B, 2H)
        return self.classifier(fused)


class FusionModel(nn.Module):
    """
    Обёртка, объединяющая замороженные видео- и аудио-модели
    с обучаемым fusion-модулем.
    """

    def __init__(
        self,
        video_model: nn.Module,
        audio_model: nn.Module,
        fusion_type: str = 'attention',   # 'weighted' | 'attention'
        num_classes: int = 8,
        video_embed_dim: int = 256,       # выход TCN
        audio_embed_dim: int = 768,       # hidden_size HuBERT
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.video_model = video_model
        self.audio_model = audio_model

        # Заморозка базовых моделей
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

    def forward(
        self,
        frames: torch.Tensor,
        waveform: torch.Tensor,
    ) -> dict:
        """
        Returns dict с logits всех ветвей для удобного логирования.
        """
        with torch.no_grad():
            logits_v = self.video_model(frames)
            logits_a = self.audio_model(waveform)

        if self.use_logits:
            logits_f = self.fusion(logits_v, logits_a)
        else:
            # Нужны эмбеддинги, а не логиты — достаём напрямую
            embed_v = self._get_video_embed(frames)
            embed_a = self._get_audio_embed(waveform)
            logits_f = self.fusion(embed_v, embed_a)

        return {
            'logits_video':  logits_v,
            'logits_audio':  logits_a,
            'logits_fusion': logits_f,
        }

    def _get_video_embed(self, frames):
        B, T, C, H, W = frames.shape
        flat = frames.view(B * T, C, H, W)
        embeds = self.video_model.backbone(flat).view(B, T, -1)
        return self.video_model.tcn(embeds)  # (B, num_channels)

    def _get_audio_embed(self, waveform):
        outputs = self.audio_model.hubert(input_values=waveform)
        return outputs.last_hidden_state.mean(dim=1)  # (B, hidden_size)
