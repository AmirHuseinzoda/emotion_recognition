"""
Видео-модель: Face-emotion backbone (HSEmotion/AffectNet) + Bi-LSTM.

FrameBackbone: EfficientNet-B0, AffectNet/AFEW pretrained (HSEmotion).
TemporalLSTM: bidirectional LSTM, лучше TCN/Transformer на малых датасетах,
              стабилен в realtime-инференсе.
"""

import torch
import torch.nn as nn


class FrameBackbone(nn.Module):
    """EfficientNet-B0 (HSEmotion/AffectNet) без классификатора — (N,C,H,W) -> (N,1280)."""

    def __init__(self, pretrained: bool = True, frozen: bool = False):
        super().__init__()
        self.embed_dim = 1280

        import timm
        self.model = timm.create_model(
            'tf_efficientnet_b0', pretrained=False, num_classes=0, global_pool='avg',
        )
        if pretrained:
            self._load_hsemotion_weights()
        if frozen:
            for p in self.parameters():
                p.requires_grad = False

    def _load_hsemotion_weights(self):
        try:
            import urllib.request, tempfile, os
            url = ('https://github.com/HSE-asavchenko/face-emotion-recognition'
                   '/blob/main/models/affectnet_emotions/enet_b0_8_best_afew.pt?raw=true')
            tmp = os.path.join(tempfile.gettempdir(), 'enet_b0_8_best_afew.pt')
            if not os.path.exists(tmp):
                print("[Backbone] Downloading HSEmotion weights...")
                urllib.request.urlretrieve(url, tmp)
            src = torch.load(tmp, map_location='cpu', weights_only=False)
            src_state = src.state_dict() if hasattr(src, 'state_dict') else src
            dst_state = self.model.state_dict()
            matched = {k: v for k, v in src_state.items()
                       if k in dst_state and dst_state[k].shape == v.shape}
            dst_state.update(matched)
            self.model.load_state_dict(dst_state)
            print(f"[Backbone] HSEmotion weights loaded ({len(matched)}/{len(dst_state)} layers)")
        except Exception as e:
            print(f"[Backbone] HSEmotion weights unavailable ({e}), using random init")

    def forward(self, x):
        return self.model(x)

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True


class TemporalLSTM(nn.Module):
    """
    Bidirectional LSTM для временного моделирования последовательности кадров.

    Преимущества над TCN/Transformer:
    - Меньше параметров -> меньше переобучение на малом датасете
    - Быстрый инференс (нет O(T^2) attention)
    - Bi-LSTM видит контекст в обоих направлениях
    - Среднее по времени устойчивее к пропущенным/плохим кадрам в realtime
    """

    def __init__(self, input_dim: int, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        # Проекция 1280 -> 256 перед LSTM (ускоряет обучение)
        proj_dim = hidden_size * 2
        self.proj = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=proj_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn = nn.Linear(hidden_size * 2, 1)   # attention pooling
        self.norm = nn.LayerNorm(hidden_size * 2)
        self.output_dim = hidden_size * 2  # 256 — совместимо с FusionModel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, input_dim) -> (B, 256)"""
        x = self.proj(x)                        # (B, T, 256)
        out, _ = self.lstm(x)                   # (B, T, 256)
        # Attention pooling: обучаемые веса вместо простого среднего.
        # Фокусирует на кадрах с наиболее выразительной мимикой.
        w = torch.softmax(self.attn(out), dim=1)  # (B, T, 1)
        pooled = (out * w).sum(dim=1)           # (B, 256)
        return self.norm(pooled)


class VideoEmotionModel(nn.Module):
    """FrameBackbone + TemporalLSTM + classifier. Выход: 256-dim (совместимо с Fusion)."""

    def __init__(
        self,
        num_classes: int = 8,
        num_channels: int = 256,
        num_levels: int = 5,    # не используется — сохранён для совместимости API
        kernel_size: int = 3,   # не используется — сохранён для совместимости API
        dropout: float = 0.2,
        pretrained: bool = True,
        frozen_backbone: bool = True,
    ):
        super().__init__()
        self.backbone = FrameBackbone(pretrained=pretrained, frozen=frozen_backbone)
        self.temporal = TemporalLSTM(
            input_dim=self.backbone.embed_dim,
            hidden_size=num_channels // 2,  # 128 -> bidirectional -> 256
            num_layers=2,
            dropout=dropout,  # same as classifier — both controlled by config
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_channels, num_classes),
        )

    def embed(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: (B, T, C, H, W) -> temporal embed: (B, 256)"""
        B, T, C, H, W = frames.shape
        frame_embeds = self.backbone(frames.view(B * T, C, H, W)).view(B, T, -1)
        return self.temporal(frame_embeds)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: (B, T, C, H, W) -> logits: (B, num_classes)"""
        return self.classifier(self.embed(frames))

    def unfreeze_backbone(self):
        self.backbone.unfreeze()
