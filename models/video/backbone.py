"""
Видео-модель: Face-emotion backbone (HSEmotion/AffectNet) + TCN.

FrameBackbone загружает EfficientNet-B0 с весами HSEmotion, обученными
на AffectNet (450k лиц) — вместо ImageNet. Это даёт модели стартовое
знание о выражениях лиц до обучения на RAVDESS.
"""

import torch
import torch.nn as nn

from models.video.tcn import TCN


class FrameBackbone(nn.Module):
    """
    EfficientNet-B0 с весами HSEmotion (AffectNet) без финального классификатора.
    Обрабатывает один кадр → вектор признаков (embed_dim,).
    """

    def __init__(self, pretrained: bool = True, frozen: bool = False):
        super().__init__()
        self.embed_dim = 1280

        import timm
        self.model = timm.create_model(
            'tf_efficientnet_b0',
            pretrained=False,
            num_classes=0,      # убираем classifier, выход = avgpool features
            global_pool='avg',
        )

        if pretrained:
            self._load_hsemotion_weights()

        if frozen:
            for p in self.parameters():
                p.requires_grad = False

    def _load_hsemotion_weights(self):
        try:
            import timm
            import urllib.request
            import tempfile
            import os

            url = ('https://github.com/HSE-asavchenko/face-emotion-recognition'
                   '/blob/main/models/affectnet_emotions/enet_b0_8_best_afew.pt?raw=true')
            tmp = os.path.join(tempfile.gettempdir(), 'enet_b0_8_best_afew.pt')
            if not os.path.exists(tmp):
                print("[Backbone] Downloading HSEmotion weights...")
                urllib.request.urlretrieve(url, tmp)

            # weights_only=False — доверяем официальному репозиторию HSE
            src_model = torch.load(tmp, map_location='cpu', weights_only=False)
            src_state = src_model.state_dict() if hasattr(src_model, 'state_dict') else src_model

            dst_state = self.model.state_dict()
            transferred = {k: v for k, v in src_state.items()
                           if k in dst_state and dst_state[k].shape == v.shape}
            dst_state.update(transferred)
            self.model.load_state_dict(dst_state)
            print(f"[Backbone] HSEmotion weights loaded ({len(transferred)}/{len(dst_state)} layers)")
        except Exception as e:
            print(f"[Backbone] HSEmotion weights unavailable ({e}), using random init")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, C, H, W) → (N, embed_dim)"""
        return self.model(x)

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True


class VideoEmotionModel(nn.Module):
    """
    Полная модель: FrameBackbone (per-frame) + TCN (temporal) + classifier.
    """

    def __init__(
        self,
        num_classes: int = 8,
        num_channels: int = 256,
        num_levels: int = 5,
        kernel_size: int = 3,
        dropout: float = 0.2,
        pretrained: bool = True,
        frozen_backbone: bool = True,
    ):
        super().__init__()
        self.backbone = FrameBackbone(pretrained=pretrained, frozen=frozen_backbone)
        self.tcn = TCN(
            input_dim=self.backbone.embed_dim,
            num_channels=num_channels,
            num_levels=num_levels,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_channels, num_classes),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: (B, T, C, H, W) → logits: (B, num_classes)"""
        B, T, C, H, W = frames.shape
        embeds = self.backbone(frames.view(B * T, C, H, W)).view(B, T, -1)
        return self.classifier(self.tcn(embeds))

    def unfreeze_backbone(self):
        self.backbone.unfreeze()
