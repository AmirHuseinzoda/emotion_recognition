"""
Видео-модель: EfficientNet-B0 (HSEmotion/VGAF) + Temporal Transformer.

Pipeline:
  видеокадры (B,T,C,H,W) → EfficientNet-B0 → (B,T,1280) → проекция в 256
  → positional encoding → 2-layer TransformerEncoder → CLS token → (B,256)
  → Dropout → Linear → logits (B, num_classes)

Backbone: EfficientNet-B0 с весами enet_b0_8_best_vgaf.pt (VGGFace2+AffectNet+AFEW).
Temporal: TemporalTransformer — 2-layer self-attention, 8 голов, CLS token.
"""

import math
import torch
import torch.nn as nn


class FrameBackbone(nn.Module):
    """EfficientNet-B0 (HSEmotion) без классификатора — (N,C,H,W) -> (N,1280)."""

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
                   '/blob/main/models/affectnet_emotions/enet_b0_8_best_vgaf.pt?raw=true')
            tmp = os.path.join(tempfile.gettempdir(), 'enet_b0_8_best_vgaf.pt')
            if not os.path.exists(tmp):
                print("[Backbone] Downloading HSEmotion VGAF weights...")
                urllib.request.urlretrieve(url, tmp)
            src = torch.load(tmp, map_location='cpu', weights_only=False)
            src_state = src.state_dict() if hasattr(src, 'state_dict') else src
            dst_state = self.model.state_dict()
            matched = {k: v for k, v in src_state.items()
                       if k in dst_state and dst_state[k].shape == v.shape}
            dst_state.update(matched)
            self.model.load_state_dict(dst_state)
            print(f"[Backbone] HSEmotion VGAF weights loaded ({len(matched)}/{len(dst_state)} layers)")
        except Exception as e:
            print(f"[Backbone] HSEmotion weights unavailable ({e}), using random init")

    def forward(self, x):
        return self.model(x)

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True


class TemporalTransformer(nn.Module):
    """
    Temporal Transformer для последовательности кадров.
    CLS token + positional encoding + 2-layer TransformerEncoder.
    """

    def __init__(self, input_dim: int, hidden_size: int = 256,
                 num_layers: int = 2, num_heads: int = 8,
                 dropout: float = 0.3, max_len: int = 64):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_dim = hidden_size

        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len + 1, hidden_size) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, input_dim) -> (B, hidden_size)"""
        B, T, _ = x.shape
        x = self.proj(x)                                       # (B, T, H)

        cls = self.cls_token.expand(B, -1, -1)                 # (B, 1, H)
        x = torch.cat([cls, x], dim=1)                         # (B, T+1, H)
        x = x + self.pos_embed[:, :T + 1, :]                   # positional encoding

        x = self.encoder(x)                                     # (B, T+1, H)
        return self.norm(x[:, 0])                               # CLS token -> (B, H)


class TemporalLSTM(nn.Module):
    """Bidirectional LSTM (сохранён для обратной совместимости)."""

    def __init__(self, input_dim: int, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
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
        self.attn = nn.Linear(hidden_size * 2, 1)
        self.norm = nn.LayerNorm(hidden_size * 2)
        self.output_dim = hidden_size * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, input_dim) -> (B, 256)"""
        x = self.proj(x)
        out, _ = self.lstm(x)
        w = torch.softmax(self.attn(out), dim=1)
        pooled = (out * w).sum(dim=1)
        return self.norm(pooled)


class VideoEmotionModel(nn.Module):
    """
    FrameBackbone + TemporalTransformer/LSTM + classifier.
    Выход: num_channels-dim (совместимо с Fusion).

    temporal_type: 'transformer' (default, v2) или 'lstm' (v1, обратная совместимость)
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
        temporal_type: str = 'transformer',
    ):
        super().__init__()
        self.backbone = FrameBackbone(pretrained=pretrained, frozen=frozen_backbone)

        if temporal_type == 'transformer':
            self.temporal = TemporalTransformer(
                input_dim=self.backbone.embed_dim,
                hidden_size=num_channels,
                num_layers=2,
                num_heads=8,
                dropout=dropout,
            )
        else:
            self.temporal = TemporalLSTM(
                input_dim=self.backbone.embed_dim,
                hidden_size=num_channels // 2,
                num_layers=2,
                dropout=dropout,
            )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_channels, num_classes),
        )

    def embed(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: (B, T, C, H, W) -> temporal embed: (B, num_channels)"""
        B, T, C, H, W = frames.shape
        frame_embeds = self.backbone(frames.view(B * T, C, H, W)).view(B, T, -1)
        return self.temporal(frame_embeds)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: (B, T, C, H, W) -> logits: (B, num_classes)"""
        return self.classifier(self.embed(frames))

    def unfreeze_backbone(self):
        self.backbone.unfreeze()
