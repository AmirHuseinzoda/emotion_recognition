"""
Temporal Convolutional Network (TCN) для анализа последовательностей кадров.
Реализация на основе: Bai et al. "An Empirical Evaluation of Generic
Convolutional and Recurrent Networks for Sequence Modeling" (2018).

Дилатированные каузальные свёртки + residual connections.
"""

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class CausalConv1d(nn.Module):
    """Каузальная свёртка: не смотрит в будущее."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = weight_norm(nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        ))

    def forward(self, x):
        out = self.conv(x)
        # Убираем padding справа (будущие шаги)
        return out[:, :, :-self.padding] if self.padding > 0 else out


class TCNBlock(nn.Module):
    """Один residual блок TCN: два dilated causal conv + skip connection."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)

        # Проекция если размерности не совпадают
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else None
        )

    def forward(self, x):
        residual = x

        out = self.relu(self.norm1(self.conv1(x)))
        out = self.dropout(out)
        out = self.relu(self.norm2(self.conv2(out)))
        out = self.dropout(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        return self.relu(out + residual)


class TCN(nn.Module):
    """
    Стек TCN-блоков с экспоненциально растущей дилатацией.

    Args:
        input_dim:    размерность входного вектора на каждом шаге (D)
        num_channels: число каналов в каждом блоке
        num_levels:   количество блоков (дилатация: 1, 2, 4, ..., 2^(num_levels-1))
        kernel_size:  размер ядра свёртки
        dropout:      dropout после каждой свёртки
    """

    def __init__(
        self,
        input_dim: int,
        num_channels: int = 256,
        num_levels: int = 5,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        layers = []
        for i in range(num_levels):
            dilation = 2 ** i
            in_ch = input_dim if i == 0 else num_channels
            layers.append(TCNBlock(in_ch, num_channels, kernel_size, dilation, dropout))
        self.network = nn.Sequential(*layers)
        self.output_dim = num_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) — последовательность эмбеддингов
        Returns:
            (B, num_channels) — агрегированное представление последовательности
        """
        x = x.permute(0, 2, 1)       # (B, D, T)
        x = self.network(x)            # (B, num_channels, T)
        x = x.mean(dim=-1)            # (B, num_channels) — global avg pool по времени
        return x
