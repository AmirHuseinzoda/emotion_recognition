"""
Shared utilities for all training scripts.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler


class Mixup:
    """Mixup augmentation (Zhang et al., 2018).

    Mix two random samples: x_mix = lam*x_i + (1-lam)*x_j.
    lam is clamped to [0.5, 1.0] so the "original" sample always dominates,
    which keeps the dominant label meaningful for top-1 evaluation.
    """

    def __init__(self, alpha: float = 0.4):
        self.dist = torch.distributions.Beta(
            torch.tensor(alpha), torch.tensor(alpha)
        )

    def mix(self, x: torch.Tensor, y: torch.Tensor):
        """Returns x_mix, y_a, y_b, lam."""
        lam = self.dist.sample().item()
        lam = max(lam, 1.0 - lam)          # always >= 0.5
        idx = torch.randperm(x.size(0), device=x.device)
        x_mix = lam * x + (1.0 - lam) * x[idx]
        return x_mix, y, y[idx], lam


def mixup_loss(criterion, logits, y_a, y_b, lam):
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)


class FocalLoss(nn.Module):
    """Focal Loss (Lin et al., 2017) with optional label smoothing and class weights."""

    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.0,
                 num_classes: int = 6, class_weight: torch.Tensor = None):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        self.register_buffer('class_weight', class_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=-1)
        p_t = log_p.exp().gather(1, targets.unsqueeze(1)).squeeze(1)

        if self.label_smoothing > 0:
            smooth = self.label_smoothing / (self.num_classes - 1)
            with torch.no_grad():
                q = torch.full_like(log_p, smooth)
                q.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            ce = -(q * log_p).sum(dim=-1)
        else:
            ce = F.nll_loss(log_p, targets, reduction='none')

        focal = (1 - p_t) ** self.gamma * ce

        if self.class_weight is not None:
            w = self.class_weight[targets]
            focal = focal * w

        return focal.mean()


class EarlyStopping:
    """Stops training when a monitored metric stops improving."""

    def __init__(self, patience: int = 8, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self._counter = 0
        self._best = -float('inf')

    def step(self, metric: float) -> bool:
        """Returns True when training should stop."""
        if metric > self._best + self.min_delta:
            self._best = metric
            self._counter = 0
        else:
            self._counter += 1
        return self._counter >= self.patience


def make_weighted_sampler(labels: np.ndarray, num_classes: int,
                          return_weights: bool = False):
    """Class-balanced sampler: each class is drawn with equal probability.

    If return_weights=True, returns the per-sample weight array instead of a
    sampler — useful when combining class weights with source weights externally.
    """
    counts  = np.bincount(labels, minlength=num_classes).astype(float)
    class_w = np.where(counts > 0, 1.0 / counts, 0.0)
    sample_w = class_w[labels]
    if return_weights:
        return sample_w
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_w, dtype=torch.float64),
        num_samples=len(sample_w),
        replacement=True,
    )
