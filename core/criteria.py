import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List


class SoftFocalLoss(nn.Module):
    def __init__(
        self,
        alpha: Optional[Union[float, List[float], torch.Tensor]] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        eps: float = 1e-6,
    ):
        super().__init__()
        self._alpha = alpha
        self._gamma = gamma
        self._reduction = reduction
        self._eps = eps

    @property
    def alpha(self) -> Optional[Union[float, List[float], torch.Tensor]]:
        return self._alpha

    @property
    def gamma(self) -> float:
        return self._gamma

    @property
    def reduction(self) -> str:
        return self._reduction

    @property
    def eps(self) -> float:
        return self._eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        _, C = logits.size()

        # Prepare targets as (N, C) soft labels
        y_soft = self._prepare_targets(targets, C)

        # Softmax probabilities with clamping
        p = F.softmax(logits, dim=1).clamp(min=self._eps, max=1.0 - self._eps)

        # Focal modulation
        mod = (1.0 - p) ** self._gamma

        # Class weights
        alpha_vector = self._compute_alpha_vector(C).to(
            device=logits.device, dtype=logits.dtype
        )

        # Compute loss per class
        loss_mat = -alpha_vector.unsqueeze(0) * mod * y_soft * torch.log(p)
        loss = loss_mat.sum(dim=1)

        # Reduce
        if self._reduction == "mean":
            return loss.mean()
        if self._reduction == "sum":
            return loss.sum()
        return loss

    def _prepare_targets(self, targets: torch.Tensor, C: int) -> torch.Tensor:
        if targets.dim() == 1:
            return F.one_hot(targets, num_classes=C).float()
        return targets.float()

    def _compute_alpha_vector(self, C: int) -> torch.Tensor:
        if self._alpha is None:
            return torch.ones(C)
        if isinstance(self._alpha, float):
            a0 = self._alpha
            rest = (1.0 - a0) / (C - 1)
            return torch.tensor([a0] + [rest] * (C - 1), dtype=torch.float)
        # list or tensor
        return torch.tensor(self._alpha, dtype=torch.float)
