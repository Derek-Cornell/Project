"""Reversible Instance Normalization (Kim et al., ICLR 2022).

Normalizes each (sample, channel) along the time dimension before the model and
denormalizes the prediction afterwards. Critical for distribution-shift robustness
and used by PatchTST in the supervised setting.
"""

import torch
import torch.nn as nn


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        # x: (B, L, M)
        if mode == "norm":
            self._get_statistics(x)
            return self._normalize(x)
        if mode == "denorm":
            return self._denormalize(x)
        raise ValueError(f"RevIN mode must be 'norm' or 'denorm', got {mode!r}")

    def _get_statistics(self, x: torch.Tensor) -> None:
        dim = 1  # time axis
        self.mean = x.mean(dim=dim, keepdim=True).detach()
        self.stdev = torch.sqrt(
            x.var(dim=dim, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev + self.mean
        return x
