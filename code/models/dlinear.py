"""DLinear baseline.

Reference: Zeng et al., "Are Transformers Effective for Time Series Forecasting?", AAAI 2023.

Decomposes the input series into a moving-average trend and a seasonal residual, applies a single
linear layer to each component, and sums the outputs. Despite its simplicity it was the strongest
baseline in the PatchTST paper among non-Transformer models, which is why we compare against it.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _MovingAvg(nn.Module):
    """Length-preserving moving average via reflection-style end padding."""

    def __init__(self, kernel_size: int, stride: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, M)
        pad_left = (self.kernel_size - 1) // 2
        pad_right = self.kernel_size - 1 - pad_left
        front = x[:, :1, :].repeat(1, pad_left, 1)
        end = x[:, -1:, :].repeat(1, pad_right, 1)
        x_padded = torch.cat([front, x, end], dim=1)
        # avg pool over the time axis
        x_avg = self.avg(x_padded.permute(0, 2, 1)).permute(0, 2, 1)
        return x_avg


class _SeriesDecomp(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_avg = _MovingAvg(kernel_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class DLinear(nn.Module):
    """DLinear with optional channel-independent (per-channel) linear heads."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        c_in: int,
        kernel_size: int = 25,
        individual: bool = True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.c_in = c_in
        self.individual = individual
        self.decomp = _SeriesDecomp(kernel_size)

        if individual:
            self.linear_seasonal = nn.ModuleList(
                [nn.Linear(seq_len, pred_len) for _ in range(c_in)]
            )
            self.linear_trend = nn.ModuleList(
                [nn.Linear(seq_len, pred_len) for _ in range(c_in)]
            )
        else:
            self.linear_seasonal = nn.Linear(seq_len, pred_len)
            self.linear_trend = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, M)
        seasonal, trend = self.decomp(x)
        # Reshape to (B, M, L) for the per-time-step linear layers
        seasonal = seasonal.permute(0, 2, 1)
        trend = trend.permute(0, 2, 1)

        if self.individual:
            B = seasonal.size(0)
            out_s = torch.zeros(B, self.c_in, self.pred_len, device=x.device, dtype=x.dtype)
            out_t = torch.zeros_like(out_s)
            for i in range(self.c_in):
                out_s[:, i, :] = self.linear_seasonal[i](seasonal[:, i, :])
                out_t[:, i, :] = self.linear_trend[i](trend[:, i, :])
        else:
            out_s = self.linear_seasonal(seasonal)
            out_t = self.linear_trend(trend)

        return (out_s + out_t).permute(0, 2, 1)  # (B, T, M)
