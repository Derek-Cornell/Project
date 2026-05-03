"""PatchTST — supervised variant.

Reference: Nie et al., "A Time Series is Worth 64 Words", ICLR 2023.

Key design points from §3.1 of the paper that this file implements faithfully:
  * Channel-independence — every channel of a multivariate input is processed by the same
    Transformer backbone, decoupled from the others. Implemented as a B×M reshape into the
    batch dimension before the encoder.
  * Patching with replication padding of S values at the end, giving N = (L - P)/S + 2 patches.
  * Encoder uses BatchNorm instead of LayerNorm (footnote 1 of the paper, citing Zerveas 2021).
  * RevIN normalization on the input series, denormalized on the prediction.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from utils.revin import RevIN


class _BatchNormSeq(nn.Module):
    """BatchNorm1d applied across the d_model dimension of a (B, N, D) sequence."""

    def __init__(self, d_model: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, N, D) -> (B, D, N) -> bn -> (B, N, D)
        return self.bn(x.transpose(1, 2)).transpose(1, 2)


class _EncoderLayer(nn.Module):
    """Pre-norm transformer encoder block with BatchNorm in place of LayerNorm.

    Paper footnote: BatchNorm outperforms LayerNorm on time series Transformers
    (Zerveas et al., 2021). Residual + BN ordering matches the reference impl.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.dropout_attn = nn.Dropout(dropout)
        self.norm_attn = _BatchNormSeq(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout_ff = nn.Dropout(dropout)
        self.norm_ff = _BatchNormSeq(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm_attn(x + self.dropout_attn(attn_out))
        x = self.norm_ff(x + self.dropout_ff(self.ff(x)))
        return x


class PatchTST(nn.Module):
    """Supervised PatchTST/N model.

    Parameters
    ----------
    c_in       : number of input channels (M)
    seq_len    : look-back window length (L)
    pred_len   : forecast horizon (T)
    patch_len  : patch length P (default 16, per paper)
    stride     : patch stride S (default 8, gives 50% overlap)
    d_model    : latent dimension D
    n_heads    : number of attention heads H
    n_layers   : number of encoder layers
    d_ff       : feed-forward inner dimension
    dropout    : dropout used inside encoder layers and after patch projection
    head_dropout : dropout applied right before the flatten+linear head
    revin      : whether to apply Reversible Instance Normalization
    forecasting_mode : "direct" maps the encoder output to all T future steps in
                 one shot (paper default). "autoregressive" uses a smaller head
                 that emits one patch at a time and rolls out a sliding window
                 until pred_len is reached.
    """

    def __init__(
        self,
        c_in: int,
        seq_len: int,
        pred_len: int,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_heads: int = 16,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.2,
        head_dropout: float = 0.0,
        revin: bool = True,
        forecasting_mode: str = "direct",
    ):
        super().__init__()
        if forecasting_mode not in ("direct", "autoregressive"):
            raise ValueError(
                f"forecasting_mode must be 'direct' or 'autoregressive', got {forecasting_mode!r}"
            )
        self.c_in = c_in
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.forecasting_mode = forecasting_mode

        # Number of patches after padding the last S values onto the end of the series.
        self.patch_num = (seq_len - patch_len) // stride + 2

        self.padding_patch = nn.ReplicationPad1d((0, stride))

        self.revin = RevIN(c_in, affine=True) if revin else None

        # Per-patch linear projection P -> D
        self.W_p = nn.Linear(patch_len, d_model)
        # Learnable positional encoding for the patch index
        self.W_pos = nn.Parameter(torch.empty(self.patch_num, d_model))
        nn.init.uniform_(self.W_pos, -0.02, 0.02)
        self.dropout = nn.Dropout(dropout)

        self.encoder = nn.ModuleList(
            [_EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )

        # Head output width depends on mode: pred_len for direct, patch_len for AR.
        head_out = pred_len if forecasting_mode == "direct" else patch_len
        self.head = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Dropout(head_dropout),
            nn.Linear(self.patch_num * d_model, head_out),
        )

    def _encode(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Patch + project + encode. Input is already RevIN-normalized.

        x_norm : (B, L, M)
        returns: (B*M, N, D)
        """
        B, L, M = x_norm.shape
        assert L == self.seq_len, f"expected look-back {self.seq_len}, got {L}"
        assert M == self.c_in, f"expected {self.c_in} channels, got {M}"

        # (B, L, M) -> (B, M, L) -> pad -> unfold along time
        x = x_norm.permute(0, 2, 1)
        x = self.padding_patch(x)  # (B, M, L+S)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # (B, M, N, P)

        # Channel independence: fold M into the batch dimension
        x = x.reshape(B * M, self.patch_num, self.patch_len)
        x = self.W_p(x) + self.W_pos.unsqueeze(0)  # (B*M, N, D)
        x = self.dropout(x)

        for layer in self.encoder:
            x = layer(x)
        return x  # (B*M, N, D)

    def _direct_step(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Direct forecast in normalized space: (B, L, M) -> (B, pred_len, M)."""
        B, _, M = x_norm.shape
        z = self._encode(x_norm)
        out = self.head(z)  # (B*M, pred_len)
        return out.reshape(B, M, self.pred_len).permute(0, 2, 1)

    def _ar_step(self, x_norm: torch.Tensor) -> torch.Tensor:
        """One AR step in normalized space: (B, L, M) -> (B, patch_len, M)."""
        B, _, M = x_norm.shape
        z = self._encode(x_norm)
        out = self.head(z)  # (B*M, patch_len)
        return out.reshape(B, M, self.patch_len).permute(0, 2, 1)

    def autoregressive_forecast(self, x: torch.Tensor, pred_len: int | None = None) -> torch.Tensor:
        """Roll out predictions one patch at a time until ``pred_len`` is reached.

        Each iteration predicts the next ``patch_len`` values, slides the L-length
        context forward by ``patch_len`` (dropping the oldest values, appending the
        new patch), and continues. The final output is trimmed to exactly pred_len.

        x        : (B, L, M)
        pred_len : forecast horizon. Defaults to ``self.pred_len``.
        returns  : (B, pred_len, M)
        """
        if pred_len is None:
            pred_len = self.pred_len
        if pred_len < 1:
            raise ValueError(f"pred_len must be >= 1, got {pred_len}")
        B, L, M = x.shape
        assert L == self.seq_len, f"expected look-back {self.seq_len}, got {L}"
        assert M == self.c_in, f"expected {self.c_in} channels, got {M}"

        # Normalize once so the rolled-out context lives in a single normalized space;
        # denormalize the concatenated predictions at the end.
        if self.revin is not None:
            x_norm = self.revin(x, mode="norm")
        else:
            x_norm = x

        context = x_norm
        chunks = []
        produced = 0
        while produced < pred_len:
            patch = self._ar_step(context)  # (B, patch_len, M), still normalized
            chunks.append(patch)
            produced += self.patch_len
            if produced < pred_len:
                # Slide the L-length window forward by patch_len.
                context = torch.cat([context[:, self.patch_len:, :], patch], dim=1)

        out = torch.cat(chunks, dim=1)[:, :pred_len, :]  # (B, pred_len, M)

        if self.revin is not None:
            out = self.revin(out, mode="denorm")
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, M) — standard time-series convention
        if self.forecasting_mode == "autoregressive":
            return self.autoregressive_forecast(x, self.pred_len)

        if self.revin is not None:
            x_norm = self.revin(x, mode="norm")
        else:
            x_norm = x
        out = self._direct_step(x_norm)
        if self.revin is not None:
            out = self.revin(out, mode="denorm")
        return out
