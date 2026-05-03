"""PatchTST — supervised variant.

Reference: Nie et al., "A Time Series is Worth 64 Words", ICLR 2023.

Implements the supervised PatchTST faithfully against the official
``PatchTST/PatchTST_supervised/layers/PatchTST_backbone.py`` reference, including:

  * Channel-independence — every channel of a multivariate input is processed by the
    same Transformer backbone, decoupled from the others (B*M reshape into batch).
  * Patching with ReplicationPad1d of S values at the end, giving N = (L - P)/S + 2.
  * Custom multi-head attention with separate Q/K/V projections, an output projection
    + dropout (``proj_dropout``), and RealFormer-style residual attention scores
    propagated across encoder layers (``res_attention=True``).
  * BatchNorm in place of LayerNorm in the encoder block (paper footnote 1).
  * Residual dropout on both sublayers in addition to the in-attention proj dropout.
  * Flatten -> Linear -> Dropout head ordering, matching ``Flatten_Head``.
  * RevIN with configurable ``affine`` (the reference electricity script uses False).

The autoregressive forecasting mode is an extension we added for ablation; in
``direct`` mode the architecture and forward computation match the reference.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.revin import RevIN


class _BatchNormSeq(nn.Module):
    """BatchNorm1d applied across the d_model dimension of a (B, N, D) sequence."""

    def __init__(self, d_model: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(x.transpose(1, 2)).transpose(1, 2)


class _TransposeLast2(nn.Module):
    """Swap the last two dimensions. Used to match the source head's flatten order."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(-1, -2)


class _ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention with optional residual attention scores."""

    def __init__(self, d_model: int, n_heads: int, attn_dropout: float = 0.0,
                 res_attention: bool = False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        # Stored as a non-trainable parameter to match the reference state_dict layout.
        self.scale = nn.Parameter(
            torch.tensor(head_dim ** -0.5), requires_grad=False
        )

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                prev: Optional[torch.Tensor] = None):
        # q: (bs, n_heads, q_len, d_k)
        # k: (bs, n_heads, d_k, q_len)
        # v: (bs, n_heads, q_len, d_v)
        attn_scores = torch.matmul(q, k) * self.scale
        if prev is not None:
            attn_scores = attn_scores + prev
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        output = torch.matmul(attn_weights, v)
        if self.res_attention:
            return output, attn_weights, attn_scores
        return output, attn_weights


class _MultiheadAttention(nn.Module):
    """Multi-head attention matching the supervised PatchTST reference."""

    def __init__(self, d_model: int, n_heads: int, attn_dropout: float = 0.0,
                 proj_dropout: float = 0.0, qkv_bias: bool = True,
                 res_attention: bool = False):
        super().__init__()
        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        self.res_attention = res_attention

        self.W_Q = nn.Linear(d_model, self.d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, self.d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, self.d_v * n_heads, bias=qkv_bias)

        self.sdp_attn = _ScaledDotProductAttention(
            d_model, n_heads, attn_dropout=attn_dropout, res_attention=res_attention
        )

        # Output projection followed by a dropout — both applied before the residual
        # add. The residual dropout in the encoder layer is *additional*.
        self.to_out = nn.Sequential(
            nn.Linear(n_heads * self.d_v, d_model),
            nn.Dropout(proj_dropout),
        )

    def forward(self, Q: torch.Tensor, K: Optional[torch.Tensor] = None,
                V: Optional[torch.Tensor] = None,
                prev: Optional[torch.Tensor] = None):
        bs = Q.size(0)
        if K is None:
            K = Q
        if V is None:
            V = Q

        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0, 2, 3, 1)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1, 2)

        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s)

        output = output.transpose(1, 2).contiguous().view(
            bs, -1, self.n_heads * self.d_v
        )
        output = self.to_out(output)

        if self.res_attention:
            return output, attn_weights, attn_scores
        return output, attn_weights


class _EncoderLayer(nn.Module):
    """Pre-norm transformer encoder block with BatchNorm in place of LayerNorm.

    Layout matches ``TSTEncoderLayer`` (post-norm by default): residual + dropout
    + BatchNorm. With ``res_attention=True`` the pre-softmax attention scores from
    the previous layer are added to this layer's, RealFormer-style.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float,
                 attn_dropout: float = 0.0, res_attention: bool = True):
        super().__init__()
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(
            d_model, n_heads,
            attn_dropout=attn_dropout, proj_dropout=dropout,
            res_attention=res_attention,
        )
        self.dropout_attn = nn.Dropout(dropout)
        self.norm_attn = _BatchNormSeq(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=True),
        )
        self.dropout_ffn = nn.Dropout(dropout)
        self.norm_ffn = _BatchNormSeq(d_model)

    def forward(self, x: torch.Tensor, prev: Optional[torch.Tensor] = None):
        if self.res_attention:
            x2, _, scores = self.self_attn(x, x, x, prev=prev)
        else:
            x2, _ = self.self_attn(x, x, x)
        x = self.norm_attn(x + self.dropout_attn(x2))

        x2 = self.ff(x)
        x = self.norm_ffn(x + self.dropout_ffn(x2))
        if self.res_attention:
            return x, scores
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
    attn_dropout : dropout on softmaxed attention weights (reference default 0)
    head_dropout : dropout applied inside the flatten+linear head
    revin      : whether to apply Reversible Instance Normalization
    affine     : whether RevIN has learnable per-channel affine (reference electricity
                 script default: False)
    res_attention : pass pre-softmax attention scores between encoder layers
                 (reference default: True)
    forecasting_mode : "direct" maps the encoder output to all T future steps in
                 one shot (paper default). "autoregressive" uses a smaller head
                 that emits one patch at a time and rolls out a sliding window
                 until pred_len is reached (extension; not part of the paper).
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
        attn_dropout: float = 0.0,
        head_dropout: float = 0.0,
        revin: bool = True,
        affine: bool = False,
        res_attention: bool = True,
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
        self.res_attention = res_attention
        self.forecasting_mode = forecasting_mode

        # Number of patches after padding the last S values onto the end of the series.
        self.patch_num = (seq_len - patch_len) // stride + 2

        self.padding_patch = nn.ReplicationPad1d((0, stride))

        self.revin = RevIN(c_in, affine=affine) if revin else None

        # Per-patch linear projection P -> D
        self.W_p = nn.Linear(patch_len, d_model)
        # Learnable positional encoding for the patch index (reference: pe='zeros',
        # learn_pe=True, init uniform [-0.02, 0.02]).
        W_pos = torch.empty(self.patch_num, d_model)
        nn.init.uniform_(W_pos, -0.02, 0.02)
        self.W_pos = nn.Parameter(W_pos, requires_grad=True)
        self.dropout = nn.Dropout(dropout)

        self.encoder = nn.ModuleList(
            [
                _EncoderLayer(
                    d_model, n_heads, d_ff, dropout,
                    attn_dropout=attn_dropout, res_attention=res_attention,
                )
                for _ in range(n_layers)
            ]
        )

        # Head: Transpose -> Flatten -> Linear -> Dropout, matching Flatten_Head
        # (which sees (..., d_model, patch_num) and flattens to d_model-major
        # ordering). Output width depends on mode: pred_len for direct, patch_len
        # for AR.
        head_out = pred_len if forecasting_mode == "direct" else patch_len
        self.head = nn.Sequential(
            _TransposeLast2(),  # (B*M, N, D) -> (B*M, D, N) to match source layout
            nn.Flatten(start_dim=-2),
            nn.Linear(self.patch_num * d_model, head_out),
            nn.Dropout(head_dropout),
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
        x = self.W_p(x)  # (B*M, N, D)
        x = self.dropout(x + self.W_pos)

        scores = None
        for layer in self.encoder:
            if self.res_attention:
                x, scores = layer(x, prev=scores)
            else:
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

    def autoregressive_forecast(self, x: torch.Tensor, pred_len: Optional[int] = None) -> torch.Tensor:
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
