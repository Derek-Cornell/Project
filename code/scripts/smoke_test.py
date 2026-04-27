"""Quick CPU-only smoke test: builds both models with tiny dimensions, runs a forward
and a backward pass on synthetic data, and prints output shapes + parameter counts.

Run with::

    python code/scripts/smoke_test.py

Should finish in a few seconds and print no errors.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import DLinear, PatchTST  # noqa: E402


def _check(name: str, model: torch.nn.Module, x: torch.Tensor, expected_shape: tuple) -> None:
    y = model(x)
    assert tuple(y.shape) == expected_shape, f"{name}: got {tuple(y.shape)}, expected {expected_shape}"
    loss = y.pow(2).mean()
    loss.backward()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  {name:10s}  out={tuple(y.shape)}  loss={loss.item():.4f}  params={n_params:,}")


def main() -> None:
    torch.manual_seed(0)
    B, L, M, T = 4, 336, 8, 96
    x = torch.randn(B, L, M)
    print(f"Input shape: {tuple(x.shape)}\n")

    print("Tiny PatchTST (d_model=32, heads=4, layers=2):")
    _check(
        "patchtst",
        PatchTST(c_in=M, seq_len=L, pred_len=T, d_model=32, n_heads=4, n_layers=2, d_ff=64),
        x,
        (B, T, M),
    )

    print("\nDLinear (individual=True):")
    _check(
        "dlinear",
        DLinear(seq_len=L, pred_len=T, c_in=M, individual=True),
        x,
        (B, T, M),
    )

    print("\nAll good.")


if __name__ == "__main__":
    main()
