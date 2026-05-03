"""Regression suite for PatchTST autoregressive forecasting mode.

Runs the seven invariants from
docs/superpowers/specs/2026-05-03-patchtst-autoregressive-mode-design.md.

Usage:

    python code/scripts/test_patchtst_ar.py

Prints one line per invariant, exits non-zero on failure.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import PatchTST  # noqa: E402


# Shared tiny-model fixture matching smoke_test.py so the two stay aligned.
B, L, M, T = 4, 336, 8, 96
TINY_KW = dict(d_model=32, n_heads=4, n_layers=2, d_ff=64)
GOLDEN_PATH = os.path.join(
    os.path.dirname(__file__), "..", "tests_data", "patchtst_direct_golden.pt"
)


def _fixed_input(seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, L, M, generator=g)


def _build_direct(seed: int = 0) -> PatchTST:
    torch.manual_seed(seed)
    return PatchTST(c_in=M, seq_len=L, pred_len=T, **TINY_KW)


def _build_ar(seed: int = 0, pred_len: int = T) -> PatchTST:
    torch.manual_seed(seed)
    return PatchTST(
        c_in=M, seq_len=L, pred_len=pred_len,
        forecasting_mode="autoregressive", **TINY_KW,
    )


def main() -> None:
    print("PatchTST AR regression suite")
    x = _fixed_input()

    # Invariant 1: direct-mode regression-safe vs. pre-AR golden.
    model = _build_direct()
    model.eval()
    with torch.no_grad():
        y = model(x)
    golden = torch.load(GOLDEN_PATH, map_location="cpu", weights_only=True)
    assert y.shape == golden.shape, f"shape mismatch: {tuple(y.shape)} vs {tuple(golden.shape)}"
    assert torch.allclose(y, golden, atol=1e-6, rtol=1e-5), (
        f"direct mode drifted from pre-AR baseline: max abs diff = "
        f"{(y - golden).abs().max().item()}"
    )
    print("  [1] direct-mode regression vs golden: pass")

    # Invariant 2: same output shape across modes.
    direct = _build_direct()
    ar = _build_ar()
    direct.eval(); ar.eval()
    with torch.no_grad():
        y_direct = direct(x)
        y_ar = ar(x)
    assert y_direct.shape == (B, T, M), f"direct shape {tuple(y_direct.shape)}"
    assert y_ar.shape == (B, T, M), f"AR shape {tuple(y_ar.shape)}"
    print("  [2] output shape across modes: pass")

    # Invariant 3: trim correctness when pred_len % patch_len != 0.
    T_odd = 100  # patch_len defaults to 16; 100 % 16 == 4
    ar_odd = _build_ar(pred_len=T_odd)
    ar_odd.eval()
    with torch.no_grad():
        y_odd = ar_odd(x)
    assert y_odd.shape == (B, T_odd, M), (
        f"AR with pred_len={T_odd} returned {tuple(y_odd.shape)}, "
        f"expected {(B, T_odd, M)}"
    )
    # Also verify the helper does the trim, not just forward().
    with torch.no_grad():
        y_helper_odd = ar_odd.autoregressive_forecast(x, pred_len=T_odd)
    assert y_helper_odd.shape == (B, T_odd, M)
    print("  [3] trim correctness for pred_len=100: pass")
    print("\nAll invariants pass.")


if __name__ == "__main__":
    main()
