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
    assert y_helper_odd.shape == (B, T_odd, M), (
        f"autoregressive_forecast trim returned {tuple(y_helper_odd.shape)}, "
        f"expected {(B, T_odd, M)}"
    )
    print("  [3] trim correctness for pred_len=100: pass")

    # Invariant 4: gradient flow through the AR rollout.
    ar_grad = _build_ar()
    ar_grad.train()
    x_grad = _fixed_input(seed=1)
    target = torch.zeros(B, T, M)
    out = ar_grad(x_grad)
    loss = (out - target).pow(2).mean()
    loss.backward()
    # The patch projection is in the shared encoder — must receive gradient.
    g = ar_grad.W_p.weight.grad
    assert g is not None, "no grad on W_p — autograd graph broken in AR mode"
    assert torch.isfinite(g).all(), "non-finite grad on W_p"
    assert g.abs().sum().item() > 0, "all-zero grad on W_p — rollout not differentiating"
    # The AR head must also receive gradient.
    head_linear = next(m for m in ar_grad.head if isinstance(m, torch.nn.Linear))
    hg = head_linear.weight.grad
    assert hg is not None and hg.abs().sum().item() > 0, "no grad on AR head"
    print("  [4] gradient flow in AR mode: pass")

    # Invariant 5: autoregressive_forecast() agrees with forward() in AR mode.
    ar_agree = _build_ar()
    ar_agree.eval()
    with torch.no_grad():
        y_fwd = ar_agree(x)
        y_helper = ar_agree.autoregressive_forecast(x, pred_len=T)
    assert torch.allclose(y_fwd, y_helper), (
        "forward() and autoregressive_forecast() disagree in AR mode; "
        f"max abs diff = {(y_fwd - y_helper).abs().max().item()}"
    )
    # `pred_len=None` must resolve to self.pred_len. Verify by building a
    # second model with a different pred_len and confirming the helper's
    # default matches that model's forward(), not ar_agree's.
    other = _build_ar(pred_len=48)
    other.eval()
    with torch.no_grad():
        y_other_fwd = other(x)
        y_other_default = other.autoregressive_forecast(x)
    assert y_other_default.shape == y_other_fwd.shape, (
        f"default pred_len resolved to wrong horizon: "
        f"{tuple(y_other_default.shape)} vs forward {tuple(y_other_fwd.shape)}"
    )
    assert torch.allclose(y_other_fwd, y_other_default), (
        "autoregressive_forecast(x) (default pred_len) disagrees with forward() "
        f"on a model with pred_len=48; max abs diff = "
        f"{(y_other_fwd - y_other_default).abs().max().item()}"
    )
    print("  [5] helper agreement: pass")

    # Invariant 6: unknown forecasting_mode raises ValueError.
    raised = False
    try:
        PatchTST(
            c_in=M, seq_len=L, pred_len=T,
            forecasting_mode="banana", **TINY_KW,
        )
    except ValueError:
        raised = True
    assert raised, "unknown forecasting_mode should raise ValueError"
    print("  [6] mode validation raises ValueError: pass")

    # Invariant 7: pred_len=1 works in both modes (no special lower-bound check).
    direct1 = PatchTST(c_in=M, seq_len=L, pred_len=1, **TINY_KW)
    ar1 = PatchTST(
        c_in=M, seq_len=L, pred_len=1,
        forecasting_mode="autoregressive", **TINY_KW,
    )
    direct1.eval(); ar1.eval()
    with torch.no_grad():
        y_d1 = direct1(x)
        y_a1 = ar1(x)
    assert y_d1.shape == (B, 1, M), (
        f"direct mode pred_len=1 returned {tuple(y_d1.shape)}, expected {(B, 1, M)}"
    )
    assert y_a1.shape == (B, 1, M), (
        f"AR mode pred_len=1 returned {tuple(y_a1.shape)}, expected {(B, 1, M)}"
    )
    print("  [7] pred_len=1 in both modes: pass")
    print("\nAll invariants pass.")


if __name__ == "__main__":
    main()
