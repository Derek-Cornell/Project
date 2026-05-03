# PatchTST autoregressive forecasting mode — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate the already-shipped autoregressive forecasting mode in PatchTST by writing a dedicated regression test suite that pins every invariant in [the design spec](../specs/2026-05-03-patchtst-autoregressive-mode-design.md), then commit the implementation.

**Architecture:** Implementation lives unstaged in the working tree (`code/models/patchtst.py`, `code/train.py`, `code/scripts/smoke_test.py`). This plan writes a new test script `code/scripts/test_patchtst_ar.py` that exercises each of the seven spec invariants, captures a one-time golden tensor for direct-mode regression against the pre-change `HEAD`, then trims the bolted-on inline AR assertions out of `smoke_test.py` and commits everything.

**Tech Stack:** PyTorch (already a dependency). Script-style tests with `assert` statements, matching the existing `smoke_test.py` convention. No new pytest dependency.

**Important framing:** This is retroactive validation, not first-time TDD. Most test-writing steps will see the assertion pass on first run because the implementation already exists. That is expected and noted in each step's "Expected" output. The one genuinely new artifact is the golden tensor for invariant 1, which requires a `git stash` round-trip to capture against the pre-AR `HEAD`.

---

## File Structure

| File | Status | Responsibility |
| --- | --- | --- |
| `code/models/patchtst.py` | already modified (working tree) | `forecasting_mode` field, `_encode` / `_direct_step` / `_ar_step` helpers, `autoregressive_forecast`, mode dispatch in `forward`. |
| `code/train.py` | already modified (working tree) | `Config.forecasting_mode` field, threaded into `_build_model`. |
| `code/scripts/smoke_test.py` | already modified (working tree) — to be **partially reverted** | Quick CPU-only forward+backward sanity for direct PatchTST + DLinear + a single AR shape check. The earlier pass added four AR assertions here; this plan moves the comprehensive ones into a dedicated test script and keeps just one AR shape check inline. |
| `code/scripts/test_patchtst_ar.py` | **create** | Comprehensive AR-mode regression suite covering invariants 1–7. Run with `python code/scripts/test_patchtst_ar.py`. |
| `code/tests_data/patchtst_direct_golden.pt` | **create** (committed binary) | Deterministic direct-mode forward output captured against pre-AR `HEAD`. Loaded by invariant-1 test to detect any direct-mode numeric regression. |

---

## Task 1: Create the test script skeleton

**Files:**
- Create: `code/scripts/test_patchtst_ar.py`

- [ ] **Step 1: Write the script skeleton with shared fixture**

```python
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
    # Invariants get added in subsequent tasks.
    print("\nAll invariants pass.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the skeleton to confirm it loads**

Run: `python code/scripts/test_patchtst_ar.py`
Expected output:
```
PatchTST AR regression suite

All invariants pass.
```

- [ ] **Step 3: Commit**

```bash
git add code/scripts/test_patchtst_ar.py
git commit -m "test: scaffold AR regression suite for PatchTST"
```

---

## Task 2: Capture direct-mode golden output

This is a one-time capture against the pre-AR `HEAD` (commit `92fb81c` is the design doc; the AR implementation is unstaged). We use `git stash` to temporarily revert the source files, run the capture, then restore.

**Files:**
- Create: `code/tests_data/patchtst_direct_golden.pt` (binary)

- [ ] **Step 1: Verify working tree state before stashing**

Run: `git status --short`
Expected: lines including ` M code/models/patchtst.py`, ` M code/train.py`, ` M code/scripts/smoke_test.py`. If `code/models/patchtst.py` is NOT in the list, stop — the AR implementation is not where this plan assumes.

- [ ] **Step 2: Stash only the three implementation files**

Run:
```bash
git stash push --message "ar-impl-temp" -- code/models/patchtst.py code/train.py code/scripts/smoke_test.py
```
Expected: `Saved working directory and index state On main: ar-impl-temp`. After this, `git status --short` should show no modified tracked files.

- [ ] **Step 3: Create the golden-capture script**

Create `code/scripts/_capture_direct_golden.py` (this file is throwaway — deleted after use):

```python
"""One-time capture of pre-AR direct-mode output. Deleted after use."""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import PatchTST

B, L, M, T = 4, 336, 8, 96
TINY_KW = dict(d_model=32, n_heads=4, n_layers=2, d_ff=64)

torch.manual_seed(0)
model = PatchTST(c_in=M, seq_len=L, pred_len=T, **TINY_KW)
model.eval()

g = torch.Generator().manual_seed(0)
x = torch.randn(B, L, M, generator=g)

with torch.no_grad():
    y = model(x)

out_dir = os.path.join(os.path.dirname(__file__), "..", "tests_data")
os.makedirs(out_dir, exist_ok=True)
torch.save(y, os.path.join(out_dir, "patchtst_direct_golden.pt"))
print(f"saved golden, shape={tuple(y.shape)}, mean={y.mean().item():.6f}")
```

- [ ] **Step 4: Run the capture against pre-AR code**

Run: `python code/scripts/_capture_direct_golden.py`
Expected output: `saved golden, shape=(4, 96, 8), mean=<some float>`. The numeric mean isn't important — what matters is the file `code/tests_data/patchtst_direct_golden.pt` now exists.

- [ ] **Step 5: Verify the golden file exists**

Run: `ls code/tests_data/patchtst_direct_golden.pt`
Expected: file exists.

- [ ] **Step 6: Restore the AR implementation from stash**

Run: `git stash pop`
Expected: `On branch main` and the three modified files reappear in `git status --short`.

- [ ] **Step 7: Delete the throwaway capture script**

Run: `rm code/scripts/_capture_direct_golden.py`
Expected: file gone.

- [ ] **Step 8: Commit the golden file**

```bash
git add code/tests_data/patchtst_direct_golden.pt
git commit -m "test: capture pre-AR direct-mode golden output"
```

---

## Task 3: Invariant 1 — Direct-mode regression

The current direct-mode forward must produce the same tensor we captured pre-AR. Catches any accidental numeric drift in direct mode caused by the refactor.

**Files:**
- Modify: `code/scripts/test_patchtst_ar.py`

- [ ] **Step 1: Add the invariant-1 check inside `main()`**

Replace the `# Invariants get added in subsequent tasks.` line with:

```python
    # Invariant 1: direct-mode regression-safe vs. pre-AR golden.
    model = _build_direct()
    model.eval()
    x = _fixed_input()
    with torch.no_grad():
        y = model(x)
    golden = torch.load(GOLDEN_PATH, map_location="cpu", weights_only=True)
    assert y.shape == golden.shape, f"shape mismatch: {tuple(y.shape)} vs {tuple(golden.shape)}"
    assert torch.allclose(y, golden, atol=1e-6, rtol=1e-5), (
        f"direct mode drifted from pre-AR baseline: max abs diff = "
        f"{(y - golden).abs().max().item()}"
    )
    print("  [1] direct-mode regression vs golden: pass")
```

- [ ] **Step 2: Run the test**

Run: `python code/scripts/test_patchtst_ar.py`
Expected output includes:
```
  [1] direct-mode regression vs golden: pass
```
If it fails with a max-abs-diff message, the refactor changed direct-mode numerics — investigate before continuing.

- [ ] **Step 3: Commit**

```bash
git add code/scripts/test_patchtst_ar.py
git commit -m "test: invariant 1 — direct-mode regression vs golden"
```

---

## Task 4: Invariant 2 — Output shape across modes

Both modes return `(B, pred_len, M)` for any `pred_len`.

**Files:**
- Modify: `code/scripts/test_patchtst_ar.py`

- [ ] **Step 1: Add the invariant-2 block in `main()` after invariant 1**

```python
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
```

- [ ] **Step 2: Run the test**

Run: `python code/scripts/test_patchtst_ar.py`
Expected output adds `  [2] output shape across modes: pass`.

- [ ] **Step 3: Commit**

```bash
git add code/scripts/test_patchtst_ar.py
git commit -m "test: invariant 2 — output shape across modes"
```

---

## Task 5: Invariant 3 — Trim correctness

When `pred_len % patch_len != 0`, AR mode produces exactly `pred_len` time steps.

**Files:**
- Modify: `code/scripts/test_patchtst_ar.py`

- [ ] **Step 1: Add the invariant-3 block in `main()`**

```python
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
```

- [ ] **Step 2: Run the test**

Run: `python code/scripts/test_patchtst_ar.py`
Expected: `  [3] trim correctness for pred_len=100: pass`.

- [ ] **Step 3: Commit**

```bash
git add code/scripts/test_patchtst_ar.py
git commit -m "test: invariant 3 — AR trim correctness"
```

---

## Task 6: Invariant 4 — Gradient flow in AR mode

`loss.backward()` succeeds in AR mode and produces non-zero gradients on shared encoder parameters.

**Files:**
- Modify: `code/scripts/test_patchtst_ar.py`

- [ ] **Step 1: Add the invariant-4 block in `main()`**

```python
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
    head_linear = ar_grad.head[-1]  # final nn.Linear in the head Sequential
    hg = head_linear.weight.grad
    assert hg is not None and hg.abs().sum().item() > 0, "no grad on AR head"
    print("  [4] gradient flow in AR mode: pass")
```

- [ ] **Step 2: Run the test**

Run: `python code/scripts/test_patchtst_ar.py`
Expected: `  [4] gradient flow in AR mode: pass`.

- [ ] **Step 3: Commit**

```bash
git add code/scripts/test_patchtst_ar.py
git commit -m "test: invariant 4 — AR gradient flow"
```

---

## Task 7: Invariant 5 — Helper agreement

`autoregressive_forecast(x, pred_len)` returns the same tensor as `forward(x)` in AR mode (eval, no dropout).

**Files:**
- Modify: `code/scripts/test_patchtst_ar.py`

- [ ] **Step 1: Add the invariant-5 block in `main()`**

```python
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
    # Default pred_len=None should also work.
    with torch.no_grad():
        y_helper_default = ar_agree.autoregressive_forecast(x)
    assert torch.allclose(y_fwd, y_helper_default)
    print("  [5] helper agreement: pass")
```

- [ ] **Step 2: Run the test**

Run: `python code/scripts/test_patchtst_ar.py`
Expected: `  [5] helper agreement: pass`.

- [ ] **Step 3: Commit**

```bash
git add code/scripts/test_patchtst_ar.py
git commit -m "test: invariant 5 — AR helper agrees with forward"
```

---

## Task 8: Invariant 6 — Mode validation

Constructing PatchTST with an unknown `forecasting_mode` raises `ValueError`.

**Files:**
- Modify: `code/scripts/test_patchtst_ar.py`

- [ ] **Step 1: Add the invariant-6 block in `main()`**

```python
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
```

- [ ] **Step 2: Run the test**

Run: `python code/scripts/test_patchtst_ar.py`
Expected: `  [6] mode validation raises ValueError: pass`.

- [ ] **Step 3: Commit**

```bash
git add code/scripts/test_patchtst_ar.py
git commit -m "test: invariant 6 — mode validation"
```

---

## Task 9: Invariant 7 — pred_len=1 still constructs and runs

The constructor must not impose any AR-specific lower bound beyond what direct mode requires. Verifies AR mode handles a 1-step horizon (1 iteration, full trim from `patch_len` to 1).

**Files:**
- Modify: `code/scripts/test_patchtst_ar.py`

- [ ] **Step 1: Add the invariant-7 block in `main()`**

```python
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
    assert y_d1.shape == (B, 1, M)
    assert y_a1.shape == (B, 1, M)
    print("  [7] pred_len=1 in both modes: pass")
```

- [ ] **Step 2: Run the test**

Run: `python code/scripts/test_patchtst_ar.py`
Expected: `  [7] pred_len=1 in both modes: pass`.

- [ ] **Step 3: Commit**

```bash
git add code/scripts/test_patchtst_ar.py
git commit -m "test: invariant 7 — pred_len=1 in both modes"
```

---

## Task 10: Trim AR clutter from smoke_test.py

The earlier (skill-skipping) pass bolted four AR assertions onto `smoke_test.py`. Now that we have a dedicated AR test script, `smoke_test.py` should go back to its quick-sanity role: one direct PatchTST check, one AR shape check (so anyone running smoke catches a totally broken AR path), one DLinear check.

**Files:**
- Modify: `code/scripts/smoke_test.py`

- [ ] **Step 1: Replace the AR section with a single shape check**

Read the current file first to confirm its state, then replace the block from `print("\nPatchTST autoregressive (pred_len=96, divides patch_len=16):")` through (and including) the `print("  direct vs AR: shapes match, values differ, helper agrees with forward().")` line — i.e., the four AR-related blocks I added — with this single block:

```python
    print("\nPatchTST autoregressive (smoke shape check only — full suite in test_patchtst_ar.py):")
    _check(
        "patchtst-ar",
        PatchTST(
            c_in=M, seq_len=L, pred_len=T, d_model=32, n_heads=4, n_layers=2, d_ff=64,
            forecasting_mode="autoregressive",
        ),
        x,
        (B, T, M),
    )
```

The DLinear block and the closing `print("\nAll good.")` stay as they are.

- [ ] **Step 2: Run the smoke test**

Run: `python code/scripts/smoke_test.py`
Expected output ends with:
```
PatchTST autoregressive (smoke shape check only — full suite in test_patchtst_ar.py):
  patchtst-ar  out=(4, 96, 8)  loss=...  params=...

DLinear (individual=True):
  dlinear     out=(4, 96, 8)  loss=...  params=...

All good.
```

- [ ] **Step 3: Run the full AR suite to confirm nothing broke**

Run: `python code/scripts/test_patchtst_ar.py`
Expected: all seven `[N] ...: pass` lines and final `All invariants pass.`

- [ ] **Step 4: Commit smoke_test.py and the implementation**

The implementation files are still unstaged. Commit them together with the smoke trim, since they belong to the same feature.

```bash
git add code/models/patchtst.py code/train.py code/scripts/smoke_test.py
git commit -m "feat(patchtst): add optional autoregressive forecasting mode

Adds forecasting_mode={direct,autoregressive} to PatchTST and threads
it through Config. Direct mode is unchanged. Autoregressive mode emits
one patch per iteration, slides the L-length context, and trims to
pred_len. RevIN normalize/denormalize once around the rollout.

Design: docs/superpowers/specs/2026-05-03-patchtst-autoregressive-mode-design.md
Tests:  code/scripts/test_patchtst_ar.py (seven invariants)"
```

---

## Self-Review

Verifying the plan against the spec invariants:

| Spec invariant | Task |
| --- | --- |
| 1. Direct-mode regression-safe | Task 2 (golden capture) + Task 3 (test) |
| 2. Same output shape across modes | Task 4 |
| 3. Trim correctness | Task 5 |
| 4. Gradient flow | Task 6 |
| 5. Helper agreement | Task 7 |
| 6. Mode validation | Task 8 |
| 7. pred_len lower bound | Task 9 |

Plus Task 1 (scaffold) and Task 10 (cleanup + final commit).

**Placeholder scan:** every code block is concrete, every command is exact, every "expected" output is named. No TBDs.

**Type/name consistency:** `_build_direct`, `_build_ar`, `_fixed_input`, `TINY_KW`, `GOLDEN_PATH`, `B/L/M/T` are defined in Task 1 and used consistently in Tasks 3–9. Method names `autoregressive_forecast` and `forecasting_mode` match the spec and the working-tree implementation.

**Spec coverage:** all seven spec invariants have corresponding test tasks. The "Files touched" section of the spec maps directly to the working-tree files this plan commits in Task 10.
