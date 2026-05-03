# PatchTST autoregressive forecasting mode — design

**Date:** 2026-05-03
**Status:** approved (retroactive validation of already-shipped code)
**Scope:** PatchTST only. DLinear is untouched.

## Background

The existing implementation is direct multi-step forecasting: a single forward
pass maps the L-length lookback window to all `pred_len` future steps via a
flatten + linear head of width `pred_len`. Training and inference both rely on
this single shot.

This spec adds an optional autoregressive (AR) mode where the model emits one
`patch_len`-sized chunk per iteration and rolls out a sliding context window
until the full horizon is covered.

## Non-goals

- Step-level AR (one timestep per iteration).
- Teacher forcing during training.
- AR support for DLinear.
- A new YAML config or sweep entry for AR mode.
- Hyperparameter tuning for AR mode.
- Any change to the direct-mode numeric behavior.

## Config surface

A new field on `Config` (`code/train.py`):

```python
forecasting_mode: str = "direct"   # PatchTST only: "direct" | "autoregressive"
```

- Default `"direct"` keeps existing behavior bit-identical given the same seed.
- Threaded into `_build_model` → `PatchTST(...)`.
- DLinear ignores the field (matches how `kernel_size` / `individual` are
  handled today: shared dataclass, model-specific consumption).
- Opt-in at the CLI: `--override forecasting_mode=autoregressive`. No new YAML
  file is shipped.

## Model architecture

Same encoder in both modes:

- RevIN (optional, `revin=True` default).
- Replication-pad + unfold patching: `N = (L - P)/S + 2` patches.
- Per-patch linear projection `P → D`, learnable positional encoding.
- `n_layers` of pre-norm transformer blocks with `_BatchNormSeq` instead of
  LayerNorm (paper footnote 1).

The only architectural difference between modes is the head's output width:

| mode            | head output width |
|-----------------|-------------------|
| `direct`        | `pred_len`        |
| `autoregressive`| `patch_len`       |

In both cases the head is `Flatten(start_dim=-2) → Dropout(head_dropout) →
Linear(patch_num · d_model → head_out)` applied per channel (channels are
folded into the batch dim, paper "channel independence").

## AR rollout algorithm

Inputs: `x: (B, L, M)`, target horizon `T = pred_len`, patch length `P =
patch_len`.

1. RevIN-normalize `x` once → `x_norm`. Stash `(mean, std)` on the RevIN
   module.
2. `context ← x_norm` (shape `(B, L, M)`).
3. Repeat:
   - Encode `context` with the shared encoder → `(B·M, N, D)`.
   - Apply head, reshape to `(B, P, M)` — one predicted patch in normalized
     space.
   - Append the patch to a list of chunks.
   - If total predicted length `< T`: slide context by `P`:
     `context ← cat([context[:, P:, :], patch], dim=1)`.
   - Otherwise stop.
4. Concatenate all chunks along time → `(B, k·P, M)` where `k = ceil(T / P)`.
5. Trim to exactly `T`: `out ← cat[:, :T, :]`.
6. RevIN-denormalize `out` once → return.

The number of iterations is `ceil(T / P)`. With the paper defaults
(`P = 16`) and `T ∈ {96, 192, 336, 720}` this is `{6, 12, 21, 45}`.

## Training semantics in AR mode

Full rollout, **no teacher forcing**. `forward()` runs the same loop as
inference; the autograd graph spans all `ceil(T / P)` rollout steps. Loss is
MSE against the full `pred_len` target. This matches inference exactly (no
exposure-bias gap) at the cost of ~`ceil(T / P)`× slower training and a deeper
gradient path.

We accept this cost rather than introducing teacher forcing or single-step
training, because the value of AR mode here is to study what happens when the
training objective is the same multi-step rollout used at inference.

## RevIN handling in AR mode

Normalize once on the original L-length input window, denormalize once on the
trimmed output. Predicted patches are concatenated in normalized space and
slide back into the context as if they were normalized observations of the new
window.

This is approximate (the new window has different true stats) but stable: the
model's own predictions don't perturb the normalization stats. The
alternatives (re-normalize each step, or disable RevIN in AR mode) were
considered and rejected — re-normalizing creates a feedback loop where
predictions affect their own normalization, and disabling RevIN gives up the
distribution-shift robustness that PatchTST relies on.

## Public surface

```python
class PatchTST(nn.Module):
    def __init__(..., forecasting_mode: str = "direct"): ...

    def autoregressive_forecast(
        self, x: torch.Tensor, pred_len: int | None = None
    ) -> torch.Tensor:
        """Roll out predictions one patch at a time. Returns (B, pred_len, M)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Direct mode: existing behavior.
        # AR mode: delegates to self.autoregressive_forecast(x, self.pred_len).
```

Output shape and convention `(B, pred_len, M)` match direct mode exactly. The
training loop in `train.py` is unchanged — it calls `model(x)` and computes
MSE against `y`, regardless of mode.

Internal helpers used to keep `forward` and `autoregressive_forecast` from
duplicating the encoder pipeline:

- `_encode(x_norm: (B, L, M)) -> (B·M, N, D)` — patch + project + transformer.
- `_direct_step(x_norm: (B, L, M)) -> (B, pred_len, M)` — used by direct
  forward.
- `_ar_step(x_norm: (B, L, M)) -> (B, patch_len, M)` — one rollout iteration.

## Invariants the implementation must preserve

These are the contract the tests verify:

1. **Direct-mode regression-safe.** With the same seed and inputs, direct mode
   produces the same numeric output it did before this change.
2. **Same output shape across modes.** Both modes return `(B, pred_len, M)`.
3. **Trim correctness.** When `pred_len % patch_len != 0`, AR mode produces
   exactly `pred_len` time steps (not `ceil(pred_len/patch_len) · patch_len`).
4. **Gradient flow.** `loss.backward()` succeeds in AR mode and updates the
   shared encoder parameters.
5. **Helper agreement.** `model.autoregressive_forecast(x, pred_len)` returns
   the same tensor as `model(x)` in AR mode (in eval, with no dropout
   randomness).
6. **Mode validation.** Constructing PatchTST with an unknown `forecasting_mode`
   raises `ValueError`.
7. **Config validation.** AR mode requires `pred_len ≥ 1`; the constructor
   does not need a special check beyond what direct mode already does.

## Out of scope (and why)

- **Step-level AR.** `pred_len`-deep autograd graphs are too expensive at the
  paper's horizons. If we wanted that, we'd want teacher forcing — and we said
  no to that too.
- **Teacher forcing.** Creates the train/test gap (exposure bias) the user
  explicitly wants to avoid.
- **AR for DLinear.** DLinear is the comparison baseline; modifying it would
  change the very thing we're comparing against.
- **AR YAML configs.** AR mode is exploratory; opt-in via `--override` is
  enough until results justify a full sweep entry.

## Files touched

- `code/models/patchtst.py` — AR head, `_encode`, `_direct_step`, `_ar_step`,
  `autoregressive_forecast`, `forward` dispatch, `forecasting_mode` validation.
- `code/train.py` — `Config.forecasting_mode` field, threaded into
  `_build_model`.
- `code/scripts/smoke_test.py` — AR forward+backward, trim case, helper-vs-
  forward agreement.
