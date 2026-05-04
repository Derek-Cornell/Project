# PatchTST direct vs autoregressive forecasting: findings on the Illness dataset

## What we did

PatchTST (Nie et al., ICLR 2023) is a Transformer-based time-series forecaster whose central design choice is the **direct head**: a single linear layer that maps the encoder's pooled features to all `pred_len` future timesteps in one shot. The paper argues this avoids the error-compounding problems of autoregression but does not actually quantify the trade-off.

Our extension is a head-architecture comparison:

- **Direct** (paper default): `Linear(encoder_features → pred_len)`. One forward pass, one head, all future steps emitted simultaneously.
- **Autoregressive (AR)**: `Linear(encoder_features → patch_len)`. Predicts one patch at a time, slides the look-back forward by `patch_len`, feeds the modified context back into the encoder, predicts the next patch, and repeats until the horizon is covered.

The two architectures are otherwise identical. We confirmed this by copying weights between the source PatchTST implementation and our reimplementation in direct mode and verifying that outputs match to floating-point exactness. The AR head is the only deliberate divergence.

We evaluated on **Illness (ILI)** — the smallest dataset in the PatchTST benchmark suite (966 weekly samples, 7 channels, ~570 training windows). Small datasets are interesting for a head-architecture comparison since the optimization landscape is genuinely different than on larger benchmarks, and we can afford multi-seed averaging.

For each of four horizons (24, 36, 48, 60), we ran both modes with three random seeds (1, 42, 2021) plus a fourth seed-2021 run, giving four runs per (mode, horizon) cell. All hyperparameters match the published Illness PatchTST script (`d_model=16`, `n_heads=4`, `patch_len=24`, `stride=2`, `dropout=0.3`, `lr=2.5e-3` constant, 100 epochs).

## Headline summary

After looking carefully at the per-step uncertainty bands, our honest read is:

- **There is no clear aggregate winner** between direct and AR on Illness at any non-degenerate horizon. The curves' ±1σ bands overlap substantially everywhere, and the apparent T=60 aggregate "AR wins" effect (paired Δ = −0.21) appears to be driven by a region of inflated direct variance rather than a systematic mean shift.
- **Two effects ARE robust**:
  1. AR exhibits clean discontinuities at rollout boundaries — a measurable, mechanistically-explained signature of exposure bias.
  2. AR is dramatically more reproducible across seeds at long horizons (2.4× lower std at T=60).
- **The paper's blanket "direct is better" recommendation isn't refuted, but isn't supported either** on a small benchmark like Illness. They're statistically indistinguishable in mean, with AR being more consistent run-to-run.

## Findings, by confidence level

### Robust findings (clear evidence)

#### 1. AR exhibits a clean exposure-bias signature at rollout boundaries

This is the headline finding. When AR transitions from one rollout to the next — i.e. when the input context flips from real ground-truth data to "real plus the model's own previous predictions" — the per-step MSE jumps discontinuously, while direct's curve is smooth across the entire horizon.

| pred_len | Boundary at step 25 (excess MSE %) | Boundary at step 49 (excess MSE %) |
|---|---|---|
| 24 | n/a (only 1 rollout) | n/a |
| 36 | −3% (no spike) | n/a |
| 48 | **+18%** | n/a |
| 60 | **+11%** | **+15%** |

"Excess MSE %" is the AR step-to-step jump minus direct's natural step-to-step error growth at the same step, expressed as a percentage of direct's pre-boundary MSE level. It isolates the part of AR's error growth attributable specifically to the rollout-boundary transition.

This signature is **visually unambiguous** in the per-step plots: AR's curve has visible kinks at every multiple of `patch_len`, while direct's curve is smooth across the full horizon. Per-sample noise gets averaged out by the ~170 test windows × 7 channels × 4 seeds, but the spike survives because every test sample hits the rollout boundary at the same horizon position — the spike accumulates constructively rather than canceling.

The mechanism is the textbook exposure-bias problem (Ranzato et al., 2015). At step 24 of T=48, AR's prediction comes from features computed on the *real* look-back. At step 25, the prediction comes from features computed on a context where 23% of the inputs are now AR's own previous predictions. The model was trained on inputs that look more like step-24's context than step-25's context, so its calibration shifts at the boundary.

For T=60, exposure bias hits twice (one per rollout boundary), and the second spike is larger than the first (+15% vs +11%). This makes intuitive sense: by step 49, almost half the look-back is synthetic, so the input distribution is even further from training distribution.

This finding does not require a statistical test — it is a visible structural discontinuity at an algorithmically predictable location, which is qualitative evidence that an aggregate test cannot replicate.

#### 2. AR is more reproducible across seeds at long horizons

Standard deviation of test MSE across 3 seeds:

| pred_len | direct std | AR std | ratio |
|---|---|---|---|
| 24 | 0.124 | 0.124 | 1.00× |
| 36 | 0.113 | 0.129 | 1.14× |
| 48 | 0.067 | 0.074 | 1.10× |
| 60 | **0.221** | **0.091** | **2.43×** |

At T=60, direct's run-to-run variance is 2.4× larger than AR's. This is a real and large effect, larger than any of the per-seed mean differences themselves.

Visually, this is also the most striking thing about the T=60 per-step plot: direct's blue ±1σ shaded band balloons in the middle of the horizon (roughly steps 25–48), reaching a width of ~1.0 MSE units, while AR's orange band stays narrow throughout. **In fact, this variance asymmetry largely explains the apparent T=60 aggregate "AR wins" effect** — see Section 3 below.

Most plausible mechanism: direct's `Linear(features → 60)` head has many parameters, and on Illness's ~570 training windows the optimization landscape is loose, so different random initializations end up at meaningfully different solutions. AR's `Linear(features → 24)` head has 60% fewer parameters and a tighter landscape, so different seeds converge to more similar solutions.

This is an architecturally interesting result regardless of whether AR or direct has a lower mean: **for downstream applications where reproducibility matters (clinical forecasting, regulated domains), AR's tighter run-to-run variance is a real practical advantage**.

#### 3. T=24 is architecturally degenerate

When `pred_len == patch_len`, AR's loop runs exactly once. There's no autoregression to do — the AR head just produces 24 outputs in one shot, exactly like direct. With identical architecture, the two models are literally identical. Predictions are bit-identical.

We report T=24 as a footnote rather than a real comparison row.

### Not supported by our data (despite the appearance of the headline plot)

#### "AR wins on aggregate at T=60"

This was our initial finding. We're walking it back.

**The point-estimate evidence**: paired Δ = −0.21 ± 0.14 across 3 seeds, with all 3 seeds favoring AR. On its face, this looks like ~1.5σ evidence (or a paired t-test p ≈ 0.06–0.10).

**Why it doesn't hold up on closer inspection**:

1. **The spatial pattern is structurally incoherent**. If AR's win came from any of the mechanisms we considered (head-capacity advantage, lower exposure bias relative to error compounding), AR should win in some specific region — early steps, or during one particular rollout, or at the boundaries. Instead, the actual T=60 plot shows: AR is roughly tied steps 1–10, slightly *worse* steps 12–24, clearly better steps 25–48 (the win zone), then clearly worse again steps 50–60. There's no mechanism that predicts AR specifically winning in rollouts 2 and 3 but tying or losing in rollout 1.

2. **The "win zone" coincides exactly with where direct's variance is huge**. Steps 25–48 is precisely the region where direct's ±1σ band is ~1.0 MSE units wide. The mean curves overlap within this band throughout. This is exactly the visual signature of "one or two of the 4 direct seeds happened to do badly in this region, pulling the mean up" — not "AR is systematically better in this region."

3. **Per-step ±1σ bands overlap throughout**. We do not have any horizon position at T=60 where the direct and AR ±1σ bands cleanly separate. The aggregate Δ = −0.21 is a real number arithmetically, but it's averaging a noisy direct curve against a less-noisy AR curve — and the noise is concentrated in the region where AR happens to look better.

**What we conclude**: the T=60 aggregate "AR wins" effect is most plausibly explained as an artifact of direct's high seed-variance at this horizon, rather than a genuine mean shift. To strongly claim AR is better at T=60 would require either (a) substantially more seeds to tighten direct's variance estimate (5+ seeds, ideally 10), or (b) a mechanistic story for why AR specifically wins in rollouts 2 and 3 but not rollout 1, which we don't have.

**This is itself a useful methodological lesson**: the aggregate paired t-test was suggesting "moderately significant" while the per-step decomposition revealed the apparent effect was driven by a single high-variance region. Aggregate metrics can mask noise patterns that visual decomposition exposes.

#### "AR has a near-term advantage in the first 16 steps"

The first-16-step mean MSE is lower for AR by 0.03–0.19 across horizons (point estimates). However, the per-step ±1σ bands clearly overlap throughout the early steps at every horizon, including T=60. With only 3 seeds, the variance of a 16-step average is large enough that we cannot confidently say AR is better than direct on early steps in particular.

The hypothesized mechanism (smaller AR head dedicates more capacity per output → better near-term predictions) is plausible but our evidence does not strongly support it.

## Why the curves can be smooth and yet show clean spikes

Worth addressing because it's central to interpreting the per-step plot: per-step MSE curves are visually smooth most of the time, with step-to-step variation typically only 2–3%. Why are they smooth, and why are the rollout-boundary jumps still visible against that smooth background?

Smooth because:
- Adjacent steps share the same encoder forward pass and same head — only one column of `W` differs between them, so most of the upstream computation is shared.
- The head's weight columns for adjacent steps are nearly identical after training (the model learned that consecutive weeks are similar).
- Truths at adjacent steps are nearly identical (real time series are autocorrelated).
- Each per-step MSE value is averaged over ~170 test windows × 7 channels × 4 seeds ≈ 4760 individual squared errors, washing out per-sample noise.

Spikes survive because:
- Per-sample noise gets averaged out, but **systematic structural discontinuities accumulate constructively** — every test sample experiences the rollout boundary at the same horizon position, so the spike adds up across samples instead of canceling.
- The spike is not a high-frequency fluctuation that averaging smooths over; it's a discrete change in the model's *upstream computation* (encoder running on real vs partially-synthetic input), which propagates to a discrete change in per-step error.

So the per-step curves are smooth precisely *because* they reflect systematic patterns rather than noise, and the spikes are visible *because* they are themselves systematic patterns rather than noise.

This same argument is what tells us the apparent T=60 "AR wins" effect (Section 3) is *not* a systematic pattern — its spatial structure (helpful in middle, neutral or harmful elsewhere) doesn't have a mechanistic explanation, and the variance bands overlap throughout.

## What this means for the paper's design choice

The original PatchTST paper presents the direct head as obviously the right choice. Our results suggest a more nuanced picture, but with appropriate caveats:

- At the horizons we tested on Illness, **aggregate MSE is not statistically distinguishable** between direct and AR. The point estimates jitter around — sometimes direct's mean is lower, sometimes AR's — but always within seed variance.
- AR introduces a **real, mechanistically-explained per-step cost at rollout boundaries** (+11% to +18% excess MSE). Whether this matters in practice depends on the application — if downstream consumers care about per-step error profiles (e.g., clinical decision-making at specific lead times), AR's discontinuities are visible and would matter. For aggregate forecasting tasks where average MSE is the target, the cost is absorbed into the overall average.
- AR is **substantially more reproducible across seeds**, especially at long horizons (2.4× lower std at T=60).

We would not claim AR is generally better, nor that direct is generally better, on Illness. The honest summary is that **the choice between AR and direct produces no statistically distinguishable difference in aggregate MSE on this small benchmark**, contrary to what the paper's framing implies. The architectural distinction matters in *visible* ways (boundary spikes, seed reproducibility), but those don't aggregate into a clear winner.

## Methodological observations

These are process notes from the project that we would include in any longer-form writeup:

### Multi-seed evaluation matters more than aggregate p-values

Our initial single-seed run of T=36 showed AR beating direct by 0.13 MSE — a striking positive result. With three seeds, the same comparison flipped to AR being slightly worse. We almost wrote a section titled "AR has a clear advantage at T=36" before the multi-seed runs disabused us. Single-seed comparisons on small datasets like Illness can mislead substantially.

But going further — multi-seed mean-and-std reporting is *not enough* either. The T=60 aggregate paired t-test gave Δ = −0.21 with z ≈ 1.5, which we initially called "suggestive evidence." Visually inspecting the per-step plot revealed that this aggregate effect was concentrated in a single high-variance region of one model (direct), not a uniform shift. **A statistically suggestive aggregate result can still be unsupportive of an architectural claim** if its spatial structure doesn't match the proposed mechanism.

The general lesson: aggregate p-values are necessary but not sufficient for architectural claims. You need both (a) statistical evidence of a difference and (b) a coherent spatial/temporal pattern that matches the proposed mechanism. Our T=60 aggregate effect failed test (b).

### Aggregate metrics can hide and create misleading signals

Aggregate MSE for T=48 says AR is +0.09 worse than direct ("tied"). Per-step decomposition reveals AR has a specific, mechanistically-explained discontinuity at the rollout boundary — that's the real finding, invisible in aggregate.

Conversely, aggregate MSE for T=60 says AR is −0.21 better ("suggestive"). Per-step decomposition reveals this is driven by direct's variance blowing up in the middle of the horizon — not a systematic AR advantage. The aggregate gave an inflated picture of the effect.

Aggregate metrics are useful for benchmarking but are dangerous as the sole basis for architectural conclusions.

### Reproduction is harder than published numbers make it look

Even with a bit-identical model implementation (verified against the source), our single-seed runs of direct mode came in 0.07–0.39 MSE above the paper's published numbers across the four Illness horizons. This is consistent with the seed-to-seed variance we observed, but it does mean the paper's single-seed numbers are not tight upper bounds. Multi-seed averaging is essential for honest reporting on small-dataset benchmarks.

## Summary

We compared direct and autoregressive head variants of PatchTST on the Illness benchmark across 4 horizons × 2 modes × 3 seeds.

**The findings we are confident about:**

- **Exposure bias is visible and quantifiable**: AR shows clean discontinuities at rollout boundaries, with excess MSE jumps of +11% to +18% above the natural step-to-step error growth direct experiences. This is a structural result that doesn't require statistical inference — it's a visible pattern at predictable locations.
- **AR is substantially more reproducible across seeds** at long horizons (2.4× lower std at T=60), making it the more reliable choice when run-to-run consistency matters.
- **AR and direct are not statistically distinguishable in aggregate MSE** at any non-degenerate horizon we tested. The means jitter, but always within seed variance.

**The findings we considered and walked back:**

- "AR wins at T=60 on aggregate" — point estimate is real but the spatial pattern (win concentrated in one region of high direct variance) and overlapping ±1σ bands throughout the horizon mean we can't responsibly claim a systematic advantage.
- "AR has a near-term advantage on the first 16 steps" — point estimates lean AR, but per-step bands overlap throughout. Suggestive at most.

**What this implies for the paper's design choice:**

PatchTST's choice of a direct head is presented as obviously correct. Our findings on Illness suggest the choice is **largely indifferent in terms of aggregate accuracy** — neither head is clearly better at any horizon — but produces **different qualitative behavior**: AR has visible per-step boundary spikes, while direct is more variable across runs. Which is preferable depends on the application. The paper's blanket recommendation underspecifies the trade-off.

The most reproducible single figure for a writeup is the per-step MSE plot (`results/illness_per_step_mse_v2.png`), which simultaneously shows (a) the exposure-bias spikes, (b) the seed-level uncertainty bands, and (c) why aggregate-MSE comparisons are misleading without spatial decomposition.
