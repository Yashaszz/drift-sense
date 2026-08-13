# R4 engineering notes

Three findings from the matching and delivery work that changed the
implementation, each with the measurement that drove it. Written up separately
from the handoff notes because these are the parts worth putting in the deck.

Hardware for every timing below: Windows 11, AMD Zen 3, Python 3.12.13,
OpenCV 5.0.0, NumPy 2.5.1. Dataset: 108 stratified pairs from
`src/generate_dataset.py` at `seeds_per_cell=9`.

---

## 1. The tie-break tolerance was costing an order of magnitude of accuracy

**Symptom.** End-to-end accuracy at the 1 px headline tolerance was 5.6%, on a
pipeline whose individual stages all tested clean.

**Diagnosis.** Not a matcher failure. On `dram_anchored_pose-none_0003` the
correlation argmax landed on *exactly* the true top-left corner, `(183, 686)`,
and `Peak.centre()` converted it to `(232.5, 735.5)` against a ground truth of
`(232.66, 735.58)`. The right answer was found and then discarded.

It was discarded by the mandated centre tie-break. The call site converts
`config.TIE_SIGMA` into score units by multiplying by the sidelobe standard
deviation:

```python
tolerance = config.TIE_SIGMA * sidelobe_std
```

On a periodic layout the sidelobe region spans nearly the whole correlation
range, so its standard deviation is about **0.29**. The top thirty candidates
are separated by about **0.025 in total**. With `TIE_SIGMA = 1.0` every
candidate was therefore declared statistically tied, and the centre rule chose
among them — discarding a peak standing 0.023 above the rest.

**Root cause.** The handbook specifies ties as "within roughly one *noise* sigma
of the maximum". Noise sigma is the scale of random fluctuation *between
neighbouring candidates*. Sidelobe sigma is the spread of the *entire surface*.
Conflating them is what cost the accuracy.

**Measurement.**

| tie width | accuracy @ 1 px | median candidates tied |
|---:|---:|---:|
| 0.000 | **35.2%** | 1 |
| 0.002 | 32.4% | 30 |
| 0.005 | 31.5% | 30 |
| 0.010 | 25.0% | 30 |
| 0.020 | 20.4% | 30 |
| 0.050 | 3.7% | 30 |
| 0.291 | 3.7% | 30 &larr; one sidelobe sigma |

**Resolution.** `TIE_SIGMA = 0.0`, exact ties only. R3 reached the same
conclusion independently in commit `0ae1bd1`, which was never merged.

**This is a floor, not the answer.** Zero disables the mandated tie-break in
practice, which is the wrong long-run behaviour: the rule exists and the
problem statement requires it. The proper fix is to express the tie width in the
*local fluctuation scale among the top candidates* rather than in the spread of
the whole surface, which restores the tie-break to the cases it was meant for.
That work is not done.

**Transferable lesson.** A threshold expressed in units of a statistic is only
as good as the statistic. "One sigma" is meaningless until you say *sigma of
what*, and the two candidate answers here differed by more than a factor of ten.

---

## 2. Uniqueness-weighted correlation (T8)

**What it does.** Extends ZNCC with a per-pixel weight over the template, so
informative regions of the reference count for more than periodic ones. Every
mean and variance is taken *under the weights*.

**Formulation.** With `w` normalised to sum to one and `mu_T = sum(w * T)`:

```
numerator   = sum(w * (T - mu_T) * (S_uv - mu_S))
denominator = sqrt( sum(w * (T - mu_T)^2) * sum(w * (S_uv - mu_S)^2) )
```

Expanding removes every per-position sum over the template, leaving three plain
cross-correlations that OpenCV evaluates directly:

```
C1 = xcorr(S,    w * T)        numerator = C1 - mu_T * C2
C2 = xcorr(S,    w)            var_T     = sum(w * T^2) - mu_T^2   (scalar)
C3 = xcorr(S**2, w)            var_S     = C3 - C2^2
```

**Why normalise by the weight sum.** It is what makes a *constant* weight map
reproduce plain ZNCC exactly: every weighted mean collapses to the ordinary mean
and the common factor cancels between numerator and denominator. That gives the
weighted path an exact expected answer to be tested against, which is the only
reason it is verifiable at all while `uniqueness_map` still returns a constant.

Measured: constant maps at levels 0.001, 0.05, 0.5, 1.0 and 7.3 all agree with
the unweighted surface to **1.8e-07** — float32 rounding.

**Numerical conditioning.** Both inputs are standardised before correlating.
This cannot change the result, because weighted ZNCC is invariant to positive
affine rescaling of either input, but it matters: the formulation subtracts
`(sum(w*S))^2` from `sum(w*S^2)`, and on raw 8-bit data those terms are of order
1e4 and nearly equal. float32 carries about seven significant digits, so the
difference would lose most of its precision to cancellation.

**Carrying the weight onto the template grid.** `matcher.build_weight` applies
the *same* rotation, the same valid-area crop and the same area-average
decimation as `build_template`, so the two stay aligned pixel for pixel. Two
steps are deliberately **not** applied:

- **No PSF blur.** The point-spread function models how the instrument smeared
  the *signal*. The weight map is not signal — it is a statement about which
  parts of the reference are informative. Blurring it would bleed an anchor's
  importance into neighbouring periodic regions and dilute exactly the
  discrimination the weighting exists to provide.
- **No standardisation.** The weights are a non-negative importance profile.
  Centring them would make roughly half negative, which is meaningless in a
  weighted variance, and rescaling is pointless because the correlation
  normalises by their sum regardless.

**Cost.** 69.3 ms against 34.6 ms unweighted, a factor of **2.01**. Cheaper than
the 3x the three-correlation form suggests, because three raw `TM_CCORR` passes
beat one normalised `TM_CCOEFF_NORMED`.

**No longer inert — this is now the stage that earns the most.** The paragraph
here previously read "currently inert, and skipped rather than paid for",
measured while `uniqueness_map` returned a constant. R3's map has been real
since PR #13, so the skip no longer fires on normal input and weighting is paid
for and worth it: **+0.080 accuracy on the anchored stratum** (0.753 → 0.833),
the single largest contribution of any stage (`results/ablation_324.csv`).

The constant-map skip remains as an equivalence, not an approximation, and
still fires when a reference genuinely has no informative structure — which is
the case it was written for. The 105 ms / 263 ms figures in the original note
were from the superseded 108-pair set and are not comparable to current
latency; see the README for the current figure and its machine.

---

## 3. Escalation caching (T9)

**Symptom.** Auto mode cost 360 ms against a 100 ms target.

**Diagnosis.** Structural, not algorithmic. Profiling per stage across the three
escalation tiers:

| stage | calls/pair | ms/pair | % total |
|---|---:|---:|---:|
| `matcher.zncc_surface` | 3 | 68.5 | 28.0% |
| `matcher.build_template` | 3 | 65.8 | 26.9% |
| `matcher.estimate_psf_sigma` | 2 | 40.8 | 16.7% |
| `disambiguate.sidelobe_stats` | 3 | 23.5 | 9.6% |
| `disambiguate.peak_to_sidelobe` | 3 | 22.6 | 9.3% |
| `matcher.top_k_peaks` | 3 | 14.9 | 6.1% |
| `matcher.refine_subpixel_detailed` | 3 | 8.0 | 3.3% |
| **total** | | **244.3** | |

Every heavy stage ran once per tier on identical inputs. Only two things vary
between tiers — the pose hypothesis and the PSF width — and everything
downstream is a pure function of those plus the image pair.

**Resolution.** `_StageCache` memoises the whole tier outcome on
`(theta, scale, psf_sigma, weighted)`, and estimates the PSF at most once since
it depends only on the search image.

**Measurement.**

| | median | p95 |
|---|---:|---:|
| before | 360.4 ms | 368.2 ms |
| after | 105.4 ms | 145.6 ms |
| speedup | **3.42x** | 2.53x |

**Verified behaviour-preserving, not assumed.** All eleven diagnostic fields —
`x`, `y`, `psr`, `n_tied`, `tie_break_used`, `mode_used`, `ncc_peak`,
`subpixel_error`, `subpixel_method`, `confidence`, `low_confidence_flag` — were
compared across all twelve pairs of a held dataset, 132 values, before and after.
**Bit-identical.**

**The key separates tiers that genuinely differ.** On the test fixture the PSF
estimator returns 1.17 against a default of 1.0, so `fast` builds a different
template from `robust` and the two are *not* merged. A key that ignored the PSF
width would have looked even faster while silently disabling the robust path.
There is a test for exactly that, because a cache that quietly deletes work is
the failure mode this optimisation invites.

---

## Known remaining inefficiency

`disambiguate.sidelobe_stats` costs about 7.6 ms per call to compute a mean and
standard deviation over 811k floats, roughly thirty times slower than necessary.
It materialises an int64 distance array, a boolean mask and a fancy-indexed copy.
Rewriting it is R3's call and it is no longer on the critical path.
