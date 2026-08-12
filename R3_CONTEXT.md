# R3 Context — Drift-Sense PS-02

**Purpose:** persistent context for R3 (Disambiguation & Evidence). Upload to
project knowledge so code state does not have to be re-pasted each session.

**Last synced:** 2026-08-12, after the 324-pair sweep and the PR #14 rebase.
**Repo:** `~/Developer/drift-sense` · **Teammates:** R1 geometry/dataset, R2 physics/pose, R4 matcher/delivery

---

## 1. State of play

| Item | Owner | Status |
|---|---|---|
| `peak_to_sidelobe` / `sidelobe_stats` | R3 | done |
| `select_candidate` + centre tie-break | R3 | done, wired (PR #10) |
| `evaluate.py` | R3 | done (PR #11) |
| `uniqueness_map` / `uniqueness_score` | R3 | **done (PR #13)** — real implementation, `src/uniqueness.py` |
| `failure_analysis.md` | R3 | **done (PR #13)** — `docs/failure_analysis.md`, 166 lines |
| `docs/citations.md` | R3 | not started — 30% block, gated on R2's physics chain |
| `R3_CONTEXT.md` refresh | R3 | this document |
| T8 uniqueness-weighted correlation | R4 | done, live |
| T9 profiling and caching | R4 | done |
| T7 confidence calibrator | R4 | done, at chance pending R2/R3 features |
| `apply_sem_chain` | R2 | **live** — real Poisson shot + read noise per stratum |
| `estimate_pose` | R2 | still returns nominal pose, zero quality |

---

## 2. Headline results

324 stratified pairs, 1 px tolerance, **real SEM noise strata**
(`uv run python -m src.evaluate --data dataset --gt dataset/ground_truth.jsonl --out results/full_324.csv`):

| stratum | n | plain NCC | full pipeline | median err |
|---|---|---|---|---|
| overall | 324 | 0.377 | **0.426** | 111.7 px |
| anchored | 162 | 0.753 | **0.852** | **0.042 px** |
| unanchored | 162 | 0.000 | 0.000 | 442.6 px |

Ceiling on this dataset is **0.500** — R1's generator asserts an unanchored
layout carries an empty anchor list, so 162 cases are information-theoretically
unsolvable. Quote anchored 0.852, not overall 0.426, and always label the
dataset.

**Never put 0.852 and the old 0.833 in one sentence.** They are different
datasets (324 with noise vs 108 without), not an improvement.

Latency: median 206 ms end to end on the 324 set.

**The +0.049 overall gap against plain NCC is not what disambiguation buys.**
The baseline is handed nominal pose (θ=0, s=10), which is correct only on the
`pose-none` stratum. There the gap is **+0.019** — that is the disambiguation
figure. The rest is the absence of pose estimation in the baseline.

---

## 3. Stage 4a design, as built

`src/uniqueness.py`, re-exported through `src.disambiguate` so R4's imports are
unchanged.

- **Statistic:** tile the reference at 50% overlap; score each tile
  `1 - max_normalised_autocorrelation_sidelobe`. A tile's ACF *is* its ambiguity
  function as a matched filter, so this measures positional ambiguity directly
  rather than by proxy (the alternative considered was spectral flatness).
- **Prefilter:** Gaussian σ = 4 px reference, near the 10× decimation Nyquist.
  Removes frequencies the matcher physically cannot see, so noise does not
  drive the score.
- **Hann taper:** FFT autocorrelation is circular; without the taper, wraparound
  manufactures periodicity in aperiodic tiles.
- **Exclusion:** Chebyshev, radius `3σ`. Must stay well below the smallest layout
  pitch in reference px (**≥72 px** at 1 nm/px — FinFET fin at −20% of its
  90 nm nominal, since `PITCH_TOLERANCE = 0.20` randomises pitch over
  72–108 nm). If it exceeded pitch, periodic
  sidelobes would be excluded and periodic tiles would score as unique.
- **Absolute scale, never per-image normalised.** Rescaling so the best tile
  reads 1.0 would promote the least-periodic tile into a false anchor on an
  unanchored reference. Flat map → masked degrades to unmasked → escalation
  fires. This is the single most important design decision in the module.
- **Bilinear expand, not blocky repeat** — a step-function weight map adds
  synthetic periodic structure to the matched filter.
- `uniqueness_score(weight_map) = p99 - median`. Near zero means no anchor in
  frame. Takes the map, not the reference, so the reported scalar always
  describes the map that actually weighted the correlation.

---

## 4. Frozen contracts

- **`uniqueness_map(reference, tile=64) -> FloatArray`** — shape == reference
  shape, `float32`, values in `[UNIQUENESS_FLOOR, 1.0]` (floor = 0.05), never
  NaN, never all-zero, deterministic, reference resolution (1 nm/px).
- **`matcher.build_weight(weight_map, theta, scale)`** — R4 carries the map onto
  the template grid through the same rotation, valid-area crop and area-average
  decimation as `build_template`. No PSF, no standardisation.
  `_zncc_masked_fft` normalises by weight sum, so overall weight scale is
  irrelevant; a constant map reproduces unmasked ZNCC to float32 precision, and
  the pipeline skips weighting entirely when the map is constant.
- **`Peak`** — frozen: `col`, `row`, `score`, `.centre(template_shape)`.
  Coordinates `(x=column, y=row)` unconditionally.
- **Surface indexing** — `surface[r, c]` is a top-left corner in valid-mode
  correlation. Convert via `config.window_topleft_to_centre()` / `Peak.centre()`.
- **Absent measurements** — NaN, never 0.0. R4's escalation treats NaN as
  unknown → escalate.
- **Ownership** — R4 owns `confidence`, `low_confidence_flag` (`src/confidence.py`).
  R3 owns `psr`, `n_tied`, `uniqueness_score`, `tie_break_used`.
- **Coordinates** — pixel-centre origin; reference 1 nm/px, search 10 nm/px
  (10× *linear*). Downsampling is area-averaging only.

---

## 5. Config constants R3 owns
TIE_SIGMA = 0.0 # exact ties only — see §6
UNIQUENESS_PREFILTER_SIGMA_PX = 4.0
UNIQUENESS_FLOOR = 0.05
UNIQUENESS_MIN_TILE_PX = 16
PSR_EXCLUSION_RADIUS_PX = 8 # split from DEFAULT_NMS_RADIUS_PX
`PSR_EXCLUSION_RADIUS_PX` and `DEFAULT_NMS_RADIUS_PX` are both 8 today **by
coincidence** and are set by different physics — correlation main-lobe width
versus lattice pitch. They must not be recoupled when R1 randomises pitch.

---

## 6. Hard-won lessons

- **`TIE_SIGMA = 1.0` was silently broken.** A sigma width in sidelobe standard
  deviations was passed as a raw score tolerance on ZNCC's [-1, 1] range, tying
  all 30 candidates on 102 of 105 cases and handing every answer to the centre
  prior blind. Unit conversion belongs at the call site. Side effect of the fix:
  `n_tied` is always 1 and `tie_break_used` never fires, so both are dead as
  confidence features.
- **PSR does not separate correct from incorrect on periodic layouts.** The
  sidelobe region contains genuine lattice peaks, not noise, so the background
  the statistic normalises against is itself signal. DRAM mean 2.52 vs FinFET
  2.13; earlier per-case data had the highest PSR on a *wrong* answer. With
  thresholds at 8.0/4.0, **320 of 324 cases escalate** — 3 accept at `robust`,
  1 at `fast`. Thresholds are deliberately untuned — lowering them buys speed
  by making wrong answers confident.

  **PSR is not capped near 3.4.** That figure is from the 108 set and is wrong
  on 324: the observed maximum is **12.314**, with 2 cases ≥ 8.0 and 4 ≥ 4.0.
  The "everything escalates" conclusion survives at 98.8%, but quoting 3.4 as
  the ceiling is falsifiable by opening the CSV.
- **`sidelobe_stats` used a Euclidean disc while NMS uses a Chebyshev square.**
  The regions disagree at the corners, so part of the peak's shoulder counted as
  background and PSR was biased low. Fixed in PR #13; PSR values shifted.
- **A flat constant patch is not an aperiodic anchor.** It is an absence of
  features and scores *low*, correctly — a matched filter cannot localise on it
  either. Test fixtures must use aperiodic *texture*.
- **Noise suppresses uniqueness on a periodic field, it does not inflate it.**
  Broadband noise drives all sidelobes down uniformly, so every tile scores
  alike and the map stops discriminating. The prefilter restores discrimination;
  test the anchored/unanchored *gap*, not the level.
- **Unanchored 0% is a ceiling, not degradation.** Must be documented as such.
- **`gh pr merge` needs an upstream.** A branch with no tracking ref silently
  fails `git push --force-with-lease`; use `--set-upstream`.
- **Never put a ``` code fence inside a shell heredoc.** zsh drops into
  `cmdand heredoc>` and the file is never written.
- **Stale VS Code buffers:** `git status --short` before branch switches. A
  struck-through tab means the file is gone from disk but live in memory —
  Cmd+S rescues it.

---

## 7. What remains

**Done, on `r3-recall-baseline` (PR #14):** `src/baseline_ncc.py`, `src/recall.py`
and `src/ablate.py` are complete, with 324-pair outputs tracked under `results/`.
They were previously listed here as open. PR #14 is rebased onto `main`, CI
green, blocked only on review approval.

Still open:

1. `docs/citations.md` — 30% augmentation-realism block. **Unblocked**: R2's
   chain is real, so augmentation realism is now citable.
2. Citation audit across all docs.
3. Confidence-vs-accuracy calibration plot (Phase 3). Expect it flat — see §5
   of `failure_analysis.md`.
4. Open for R4: cache the uniqueness map per reference; re-fit the confidence
   calibrator now that `uniqueness_score` populates (their counterfactual put
   CV AUC at 0.506 → 0.926).
5. **Uniqueness weighting contributes nothing to final accuracy on 324.** The
   ablation reads `selected` 0.417 and `weighted` 0.417 — identical to the digit
   on both success and median error — while recall finds the peak in 8 more
   cases weighted than unweighted. The gain exists at ranking and dies before
   the final answer. Diagnose before the deck.

---

## 8. Working conventions

- `uv run` for everything; ruff pinned `v0.16.1`, `--no-fix`; pre-commit enforced.
  `W293` (whitespace on blank docstring lines) has broken commits repeatedly.
- Branch naming `r3-<feature>`.
- Dataset lives in `dataset/` (324 pairs, real noise strata). `dataset_full/`
  is the superseded 108 set — its manifest and ground truth stay tracked as
  provenance, its images are gitignored. **Never quote a 108-set number.**
- **Generation is byte-reproducible across platforms.** Verified 2026-08-12:
  R1 pre-registered the holdout's `image_tree_sha256` *and* `file_tree_sha256`
  before generation, and a Mac run reproduced both exactly against their
  Windows values (`d51df27b…` / `dcdcb969…`). Regenerate from seed; zips are
  no longer the single source.

---

## 9. Phase status (as of Aug 11, late)

Phases 0, 1, 2 complete for R3. Phase 3 build work complete. Phase 4's
`failure_analysis.md` written early but its figures predate the ablation.

**Done tonight, PR #14 (`r3-recall-baseline`), open against `main`:**
1. **recall@K** — `src/recall.py`. Reuses `_StageCache` so pose, template and
   PSF match `localize()`. Sweeps weighted on/off and `nms_radius` 2-16.
   On 324: anchored weighted r@1 0.833, r@30 **0.852**. Top-1 accuracy on
   anchored is also 0.852, so **top-1-given-recall is 1.000** — selection loses
   nothing once truth is in the list, and the 24 remaining anchored failures
   are cases where truth was never a candidate (Stage 1/3, not Stage 4).
   K=1 → K=30 buys only +0.019, so candidate depth is not the bottleneck.
   Recall is flat across every NMS radius tested — radii 2/4/8 are bit-identical
   and 16 moves one case — so the suppression concern does not hold.
   Aug 11 gate is closed.

   **Reporting hazard:** `rank = -1` means truth never entered the top 30.
   Those rows are misses. Dropping them from the denominator yields a spurious
   r@30 = 1.000, a 2.3x overstatement. `src/recall.py` is correct; the hazard
   is in downstream re-analysis of the CSV.
2. **`baseline_ncc.py`** — exists. Single-scale ZNCC at nominal 10x, argmax,
   no pose / weighting / disambiguation / sub-pixel. Anchored **0.753** at
   0.454 px median, 98 ms.
3. **Ablations** — `src/ablate.py`, anchored stratum (n=162), on 324:

   | stage | acc | median err | ms |
   |---|---|---|---|
   | ncc | 0.753 | 0.454 | 32 |
   | + weighting | 0.833 | 0.456 | 159 |
   | + selection | 0.833 | 0.456 | 162 |
   | + sub-pixel (full) | **0.852** | **0.042** | 207 |

   Weighting is the entire disambiguation gain (+0.080). **Selection buys
   exactly 0.000** — with `TIE_SIGMA = 0.0` the tolerance is always zero,
   `n_tied` always 1, and `select_candidate` returns `peaks[0]` unchanged.
   The tie-break is mandated and fires correctly when ties exist; on this
   dataset exact ties never occur. Sub-pixel buys +0.019 and an 11x error
   reduction; 0.454 px is the integer-peak quantisation floor.

   **Caveat on the overall-stratum ablation.** Across all 324 (not just
   anchored) `selected` and `weighted` both read 0.417 / 111.7 — weighting
   appears to contribute nothing. It does contribute (+0.080 on anchored);
   the unanchored half, pinned at 0.000 by construction, halves the visible
   effect and the rounding hides the rest. **Always cite the anchored
   stratum for the ablation.**

Every row cross-checks against an independent code path.

**Blocked on R4:** confidence-vs-accuracy calibration plot — needs the
calibrator re-fit now that `uniqueness_score` populates (it was at chance
when `uniqueness_score` was NaN). Also asked for `refine: bool = True` on
`localize()` so the sub-pixel ablation row comes from the same harness as
the other three instead of from `uniqueness_on.csv`.

**Blocked on R2:** `docs/citations.md` compile + audit. Re-measuring every
figure once `apply_sem_chain` is real.

**Parked, not dropped:** `src/disambiguate.py`'s module docstring still says
`uniqueness_map` returns uniform weights and `select_candidate` returns the
strongest peak. Both stale since PR #13.

**Hard date:** Aug 13 is feature freeze. Aug 15 deck + zip + clean-machine test.
