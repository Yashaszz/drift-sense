# R3 Context — Drift-Sense PS-02

**Purpose:** persistent context for R3 (Disambiguation & Evidence). Upload to
project knowledge so code state does not have to be re-pasted each session.

**Last synced:** 2026-08-11, after PR #13 merged to `main`.
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
| `apply_sem_chain` | R2 | **still identity passthrough** |
| `estimate_pose` | R2 | still returns nominal pose, zero quality |

---

## 2. Headline results

108 stratified pairs, 1 px tolerance, **no-noise renders**
(`uv run python -m src.evaluate --data dataset_full --out results/uniqueness_on.csv`):

| stratum | n | before 4a | after 4a | median err |
|---|---|---|---|---|
| overall | 108 | 0.389 | **0.417** | 115.4 px |
| anchored | 54 | 0.778 | **0.833** | **0.028 px** |
| unanchored | 54 | 0.000 | 0.000 | 455.3 px |

Ceiling on this dataset is **0.500** — R1's generator asserts an unanchored
layout carries an empty anchor list, so 54 cases are information-theoretically
unsolvable. Quote anchored 0.833, not overall 0.417, and always label as
no-noise.

Latency: median 351 ms with weighting vs 105 ms without. Map depends only on
the reference and is recomputed per case — caching is open, R4's call.

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
  pitch in reference px (≥40 px at 1 nm/px). If it exceeded pitch, periodic
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
  the statistic normalises against is itself signal. Anchored mean 2.76 vs
  unanchored 2.17; earlier per-case data had the highest PSR on a *wrong*
  answer. With thresholds at 8.0/4.0 and observed PSR capped near 3.4, every
  case escalates. Thresholds are deliberately untuned — lowering them buys speed
  by making wrong answers confident.
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

1. `docs/citations.md` — 30% augmentation-realism block. Gated on R2's physics
   chain landing; cannot cite augmentation realism for an identity passthrough.
2. Citation audit across all docs.
3. Re-measure everything once `apply_sem_chain` is real. Every number in
   `failure_analysis.md` is labelled no-noise and will move.
4. Open for R4: cache the uniqueness map per reference (351 ms → ~105 ms);
   re-fit the confidence calibrator now that `uniqueness_score` populates
   (their counterfactual put CV AUC at 0.506 → 0.926).

---

## 8. Working conventions

- `uv run` for everything; ruff pinned `v0.16.1`, `--no-fix`; pre-commit enforced.
  `W293` (whitespace on blank docstring lines) has broken commits repeatedly.
- Branch naming `r3-<feature>`.
- Dataset lives in `dataset_full/` (108 pairs, `ground_truth.jsonl`,
  `dataset_manifest.json`). `dataset/` is gitignored and empty. R1's zip is the
  single source — identical ground truth does not imply identical renders
  across Mac and WSL.

---

## 9. Phase status (as of Aug 11)

Phases 0, 1, 2 complete for R3. Phase 4's `failure_analysis.md` written early.

**Open on R3's plate, not blocked:**
1. **recall@K** — the Aug 11 gate reads "recall@K and top-1 measured separately".
   Top-1 is measured; `evaluate.py` still prints `NOTE: top-1 only. recall@K needs
   the pre-disambiguation candidate list — see recall_at_k_pass()`. recall@K is
   R4's number but R3's harness has to produce it. Gate is not closed until it does.
2. **`baseline_ncc.py`** — Phase 1 deliverable, never confirmed to exist. The
   incumbent to beat. Phase 3's baseline-vs-ours table and Phase 5's results slide
   both depend on it.
3. **Ablations** (Phase 3, Aug 12–13) — what each stage actually buys. Stage 4a now
   has a clean before/after (0.778 → 0.833 anchored) to build the rest on.
4. **Confidence-vs-accuracy calibration plot** (Phase 3).

**Blocked on R2:** `docs/citations.md` compile + audit (R1/R2 write entries beside
their own numbers). Re-measuring every figure once `apply_sem_chain` is real.

**Hard date:** Aug 13 is feature freeze. Aug 15 deck + zip + clean-machine test.
