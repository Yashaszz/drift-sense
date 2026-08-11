# R4 handoff notes

Matching and delivery: `matcher.py`, `localize.py`, `confidence.py`, the repo,
CI and packaging.

---

## Status

| Task | Scope | State |
|---|---|---|
| T0 | Repo, CI, `config.py`, `types.py`, frozen stubs | done |
| T2 | Stage 3 — ZNCC correlation surface | done |
| T3 | Stage 3b — peak extraction, NMS | done |
| T4 | Stage 2 — template construction | done |
| T5 | Stage 5 — sub-pixel refinement | done |
| T6 | `localize()` wiring, escalation ladder | done |
| T8 | Uniqueness-weighted correlation | done, **inert** pending R3 |
| T9 | Profiling and caching | done |
| T7 | Stage 6 — confidence calibrator | done, **at chance** pending R2/R3 |

474 tests. Ruff, ruff-format and mypy strict all clean. CI green.

## Headline numbers

Measured on 108 stratified pairs, 1 px tolerance, on the hardware named in
`r4_engineering_notes.md`.

| | value |
|---|---|
| accuracy, all pairs | 38.9% |
| accuracy, **anchored** references | **77.8%** |
| accuracy, unanchored references | 0.0% |
| auto-mode latency, median | 105 ms |
| auto-mode latency, p95 | 146 ms |
| sub-pixel accuracy on known shifts | 0.01 px |

The anchored/unanchored split is the central result and is not a defect. An
unanchored reference is a periodic patch with no aperiodic feature in frame; the
correlation evidence genuinely does not identify a position. Measured on a bare
lattice, **903 positions score exactly the maximum**. That is an
information-theoretic limit, not an algorithmic one, and the correct response is
to answer and flag rather than to succeed.

## Interfaces R4 owns

```python
# matcher.py
build_template(reference, theta, scale, psf_sigma_px) -> FloatArray
build_weight(weight_map, theta, scale)               -> FloatArray
zncc_surface(template, search, weight=None)          -> FloatArray
top_k_peaks(surface, k, nms_radius)                  -> list[Peak]
refine_subpixel(surface, peak, upsample=100)         -> tuple[float, float]
refine_subpixel_crop(template, search, peak, ...)    -> tuple[float, float]
refine_subpixel_detailed(...)                        -> SubpixelRefinement

# localize.py  — THE SUBMITTED DELIVERABLE
localize(search, reference, mode="auto")             -> LocalizationResult

# confidence.py
extract_features(diagnostics)                        -> Float64Array
heuristic_confidence(diagnostics)                    -> float
is_low_confidence(confidence, diagnostics, threshold) -> bool
ConfidenceModel.fit / predict / save / load / load_or_default
```

`localize()` never raises on valid input. Verified against ten adversarial
inputs including read-only arrays, Fortran-order arrays, `bool`, `uint16`,
`int8`, non-square images, a 1x1 search image and Python lists. Every one
returns a finite coordinate with a valid confidence and an honest
`failure_mode`.

## Outstanding dependencies

### R3 — `uniqueness_map` (highest value item in the project)

Still `return np.ones(...)`. The contract is agreed; the body is not written.

R4 is fully wired and requires **no change**. Verified by injecting a stand-in
map and re-running the whole pipeline
(`benchmarks/verify_uniqueness_integration.py`):

| | current stub | non-constant stand-in |
|---|---:|---:|
| accuracy @ 1 px | 38.9% | **42.6%** |
| anchored | 77.8% | **85.2%** |
| unanchored | 0.0% | 0.0% |
| confidence CV AUC | 0.504 | **0.828** |

That is from a crude spectral-flatness stand-in. A proper implementation should
do better. The confidence model is the bigger prize: in the feature-level
counterfactual a working `uniqueness_score` takes cross-validated AUC from
**0.506 to 0.926**, which is the difference between a decorative confidence
number and one that earns the 10% explainability block.

Also outstanding on R3's side:

- `sidelobe_stats` uses a **Euclidean** exclusion disc while R4's NMS suppresses
  a **Chebyshev** square. The two regions disagree, so part of the peak's own
  shoulder is counted as background and PSR is biased low.
- Commit `0ae1bd1` (`TIE_SIGMA 1.0 -> 0.0`) was never merged. R4 has since made
  the equivalent change; the branch can be dropped or rebased.
- `evaluate.py` does not exist. R4 generated its own labels from R1's ground
  truth for the confidence work; that is deliberately minimal and is not a
  substitute for the evaluation harness.

### R2 — `pose.estimate_pose`

Still returns nominal pose with zero quality. `_resolve_pose` already calls it
and discards low-quality estimates, so the real implementation is picked up with
no change in R4.

Consequences today: `theta_est` and `scale_est` are constant, which kills two of
the six confidence features, and rotated or rescaled pairs are matched at
nominal pose. Two things needed from R2:

1. **What is `quality`'s scale and distribution?** `_MIN_POSE_QUALITY = 0.2` is
   a guess and cannot be tuned without knowing the range.
2. **The pose-hypothesis interface is still undecided.** The design mandates
   testing 180-degree and 90-degree alternatives in Stage 3, but `PoseEstimate`
   holds a single value with no room for alternates. Either R2 exposes
   `pose_hypotheses(...) -> list[PoseEstimate]`, or R4 derives the alternates
   from their single estimate. This must be decided before the multi-hypothesis
   loop is written.

**Sign convention, already verified against R1's data:** pass `+rotation_deg`
straight through to `build_template`. Measured on a 3.07-degree pair, `+theta`
gives a correlation peak of 0.9749 and `-theta` gives 0.2522.

### R1 — dataset

Working and used throughout. Two notes:

- `apply_sem_chain` currently lives in `generate_dataset.py` as an identity
  passthrough, but the frozen contract places it in `sem_physics.py`. When R2
  pushes, one of the two must be deleted rather than both existing.
- The dataset carries **no physics at all** — no PSF, no noise, no edge
  brightening — because that stub is identity. Every accuracy and confidence
  number in this document is therefore measured on unrealistically clean data
  and should be re-measured once the physics chain lands.

## Edge cases and known limitations

- **Unanchored references cannot be solved.** By construction, not by defect.
- **PSF estimation declines on periodic layouts.** `estimate_psf_sigma` fits a
  Gaussian rolloff to the radial power spectrum and gates on fit quality. On a
  lattice-dominated spectrum an ungated fit returned a confident ~2.4 regardless
  of the true blur, so it refuses and returns the documented default. Correct
  behaviour, but it means the PSF is effectively never measured on our data.
- **Escalation never short-circuits.** With PSR thresholds at 8.0/4.0 and
  observed PSR in the range 1.4 to 3.4, every pair escalates to the ambiguous
  tier. Safe but slow. The thresholds are deliberately untuned: PSR does not yet
  separate correct from incorrect (correct 1.77-3.08, wrong 1.44-3.41, ranges
  overlapping), and lowering them before uniqueness weighting works would buy
  speed by making every answer falsely confident.
- **`n_tied` is always 1 and `tie_break_used` never fires**, both consequences of
  `TIE_SIGMA = 0.0`. Two more dead confidence features.
- **Recall@K is a real constraint.** `DEFAULT_TOP_K = 30` misses the truth in
  roughly half of ambiguous cases; K=200 recovers some. Not tuned, because it
  should be tuned on data with physics.

## Packaging

Resolved:

- `requirements.txt` generated from the pinned environment. **The editable
  self-install line was stripped** — `pip freeze` emits
  `-e file:///C:/Users/yasha/drift-sense`, an absolute path that would break on
  the graders' machine and which the reproducibility rules forbid.
- `uv.lock` and `.python-version` are committed, so CI and every developer
  machine resolve identically.
- CLI entry point works: `python -m src.localize search.png reference.png --json`.

Outstanding before submission:

- `README.md` is still two lines and needs what it is, how to run it, and the
  headline results.
- `docs/citations.md`, `docs/assumptions.md` and `docs/failure_analysis.md` do
  not exist. Owned by R1/R2/R3 but they gate the 30% and the 10%.
- **The package is importable as `src`**, which matches the frozen repository
  layout but is a poor distribution name and collides with anything else called
  `src` on the path. Worth revisiting if the submission is ever installed rather
  than unzipped.
- **No clean-machine test has been run yet.** Unzipping on a laptop that has
  never seen the project and running it end to end is the single highest-value
  remaining packaging task.
- No trained confidence model ships, by design. `config.CONFIDENCE_MODEL_PATH`
  is looked up and absence falls back to the heuristic, so the deliverable runs
  from a clean unzip with no artefacts.

## Reproducing every number

See `benchmarks/README.md`.
