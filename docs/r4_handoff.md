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
| T8 | Uniqueness-weighted correlation | done, **live** — R3's map has landed |
| T9 | Profiling and caching | done |
| T7 | Stage 6 — confidence calibrator | done, **at chance** (CV AUC 0.552) |
| P3 | Threshold tuning | done — swept and validated; 8.0/4.0 retained |
| P3 | Low-confidence flag hardening | done |
| P3 | Per-stage timing instrumentation | done, opt-in |
| P3 | Uniqueness-map cross-call cache | done — 4.3x on a warm reference |
| P3 | Full evaluation on 324 physics pairs | done |

586 tests on `main`. Ruff, ruff-format and mypy strict all clean. CI green,
and the clean-room check passes on the merged tree.

## Headline numbers

Measured on **324 stratified pairs carrying the full SEM physics chain**, 1 px
tolerance, on the hardware named in `r4_engineering_notes.md`. All 324 scored,
no load or run failures, `failure_mode: none` throughout.

| | value |
|---|---|
| accuracy, all pairs | 46.9% |
| accuracy, **anchored** references | **93.8%** (152/162) |
| accuracy, unanchored references | 0.0% |
| median error, anchored | 0.035 px |
| auto-mode latency, median / p95 | 725 ms / 1281 ms (Windows, post-pose) |
| confidence, cross-validated AUC | 0.552 |

**Latency is quoted per machine and accuracy is not, because only one of them
varies.** Accuracy reproduces to the digit across both of our machines. Latency
does not: the same 324 pairs run 206 ms median on R3's Apple Silicon Mac
(`results/full_324.csv`) and 356 ms on R4's Windows 11 / AMD Zen 3, a 1.7x gap.
Harness overhead is ruled out — wall-clock and `Diagnostics.elapsed_ms` agree to
0.11 ms — so it is the machine. The README quotes the Mac figure because R3 owns
the evaluation harness and its CSVs are the tracked evidence; the stage table
below is Windows. **Any latency number on a slide must name its machine.**

The anchored/unanchored split is the central result and is not a defect. An
unanchored reference is a periodic patch with no aperiodic feature in frame; the
correlation evidence genuinely does not identify a position. That is an
information-theoretic limit, not an algorithmic one, and the correct response is
to answer and flag rather than to succeed. The measurement that makes this
concrete: escalating an unanchored case buys **+0.0** accuracy. The
corresponding anchored-side figure was measured before Stage 1 and is not
requoted here.

### What physics, then pose, changed

Two dataset-level events superseded earlier figures. Physics landing did not
move accuracy; Stage 1 landing moved it a great deal.

| | 108, clean | 324, physics | **324, +pose** |
|---|---|---|---|
| accuracy, all pairs | 41.7% | 42.6% | **46.9%** |
| accuracy, anchored | 83.3% | 85.2% | **93.8%** |
| accuracy, unanchored | 0.0% | 0.0% | 0.0% |
| median error, anchored | 0.028 px | 0.042 px | **0.035 px** |

**The physics chain did not cost accuracy.** Across noise strata anchored
accuracy now reads 90.7% / 94.4% / 96.3% for low / medium / high — it rises with
noise. ZNCC is normalised, so the current magnitudes do not move it, and the
ordering runs the wrong way to be a noise effect. Worth stating deliberately
rather than claiming noise robustness by accident.

**Pose was the dominant degradation axis, and is no longer.** Anchored accuracy
used to fall from 0.963 at `pose=none` to **0.667** at `pose=large` — the worst
stratum in the system. Post-Stage-1 it reads 0.944 / 0.944 / **0.926**, so a
0.296 collapse became 0.018. That is where the +0.086 anchored came from.

**Rotation is estimated; scale is not.** `theta_est` varies over -8.44 to 7.73
degrees on 239 of 324 pairs, while `scale_est` is exactly 10.0 on all 324. One
of the two pose residuals is live, the other still pinned — unclaimed headroom
rather than a defect, since the dataset does carry scale mismatch.

### PSR and the escalation thresholds

PSR collapsed relative to the thresholds: median **2.25**, p95 3.05, against
`PSR_ACCEPT_THRESHOLD = 8.0`. **320 of 324 cases escalate to the ambiguous
tier**, so the ladder is effectively inert.

Retuning does not fix this. `AUC(psr -> correct)` is 0.633 post-pose — its best
showing yet, and still weak — so PSR barely separates correct from wrong and
every millisecond bought costs accuracy near-linearly. The sweep below was
measured **pre-Stage-1** and has not been re-run; the ordering it establishes
holds, the absolute rows do not:

| accept | stop at fast | accuracy | mean ms | false accepts |
|---:|---:|---:|---:|---:|
| 8.0 (current) | 0.3% | 42.6% | 356 | 0 |
| 3.0 | 11.7% | 42.3% | 321 | 21 |
| 2.0 | 57.7% | 39.8% | 180 | 114 |
| 0.0 (fast only) | 100% | 38.9% | 51 | 198 |

The thresholds are therefore **kept at 8.0 / 4.0**: they are the only setting
with zero false accepts, and no alternative dominates. The problem is the
statistic, not the number.

### Latency breakdown

Per-stage, from `Diagnostics.stage_ms` (enable with
`config.COLLECT_STAGE_TIMINGS`). Cold, which is what a sweep over distinct
references measures. Absolute figures are Windows 11 / AMD Zen 3 and were taken
**pre-Stage-1**, so the totals no longer match the headline latency; **the
percentage column is the portable one** and should be what gets quoted.

| stage | mean ms | % of call |
|---|---:|---:|
| `uniqueness_map` | 215.7 | 60.6% |
| `correlate` | 103.3 | 29.0% |
| `psf_sigma` | 17.6 | 4.9% |
| `sidelobe_stats` | 13.8 | 3.9% |
| `refine_subpixel` | 4.5 | 1.3% |
| `select_candidate` | 0.1 | 0.0% |

### The blocking finding: nothing detects anchoredness

Accuracy is perfectly bimodal, so a feature that separated anchored from
unanchored would nearly solve confidence outright — `AUC(anchoredness ->
correct) = 0.971`. Nothing available does:

| statistic | AUC vs anchoredness |
|---|---:|
| `psr` | 0.631 |
| `uniqueness_score` as wired (`mean` of the map) | 0.576 |
| R3's `uniqueness_score()` (`p99 - median`) | **0.493** |

R3's scorer documents itself as "this should separate R1's anchored stratum from
the unanchored one. If it does not, the map is not working." On this dataset it
does not. Switching `localize.py` to it was measured and **rejected** — at 0.493
it is worse than the current wiring, so the mean stays.

This one gap blocks two things at once: the confidence calibrator (CV AUC 0.552,
two of six features still dead constants) and a free 43% latency cut, since
skipping escalation on unanchored cases would give 203 ms at identical accuracy.

The map itself is not the problem — as a *weighting* mechanism it earns its
keep. It is the scalar summary that carries no signal.

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

### R3 — an anchoredness signal (highest value item in the project)

**Landed since this was written:** `uniqueness_map` is implemented and live in
`src/uniqueness.py`, `evaluate.py` exists, and the Euclidean/Chebyshev exclusion
mismatch in `sidelobe_stats` is fixed. `TIE_SIGMA` is 0.0 on `main`, so commit
`0ae1bd1` is moot and `r3-disambiguate` is stale — it predates the
centre-versus-corner tie-break fix and would regress `main` if merged.

What remains is narrower and sharper than "implement the map". As a *weighting*
mechanism the map works: it is worth +7.4 accuracy points on anchored references
through escalation. What does not work is any **scalar summary** of it as an
anchoredness detector — see "The blocking finding" above. Until some diagnostic
separates anchored from unanchored, the confidence calibrator stays near chance
and the unanchored escalation saving stays unavailable.

The earlier counterfactual on this page predicted CV AUC 0.926 from a working
`uniqueness_score`. That did **not** reproduce: measured on the physics dataset
with the real map it is **0.570**. The 0.926 came from a crude spectral-flatness
stand-in on geometry-only data and should not be quoted again.

R4 requires no change to consume a better signal — the feature is already
extracted, wired and fitted.

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

Working and used throughout. **Both notes previously on this page are closed:**
`apply_sem_chain` now lives only in `src/sem_physics.py` and is imported by the
generator, the duplicate is deleted, and the dataset carries the real physics
chain across low/medium/high noise strata. Every number above is measured on
that data.

One observation for R1 and R2: the noise strata do not currently produce a
difficulty gradient — anchored accuracy is 83.3% / 83.3% / 88.9% across low /
medium / high. Either that is a robustness result to claim deliberately, or the
strata need wider separation before anyone cites noise robustness.

## Edge cases and known limitations

- **Unanchored references cannot be solved.** By construction, not by defect.
- **PSF estimation declines on periodic layouts.** `estimate_psf_sigma` fits a
  Gaussian rolloff to the radial power spectrum and gates on fit quality. On a
  lattice-dominated spectrum an ungated fit returned a confident ~2.4 regardless
  of the true blur, so it refuses and returns the documented default. Correct
  behaviour, but it means the PSF is effectively never measured on our data.
- **Escalation almost never short-circuits.** 320 of 324 pairs reach the
  ambiguous tier. Safe but slow. The thresholds are now *tuned rather than
  untuned*: swept across the full range and retained at 8.0/4.0 because PSR
  separates correct from wrong at AUC 0.581 (correct median 2.33, wrong 2.22),
  so no setting buys latency without paying accuracy. See the sweep above.
- **`n_tied` is always 1 and `tie_break_used` never fires** — 0 of 324 — both
  consequences of `TIE_SIGMA = 0.0`. Two dead confidence features, alongside
  `scale_residual` and `abs_theta`, which are constant while pose is a stub.
  Four of six features carry no variance.
- **The low-confidence flag fires on 99.4% of cases.** Only one answer is
  falsely clear, which is the property that matters, but a flag that is almost
  always on carries little information. It is downstream of the same gap: with
  no anchoredness signal there is nothing to clear it with.
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

- **The package is importable as `src`**, which matches the frozen repository
  layout but is a poor distribution name and collides with anything else called
  `src` on the path. Worth revisiting if the submission is ever installed rather
  than unzipped.

Closed since this list was written:

- `docs/citations.md`, `docs/assumptions.md` and `docs/failure_analysis.md` all
  exist and ship in the archive, so the 30% and the 10% are no longer gated on
  missing documents.
- **The clean-machine test has been run and passes.**
  `scripts/clean_room_check.sh` (R3's) unpacks a `git archive` of a given ref
  into a temporary tree — tracked files only, no `.venv`, no dataset, no trained
  artefacts — resolves the environment from the lockfile and runs the suite,
  lint, types and the CLI there. On the merged integration tree it passes all
  four stages, and the CLI returns the same coordinate from the extracted
  archive as it does in the development tree.

  This is the check that catches "works on my machine", and it has already
  earned its place twice: once on an ignore rule that excluded the results
  sidecars, and once on a provenance test that read `git check-ignore`'s
  "cannot answer" exit code as "not ignored" and so failed only where no `.git`
  exists — which is precisely the grader's situation.
- No trained confidence model ships, by design. `config.CONFIDENCE_MODEL_PATH`
  is looked up and absence falls back to the heuristic, so the deliverable runs
  from a clean unzip with no artefacts.

## Reproducing every number

See `benchmarks/README.md`.
