# Scale robustness: the 9:1 to 11:1 band

Owned by R1.

## Why this set exists

Section 4A of the problem statement says the nominal magnification ratio is
10:1, and that "the sessions indicated robustness testing may include
approximately 9:1 to 11:1".

The shipped dataset does not reach that. `POSE_RANGES` tops out at `large`,
which is a scale mismatch of ±5%, so the 324 measured pairs span **9.50:1 to
10.50:1** — half the range the sponsor named. Every accuracy number in the
README, in `results/` and in the deck was measured inside that narrower band.
Outside it, nothing had ever been run.

This set closes that gap without touching anything already measured.

## The set

48 pairs in `dataset_scale_stress/`, generated from a fresh seed nobody has
tuned against:

```bash
python -m src.generate_dataset --output-dir dataset_scale_stress \
    --pose-conditions stress --seeds-per-cell 4 --seed 20260815
```

| property | value |
|---|---|
| pairs | 48 (24 anchored, 24 unanchored) |
| architectures | 24 DRAM, 24 FinFET |
| noise strata | 16 low, 16 medium, 16 high |
| scale span | **9.01:1 to 10.87:1** |
| rotation span | up to 7.94° |
| base seed | 20260815 |
| image tree sha256 | `d3d09c5cc6e2c0ea61c9a94e3bfef85aadb0f1ff812a1d604794d1db45cb6337` |

17 of the 48 pairs fall outside the ±5% band the shipped set covers.

`stress` is deliberately **not** part of the default stratification. Folding it
into `dataset/` would have invalidated every number the team has measured, three
days from submission, to answer a question that a separate set answers just as
well.

**Reproducibility.** The set was generated twice from different working trees.
Both runs produced the same `image_tree_sha256`, `d3d09c5c…`, so the images are
byte-identical and the manifest is the only thing that changed between them.

## Results

Evaluated with `src.evaluate` in `auto` mode, on the pipeline with R2's pose
estimation merged. Anchored references only — the unanchored stratum sits at
0.000 here exactly as it does on the shipped set, for the same
information-theoretic reason, and is not a scale finding. See
`docs/failure_analysis.md`.

| tolerance | anchored, no pose (n=24) | anchored, with pose (n=24) |
|---|---|---|
| ≤ 5 px | 0.958 | **0.958** |
| ≤ 4 px | 0.875 | **0.958** |
| ≤ 2 px | 0.750 | **0.958** |
| ≤ 1 px | 0.667 | **0.750** |
| median error | 0.13 px | **0.07 px** |

Latency on this set is a median of 727 ms and a p95 of 974 ms, against 388 ms
median on the shipped set. Harder pose drives more cases into the robust and
ambiguous tiers, which is the escalation behaving as designed rather than a
regression.

## What it shows

**The pipeline does not break at 9:1 to 11:1.** It localizes correctly across
the full band. Splitting the anchored cases by how far the scale sits from
nominal:

| anchored cases | ≤ 5 px | ≤ 2 px | ≤ 1 px |
|---|---|---|---|
| within ±5% — the shipped band (n=13) | 0.923 | 0.923 | 0.846 |
| beyond ±5% — never tested before (n=11) | 1.000 | 1.000 | 0.636 |

The wider band is **not** where the pipeline finds the wrong location. At 2 px
it is 11/11. What it loses is sub-pixel precision: 0.636 against 0.846 at 1 px.

That signature has a specific cause, and it is a known one. `src.pose` estimates
rotation but returns `nominal_scale` untouched — scale is not estimated at all.
So on a pair at 10.87:1 the matcher builds its template at 10:1, roughly 9%
wrong in size. The correlation peak still lands on the right structure, because
the lattice is unambiguous once an anchor is present, but the peak is broadened
and the sub-pixel refinement has less to work with.

**Estimating residual scale is the single change that would move the 1 px number
on this band.** It is the same gap that leaves the `scale_residual` confidence
feature dead.

## Honest limits

- **n = 11 beyond ±5% is a thin sample.** The 1.000 at 2 px is 11 of 11, not a
  precision claim. It supports "does not fall apart", not "is perfect".
- One anchored case fails badly: `dram_anchored_pose-stress_0002`, at 948.7 px.
  It is worth being precise about it, because it is not a scale finding. Its
  ground-truth scale is 10.455:1, **inside** the ±5% band the shipped set
  already covers — it is the single miss in the "within ±5%" row above, not in
  the new band. It escalated to `ambiguous` mode and came back with a
  peak-to-sidelobe ratio of 2.44, and the pipeline **flagged it**:
  `low_confidence_flag=1` at a confidence of 0.058. A wrong answer the system
  itself marked as untrustworthy is the confidence model doing its job.
- The unanchored half is 0.000 across the board, unchanged and for unrelated
  reasons.
- These numbers are measured on the scale-stress set only. They do not replace
  and should not be quoted alongside the headline figures without saying which
  set they came from.
