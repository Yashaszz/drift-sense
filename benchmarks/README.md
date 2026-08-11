# Benchmarks

Every number quoted in `docs/r4_engineering_notes.md` and `docs/r4_handoff.md`
is produced by one of these scripts. They take a dataset directory and print to
stdout; pass `--json` where offered to capture machine-readable output.

## Prerequisites

```bash
uv sync --all-extras
```

Then generate a dataset. This is the only slow step — roughly four minutes for
108 pairs, because each pair is supersampled and rendered twice.

```bash
uv run python -m src.generate_dataset --output-dir data --seeds-per-cell 9
```

`--seeds-per-cell 1` gives 12 pairs in about 40 seconds and is enough to smoke
test the scripts, though not to draw conclusions from.

## Scripts

### `bench_zncc.py` — Stage 3, 3b and 5 in isolation

```bash
uv run python -m benchmarks.bench_zncc --repeats 25
```

Self-contained; generates its own synthetic fields and needs no dataset. Reports
correlation timing across template sizes, peak-extraction timing across
suppression radii, and sub-pixel accuracy against exact Fourier-domain shifts
across a noise sweep.

Produces: the 34.6 ms unweighted Stage 3 figure, the Stage 3b radius table, and
the sub-pixel accuracy numbers.

### `bench_localize.py` — end-to-end profile

```bash
uv run python -m benchmarks.bench_localize --data data
uv run python -m benchmarks.bench_localize --data data --json results/timing.json
```

Reports per-stage cost across the full escalation ladder *without* tier reuse —
which is what makes the redundancy visible — then end-to-end `localize()`
timings as shipped, then weighted versus unweighted correlation.

Produces: the T9 profiling table, the 105 ms auto-mode median, and the 2.01x
weighted correlation cost ratio.

### `fit_confidence.py` — Stage 6 calibration

```bash
uv run python -m benchmarks.fit_confidence --data data
uv run python -m benchmarks.fit_confidence --data data --out models/confidence.json
```

Runs the pipeline over every labelled pair, builds the feature matrix, fits the
calibrator and reports per-feature discrimination, cross-validated AUC,
standardised coefficients, and the full threshold trade-off between false
confidence and wasted escalation.

Produces: the 0.504 cross-validated AUC, the dead-feature table, and the
accuracy-by-anchor split.

Writing a model with `--out` is optional and currently **not recommended** — the
model is at chance, and shipping it would replace an honest heuristic with a
calibrator that looks authoritative. Revisit once R2 and R3 land.

### `verify_uniqueness_integration.py` — R3 readiness

```bash
uv run python -m benchmarks.verify_uniqueness_integration --data data
```

Substitutes a stand-in uniqueness map for R3's stub and re-runs the whole
pipeline, reporting accuracy and confidence AUC with and without it. The
stand-in is a test double, not an implementation of R3's stage.

Produces: the 38.9% -> 42.6% accuracy and 0.504 -> 0.828 AUC comparison, and the
evidence that R4 needs no change when the real map lands.

## Determinism

Every script is seeded and every pipeline stage is deterministic, so repeated
runs on the same dataset give identical *results*. Timings vary with machine
load — the T9 before/after comparison was measured back to back in one session
for that reason, and a quiet machine is worth using for any figure that goes in
the deck.

## Reproducing the headline table

```bash
uv sync --all-extras
uv run python -m src.generate_dataset --output-dir data --seeds-per-cell 9
uv run python -m benchmarks.bench_localize --data data
uv run python -m benchmarks.fit_confidence --data data
uv run python -m benchmarks.verify_uniqueness_integration --data data
```
