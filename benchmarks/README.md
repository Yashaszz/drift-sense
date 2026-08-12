# Benchmarks

Every number quoted in `docs/r4_engineering_notes.md` and `docs/r4_handoff.md`
is produced by one of these scripts. They take a dataset directory and print to
stdout; pass `--json` where offered to capture machine-readable output.

## Prerequisites

```bash
uv sync --all-extras
```

Then generate a dataset. This is the only slow step — roughly twenty minutes for
the full 324 pairs, because each pair is supersampled, rendered twice and put
through the SEM physics chain.

```bash
uv run python -m src.generate_dataset --output-dir data --seeds-per-cell 9
```

Since the noise strata landed, `--seeds-per-cell` produces
`2 x 2 x 3 x 3 x seeds` pairs: **9 gives 324**, 3 gives 108, and 1 gives 36 in
about two minutes, which is enough to smoke test the scripts though not to draw
conclusions from.

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

Produces: the T9 profiling table and the weighted-versus-unweighted cost ratio.
A per-stage breakdown is also available from any `localize()` call by setting
`config.COLLECT_STAGE_TIMINGS`, which is how the 324-pair latency table in
`docs/r4_handoff.md` was produced.

### `fit_confidence.py` — Stage 6 calibration

```bash
uv run python -m benchmarks.fit_confidence --data data
uv run python -m benchmarks.fit_confidence --data data --out models/confidence.json
```

Runs the pipeline over every labelled pair, builds the feature matrix, fits the
calibrator and reports per-feature discrimination, cross-validated AUC,
standardised coefficients, and the full threshold trade-off between false
confidence and wasted escalation.

Produces: the cross-validated AUC, the dead-feature table, and the
accuracy-by-anchor split. On the 324-pair physics set: **CV AUC 0.570**, with
four of six features carrying no variance.

Writing a model with `--out` is optional and still **not recommended** — at CV
AUC 0.570 the model is near chance, and shipping it would replace an honest
heuristic with a calibrator that looks authoritative. Revisit when some
diagnostic separates anchored from unanchored references; that signal is worth
AUC 0.935 on its own.

### `verify_uniqueness_integration.py` — R3 readiness

```bash
uv run python -m benchmarks.verify_uniqueness_integration --data data
```

Substitutes a stand-in uniqueness map and re-runs the whole pipeline, reporting
accuracy and confidence AUC with and without it. The stand-in is a test double,
not an implementation of R3's stage.

Superseded as a readiness check now that R3's map has landed — it was written to
prove R4 would need no change when it did, and that held. It remains useful as
an A/B harness for a *replacement* scorer: `localize`'s uniqueness cache keys on
the implementation, so a substituted map gets its own cache entry rather than
being served the real one's.

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
