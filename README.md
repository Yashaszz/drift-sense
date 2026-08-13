# Drift-Sense

Navigation-error recovery for wafer inspection. Applied Materials Hackathon,
problem statement PS-02.

A wafer-inspection tool must photograph the same microscopic spot repeatedly,
but its motion stage accumulates error and lands roughly a micrometre off
target — and because every die is a printed copy of the same design, the wrong
location looks identical to the right one. Given the zoomed-in `reference` image
from the first visit and a zoomed-out `search` image of wherever the stage
actually landed, this locates the reference site inside the search image and
returns its centre in pixels. That offset, times the pixel size, is the stage
correction vector.

## Inputs and output

```
reference : uint8 (1000, 1000)   1 nm/px   ->  1 um field of view
search    : uint8 (1000, 1000)  10 nm/px   -> 10 um field of view

(x, y)    : float, float — centre of the match, in SEARCH-image pixels
```

The reference occupies roughly 100x100 pixels inside the search image. The two
are separate physical captures: independent noise, different point-spread
function, possibly different detector gain, and small rotation and scale error.

## Quick start

```bash
uv sync --all-extras
uv run python -m src.localize search.png reference.png --json
```

As a library:

```python
from src.localize import localize

result = localize(search, reference, mode="auto")
print(result.x, result.y, result.confidence, result.low_confidence_flag)
```

`localize()` never raises on valid input. On failure it degrades to the
search-image centre with a zero confidence and a flag, because an alignment step
on a real tool needs an answer plus an honest quality signal, not an exception.

## Results

Measured on 324 stratified pairs carrying the full SEM physics chain — edge
brightening, PSF blur, Poisson shot noise and read noise, applied independently
per capture — at the 1 px (10 nm) tolerance. All 324 scored; none failed.

| | |
|---|---|
| accuracy, **anchored** references | **85.2%** (138/162) |
| accuracy, unanchored references | 0.0% |
| accuracy, all pairs | 42.6% |
| median error, anchored | **0.042 px** |
| latency, median / p95 | 206 ms / 225 ms |

Accuracy is hardware-independent and reproduces to the digit on both of our
machines. **Latency is not, so it is quoted for one named machine:** the figures
above are from `results/full_324.csv` on an Apple Silicon Mac. The same run on
Windows 11 / AMD Zen 3 gives 356 ms / 363 ms — 1.7x slower, with the gap
concentrated in FFT-heavy work rather than spread evenly. Any latency figure
that reaches a slide should carry the machine with it.

The unanchored result is the honest one and not a defect. An unanchored
reference is a periodic patch with no aperiodic feature in frame, so the
correlation evidence does not identify a location: on a bare lattice hundreds of
distinct positions score exactly the maximum. That is an information-theoretic
limit rather than an algorithmic one, and the correct response is to answer and
flag rather than to claim success. Escalating those cases buys **+0.0** accuracy;
on anchored references the same escalation buys +7.4 points.

Two results worth stating plainly:

- **The physics chain did not cost accuracy.** Anchored accuracy is unchanged
  from the geometry-only dataset. Across noise strata it is 83.3% / 83.3% /
  88.9% for low / medium / high — ZNCC is normalised, so the current noise
  magnitudes do not move it.
- **Pose is the dominant degradation axis.** 48.1% at `pose=none` against 33.3%
  at `pose=large`. Rotation and scale estimation is not yet implemented, so
  rotated pairs are matched at nominal pose.

Latency is dominated by one stage: scoring the reference's uniqueness map is
**60.6% of the call**. That share is the portable number — it holds on either
machine, where the absolute milliseconds do not. The map depends only on the
reference, so a repeat visit to the same site is served from cache; the figures
above are the cold cost, which is what a sweep over 324 distinct references
measures.

## How it works

```
reference ──┐                              ┌── search
            └── Stage 0  normalise ────────┘
                   Stage 1  pose            rotation and scale, Fourier/log-polar
                   Stage 2  template        rotate, PSF-match, area-average
                   Stage 3  ZNCC surface    FFT matched filter
                   Stage 3b peaks + NMS     one candidate per lattice cell
                   Stage 4  disambiguate    uniqueness weighting, PSR, tie-break
                   Stage 5  subpixel        upsampled-DFT registration
                   Stage 6  confidence      calibrated scalar + escalation flag
```

Compute escalates rather than being spent up front: `fast` assumes nominal pose,
and `robust` and `ambiguous` are entered only when the detection statistic says
the cheap path was not enough.

See `docs/r4_engineering_notes.md` for the design decisions and the measurements
behind them, and `benchmarks/README.md` to reproduce every number.

## Repository

```
src/          pipeline modules; localize.py is the deliverable
tests/        498 tests
benchmarks/   reproducible timing, accuracy and calibration reports
docs/         engineering notes and handoff
```

## Development

```bash
uv run ruff check src tests benchmarks
uv run ruff format --check src tests benchmarks
uv run mypy
uv run pytest
```

CI runs all four on every push. `requirements.txt` pins the exact environment;
`uv.lock` and `.python-version` are committed so every machine resolves
identically.
