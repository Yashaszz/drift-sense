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

Measured on 108 stratified synthetic pairs at the 1 px (10 nm) tolerance.

| | |
|---|---|
| accuracy, **anchored** references | **77.8%** |
| accuracy, unanchored references | 0.0% |
| accuracy, all pairs | 38.9% |
| latency, median / p95 | 105 ms / 146 ms |
| sub-pixel accuracy on known shifts | 0.01 px |

The unanchored result is the honest one and not a defect. An unanchored
reference is a periodic patch with no aperiodic feature in frame; on a bare
lattice **903 distinct positions score exactly the maximum**, so the evidence
does not identify a location. That is an information-theoretic limit, and the
correct response is to answer and flag rather than to claim success.

These numbers are measured on geometry-only data — the SEM physics chain is not
yet implemented — and will move once noise, blur and edge brightening land.

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
tests/        474 tests
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
