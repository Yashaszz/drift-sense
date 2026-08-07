"""Benchmark the Stage 3 ZNCC correlation surface.

Runtime is scored inside the 50%, and the deck explicitly requires computation
time on a single 1000x1000 pair. This script produces that number on named
hardware, so the figure we present is measured rather than estimated.

Reported statistics
-------------------
Median and p95 rather than mean alone. A wafer tool cares about the tail: a
matcher whose median is 20 ms but whose p95 is 300 ms will miss its throughput
budget even though the average looks fine.

Usage
-----
    python -m benchmarks.bench_zncc
    python -m benchmarks.bench_zncc --repeats 50 --json results/zncc_timing.json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import matcher  # noqa: E402

SEARCH_EDGE = 1000
TEMPLATE_EDGES = (50, 100, 150, 200)
DEFAULT_REPEATS = 30
WARMUP = 3


@dataclass(frozen=True)
class TimingResult:
    """Timing statistics for one template size."""

    template_edge: int
    search_edge: int
    surface_pixels: int
    repeats: int
    median_ms: float
    p95_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float


def make_field(shape: tuple[int, int], seed: int = 1234) -> np.ndarray:
    """Build a periodic field with aperiodic texture, at search resolution.

    Mirrors the test fixture but stays self-contained: a benchmark that imports
    from the test suite would break the moment the fixtures are refactored.

    Parameters
    ----------
    shape
        Image shape as ``(rows, cols)``.
    seed
        Seed for the texture draw.

    Returns
    -------
    numpy.ndarray
        ``float32`` image.
    """
    rows, cols = shape
    grid_row, grid_col = np.mgrid[0:rows, 0:cols]
    field = np.sin(2 * np.pi * grid_col / 16.0) * np.sin(2 * np.pi * grid_row / 16.0)
    texture = np.random.default_rng(seed).normal(0.0, 0.4, size=shape)
    return (field + texture).astype(np.float32)


def time_one(template_edge: int, search_edge: int, repeats: int) -> TimingResult:
    """Time :func:`src.matcher.zncc_surface` for one template size.

    Parameters
    ----------
    template_edge
        Template edge length in pixels.
    search_edge
        Search-image edge length in pixels.
    repeats
        Number of timed iterations.

    Returns
    -------
    TimingResult
        Timing statistics in milliseconds.
    """
    search = make_field((search_edge, search_edge))
    template = np.ascontiguousarray(search[100 : 100 + template_edge, 200 : 200 + template_edge])

    for _ in range(WARMUP):
        matcher.zncc_surface(template, search)

    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        surface = matcher.zncc_surface(template, search)
        samples.append((time.perf_counter() - started) * 1000.0)

    samples.sort()
    return TimingResult(
        template_edge=template_edge,
        search_edge=search_edge,
        surface_pixels=int(surface.size),
        repeats=repeats,
        median_ms=statistics.median(samples),
        p95_ms=samples[min(len(samples) - 1, int(0.95 * len(samples)))],
        mean_ms=statistics.fmean(samples),
        min_ms=samples[0],
        max_ms=samples[-1],
    )


def describe_host() -> dict[str, str]:
    """Return a description of the machine, for the results table.

    Returns
    -------
    dict
        Platform, processor, Python version and OpenCV version.
    """
    import cv2

    return {
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark and print a results table.

    Parameters
    ----------
    argv
        Argument list; defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit status.
    """
    parser = argparse.ArgumentParser(description="Benchmark Stage 3 ZNCC.")
    parser.add_argument(
        "--repeats", type=int, default=DEFAULT_REPEATS, help="timed iterations per size"
    )
    parser.add_argument("--json", type=Path, default=None, help="also write results as JSON")
    args = parser.parse_args(argv)

    host = describe_host()
    print("Stage 3 ZNCC - cv2.matchTemplate(TM_CCOEFF_NORMED)")
    for key, value in host.items():
        print(f"  {key:10s} {value}")
    print()

    header = f"{'template':>10} {'surface':>12} {'median':>10} {'p95':>10} {'mean':>10}"
    print(header)
    print("-" * len(header))

    results = []
    for edge in TEMPLATE_EDGES:
        result = time_one(edge, SEARCH_EDGE, args.repeats)
        results.append(result)
        print(
            f"{result.template_edge:>7}^2 "
            f"{result.surface_pixels:>12,} "
            f"{result.median_ms:>9.2f}m "
            f"{result.p95_ms:>9.2f}m "
            f"{result.mean_ms:>9.2f}m"
        )

    headline = next(r for r in results if r.template_edge == 100)
    print()
    print(
        f"Headline: {headline.median_ms:.2f} ms median "
        f"({headline.p95_ms:.2f} ms p95) for a 100x100 template "
        f"on a {SEARCH_EDGE}x{SEARCH_EDGE} search image."
    )

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {"host": host, "results": [asdict(r) for r in results]}
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
