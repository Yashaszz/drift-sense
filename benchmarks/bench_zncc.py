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

from src import config, matcher  # noqa: E402
from src.types import Peak  # noqa: E402

SEARCH_EDGE = 1000
TEMPLATE_EDGES = (50, 100, 150, 200)
NMS_RADII = (2, 4, 8, 16)
NOISE_LEVELS = (0.0, 0.02, 0.05, 0.1, 0.3)
SUBPIXEL_TRIALS = 25
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


def time_peaks(template_edge: int, search_edge: int, repeats: int, nms_radius: int) -> TimingResult:
    """Time :func:`src.matcher.top_k_peaks` on a realistic correlation surface.

    Uses a surface produced by the real Stage 3, not synthetic noise: peak
    extraction cost depends on how many local maxima exist, and a periodic
    surface has far more of them than random data.

    Parameters
    ----------
    template_edge
        Template edge length in pixels.
    search_edge
        Search-image edge length in pixels.
    repeats
        Number of timed iterations.
    nms_radius
        Suppression radius in surface pixels.

    Returns
    -------
    TimingResult
        Timing statistics in milliseconds.
    """
    search = make_field((search_edge, search_edge))
    template = np.ascontiguousarray(search[100 : 100 + template_edge, 200 : 200 + template_edge])
    surface = matcher.zncc_surface(template, search)

    for _ in range(WARMUP):
        matcher.top_k_peaks(surface, k=config.DEFAULT_TOP_K, nms_radius=nms_radius)

    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        matcher.top_k_peaks(surface, k=config.DEFAULT_TOP_K, nms_radius=nms_radius)
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


def fourier_shift(image: np.ndarray, dy: float, dx: float) -> np.ndarray:
    """Translate an image by an exact fractional offset via its spectrum.

    Parameters
    ----------
    image
        Source image.
    dy
        Row offset in pixels.
    dx
        Column offset in pixels.

    Returns
    -------
    numpy.ndarray
        Shifted image, ``float32``.
    """
    spectrum = np.fft.fft2(image)
    freq_row = np.fft.fftfreq(image.shape[0])[:, None]
    freq_col = np.fft.fftfreq(image.shape[1])[None, :]
    ramp = np.exp(-2j * np.pi * (freq_row * dy + freq_col * dx))
    return np.real(np.fft.ifft2(spectrum * ramp)).astype(np.float32)


def bench_subpixel(trials: int, noise_levels: tuple[float, ...]) -> list[dict[str, float]]:
    """Measure Stage 5 accuracy and runtime against exact fractional shifts.

    Ground truth comes from a Fourier-domain translation, which is an exact
    fractional shift rather than an interpolated approximation. Comparing
    against an interpolated target would measure the interpolator instead of the
    estimator.

    Parameters
    ----------
    trials
        Random offsets evaluated per noise level.
    noise_levels
        Noise standard deviations to sweep.

    Returns
    -------
    list of dict
        One record per noise level with median and p95 error, in pixels, for
        both refinement routines, plus the primary routine's runtime.
    """
    field = make_field((256, 256), seed=3)
    template = np.ascontiguousarray(field[60:124, 90:154])
    peak = Peak(col=90, row=60, score=1.0)

    records = []
    for noise in noise_levels:
        rng = np.random.default_rng(0)
        primary, fallback, timings, integer = [], [], [], []

        for _ in range(trials):
            offset_y, offset_x = rng.uniform(-0.5, 0.5, 2)
            search = fourier_shift(field, offset_y, offset_x)
            if noise > 0.0:
                search = (search + rng.normal(0.0, noise, search.shape)).astype(np.float32)
            surface = matcher.zncc_surface(template, search)

            started = time.perf_counter()
            refined = matcher.refine_subpixel_detailed(template, search, peak, surface=surface)
            timings.append((time.perf_counter() - started) * 1000.0)

            coarse_x, coarse_y = matcher.refine_subpixel(surface, peak)
            integer.append(np.hypot(offset_x, offset_y))
            primary.append(np.hypot(offset_x - refined.dx, offset_y - refined.dy))
            fallback.append(np.hypot(offset_x - coarse_x, offset_y - coarse_y))

        records.append(
            {
                "noise": noise,
                "integer_median": float(np.median(integer)),
                "primary_median": float(np.median(primary)),
                "primary_p95": float(np.percentile(primary, 95)),
                "fallback_median": float(np.median(fallback)),
                "median_ms": float(np.median(timings)),
            }
        )
    return records


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

    print()
    print(f"Stage 3b - top_k_peaks (k={config.DEFAULT_TOP_K}), on a real Stage 3 surface")
    print(f"{'radius':>10} {'surface':>12} {'median':>10} {'p95':>10} {'mean':>10}")
    print("-" * len(header))

    peak_results = []
    for radius in NMS_RADII:
        result = time_peaks(100, SEARCH_EDGE, args.repeats, radius)
        peak_results.append((radius, result))
        print(
            f"{radius:>10} "
            f"{result.surface_pixels:>12,} "
            f"{result.median_ms:>9.2f}m "
            f"{result.p95_ms:>9.2f}m "
            f"{result.mean_ms:>9.2f}m"
        )

    print()
    print("Stage 5 - sub-pixel refinement, against exact Fourier-domain shifts")
    print(
        f"{'noise':>10} {'no refine':>11} {'primary':>10} {'p95':>9} "
        f"{'fallback':>10} {'runtime':>10}"
    )
    print("-" * 64)

    subpixel = bench_subpixel(SUBPIXEL_TRIALS, NOISE_LEVELS)
    for record in subpixel:
        print(
            f"{record['noise']:>10.2f} "
            f"{record['integer_median']:>10.3f}p "
            f"{record['primary_median']:>9.3f}p "
            f"{record['primary_p95']:>8.3f}p "
            f"{record['fallback_median']:>9.3f}p "
            f"{record['median_ms']:>9.2f}m"
        )

    headline = next(r for r in results if r.template_edge == 100)
    headline_peaks = next(r for radius, r in peak_results if radius == 8)
    combined = headline.median_ms + headline_peaks.median_ms
    print()
    print(
        f"Headline: {headline.median_ms:.2f} ms median "
        f"({headline.p95_ms:.2f} ms p95) for Stage 3 with a 100x100 template "
        f"on a {SEARCH_EDGE}x{SEARCH_EDGE} search image."
    )
    print(
        f"          {headline_peaks.median_ms:.2f} ms median for Stage 3b at radius 8, "
        f"so {combined:.2f} ms for the pair."
    )

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "host": host,
            "stage3": [asdict(r) for r in results],
            "stage3b": [{"nms_radius": radius, **asdict(r)} for radius, r in peak_results],
        }
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
