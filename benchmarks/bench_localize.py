"""Stage-by-stage profile of ``localize()`` on the generated dataset.

Runtime is scored inside the 50%, so the cost of every stage needs to be a
measured number rather than an estimate. This reports where the time goes, per
stage and as a percentage, on whichever dataset ``generate_dataset.py`` has
produced.

Why per stage rather than a single total: the total alone cannot tell you
whether a change helped for the reason you thought. The T9 optimisation was
found by noticing that three stages each ran three times per call on identical
inputs, which is invisible in an end-to-end timing.

Usage
-----
    python -m benchmarks.bench_localize --data data
    python -m benchmarks.bench_localize --data data --json results/localize_timing.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, disambiguate, matcher, pose  # noqa: E402
from src.localize import localize  # noqa: E402
from src.types import PoseEstimate  # noqa: E402

_T = TypeVar("_T")

REPEATS = 3


class _Timer:
    """Accumulates wall-clock time and call counts per stage name."""

    def __init__(self) -> None:
        self.total: dict[str, float] = defaultdict(float)
        self.calls: dict[str, int] = defaultdict(int)

    def run(self, name: str, fn: Callable[[], _T]) -> _T:
        """Time one call and record it under ``name``."""
        started = time.perf_counter()
        result = fn()
        self.total[name] += (time.perf_counter() - started) * 1000.0
        self.calls[name] += 1
        return result


def load_pairs(root: Path) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Read every pair listed in ``ground_truth.jsonl``.

    Parameters
    ----------
    root
        Dataset directory containing ``ground_truth.jsonl``.

    Returns
    -------
    list
        ``(pair_id, reference, search)`` per record.
    """
    records = [json.loads(line) for line in (root / "ground_truth.jsonl").read_text().splitlines()]
    return [
        (
            record["pair_id"],
            np.asarray(Image.open(root / record["reference_path"]).convert("L")),
            np.asarray(Image.open(root / record["search_path"]).convert("L")),
        )
        for record in records
    ]


def _profile_tier(
    timer: _Timer,
    reference: np.ndarray,
    search: np.ndarray,
    tier: str,
) -> None:
    """Time every stage of one escalation tier.

    Taking the arrays as parameters rather than closing over loop variables
    keeps each timed call bound to the values it was created with.

    Parameters
    ----------
    timer
        Accumulator to record into.
    reference
        Reference image as ``float32``.
    search
        Search image as ``float32``.
    tier
        Which tier to emulate.
    """
    if tier == "fast":
        estimate = PoseEstimate.nominal()
        psf = config.DEFAULT_PSF_SIGMA_PX
    else:
        estimate = timer.run(
            "pose.estimate_pose",
            lambda: pose.estimate_pose(reference, search, config.NOMINAL_SCALE),
        )
        psf = timer.run(
            "matcher.estimate_psf_sigma",
            lambda: matcher.estimate_psf_sigma(search),
        )

    template = timer.run(
        "matcher.build_template",
        lambda: matcher.build_template(reference, estimate.theta_deg, estimate.scale, psf),
    )
    surface = timer.run(
        "matcher.zncc_surface",
        lambda: matcher.zncc_surface(template, search),
    )
    peaks = timer.run(
        "matcher.top_k_peaks",
        lambda: matcher.top_k_peaks(surface, config.DEFAULT_TOP_K, config.DEFAULT_NMS_RADIUS_PX),
    )
    if not peaks:
        return

    _, std = timer.run(
        "disambiguate.sidelobe_stats",
        lambda: disambiguate.sidelobe_stats(surface, peaks[0], config.DEFAULT_NMS_RADIUS_PX),
    )
    tolerance = config.TIE_SIGMA * std
    timer.run(
        "disambiguate.tied_candidates",
        lambda: disambiguate.tied_candidates(peaks, tolerance),
    )
    best, _ = timer.run(
        "disambiguate.select_candidate",
        lambda: disambiguate.select_candidate(
            peaks, config.image_centre(search.shape), template.shape, tolerance=tolerance
        ),
    )
    timer.run(
        "disambiguate.peak_to_sidelobe",
        lambda: disambiguate.peak_to_sidelobe(surface, best, config.DEFAULT_NMS_RADIUS_PX),
    )
    timer.run(
        "matcher.refine_subpixel_detailed",
        lambda: matcher.refine_subpixel_detailed(
            template, search, best, surface=surface, upsample=config.DEFAULT_UPSAMPLE
        ),
    )


def profile_stages(pairs: list[tuple[str, np.ndarray, np.ndarray]]) -> _Timer:
    """Time each stage across the full ladder, without tier reuse.

    Mirrors what ``localize`` does in ``auto`` mode when no tier can reuse an
    earlier one. This is the *unoptimised* shape of the work, which is what
    makes the redundancy visible.

    Parameters
    ----------
    pairs
        Image pairs to profile.

    Returns
    -------
    _Timer
        Accumulated per-stage timings.
    """
    timer = _Timer()
    for _, reference, search in pairs:
        reference_f = np.ascontiguousarray(reference, dtype=np.float32)
        search_f = np.ascontiguousarray(search, dtype=np.float32)
        for tier in ("fast", "robust", "ambiguous"):
            _profile_tier(timer, reference_f, search_f, tier)
    return timer


def time_localize(
    pairs: list[tuple[str, np.ndarray, np.ndarray]], mode: str
) -> tuple[float, float]:
    """Time the real ``localize`` entry point.

    Parameters
    ----------
    pairs
        Image pairs to time.
    mode
        Operating mode to request.

    Returns
    -------
    tuple of float
        ``(median_ms, p95_ms)`` per pair.
    """
    samples: list[float] = []
    for _, reference, search in pairs:
        localize(search, reference, mode=mode)  # warm
        per_pair = []
        for _ in range(REPEATS):
            started = time.perf_counter()
            localize(search, reference, mode=mode)
            per_pair.append((time.perf_counter() - started) * 1000.0)
        samples.append(float(np.median(per_pair)))
    return float(np.median(samples)), float(np.percentile(samples, 95))


def bench_weighting(pairs: list[tuple[str, np.ndarray, np.ndarray]]) -> dict[str, float]:
    """Compare weighted against unweighted correlation on the dataset.

    Weighted correlation costs three cross-correlations where the unweighted
    path costs one, so the accuracy it buys has to be worth roughly triple the
    Stage 3 time. This measures both sides of that trade.

    Parameters
    ----------
    pairs
        Image pairs to evaluate.

    Returns
    -------
    dict
        Median milliseconds and mean peak height for each path, plus the
        largest surface difference seen.
    """
    plain_ms: list[float] = []
    weighted_ms: list[float] = []
    plain_peak: list[float] = []
    weighted_peak: list[float] = []
    max_difference = 0.0

    for _, reference, search in pairs:
        reference_f = np.ascontiguousarray(reference, dtype=np.float32)
        search_f = np.ascontiguousarray(search, dtype=np.float32)
        template = matcher.build_template(
            reference_f, 0.0, config.NOMINAL_SCALE, config.DEFAULT_PSF_SIGMA_PX
        )
        weights = matcher.build_weight(
            disambiguate.uniqueness_map(reference_f), 0.0, config.NOMINAL_SCALE
        )

        started = time.perf_counter()
        plain = matcher.zncc_surface(template, search_f)
        plain_ms.append((time.perf_counter() - started) * 1000.0)

        started = time.perf_counter()
        weighted = matcher.zncc_surface(template, search_f, weight=weights)
        weighted_ms.append((time.perf_counter() - started) * 1000.0)

        plain_peak.append(float(plain.max()))
        weighted_peak.append(float(weighted.max()))
        max_difference = max(max_difference, float(np.abs(weighted - plain).max()))

    return {
        "plain_median_ms": float(np.median(plain_ms)),
        "weighted_median_ms": float(np.median(weighted_ms)),
        "plain_mean_peak": float(np.mean(plain_peak)),
        "weighted_mean_peak": float(np.mean(weighted_peak)),
        "max_surface_difference": max_difference,
    }


def main(argv: list[str] | None = None) -> int:
    """Print the profile and the end-to-end timings.

    Parameters
    ----------
    argv
        Argument list; defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        ``0`` on success, ``1`` when the dataset is missing.
    """
    parser = argparse.ArgumentParser(description="Profile localize() per stage.")
    parser.add_argument("--data", type=Path, default=Path("data"), help="dataset root")
    parser.add_argument("--json", type=Path, default=None, help="also write results as JSON")
    args = parser.parse_args(argv)

    if not (args.data / "ground_truth.jsonl").is_file():
        parser.exit(
            status=1,
            message=(
                f"error: no ground_truth.jsonl under {args.data}. "
                "Generate one with: python -m src.generate_dataset\n"
            ),
        )

    pairs = load_pairs(args.data)
    print(f"{platform.platform()} | Python {platform.python_version()} | {len(pairs)} pairs")
    print()

    timer = profile_stages(pairs)
    grand_total = sum(timer.total.values())
    n = len(pairs)

    print("Per-stage cost of the full ladder, without tier reuse")
    print(f"{'stage':<34} {'calls/pair':>11} {'ms/pair':>9} {'% total':>8}")
    print("-" * 66)
    for name in sorted(timer.total, key=lambda k: -timer.total[k]):
        print(
            f"{name:<34} {timer.calls[name] // n:>11} "
            f"{timer.total[name] / n:>9.2f} {100 * timer.total[name] / grand_total:>7.1f}%"
        )
    print("-" * 66)
    print(f"{'TOTAL':<34} {'':>11} {grand_total / n:>9.2f} {100.0:>7.1f}%")

    print()
    print("End-to-end localize(), as shipped (tier reuse enabled)")
    print(f"{'mode':<12} {'median':>10} {'p95':>10}")
    print("-" * 34)
    timings = {}
    for mode in ("fast", "robust", "auto"):
        median, p95 = time_localize(pairs, mode)
        timings[mode] = {"median_ms": median, "p95_ms": p95}
        print(f"{mode:<12} {median:>9.1f}m {p95:>9.1f}m")

    print()
    print("Stage 3 - weighted vs unweighted correlation")
    weighting = bench_weighting(pairs)
    print(f"{'path':<14} {'median':>10} {'mean peak':>11}")
    print("-" * 37)
    print(
        f"{'unweighted':<14} {weighting['plain_median_ms']:>9.1f}m "
        f"{weighting['plain_mean_peak']:>11.4f}"
    )
    print(
        f"{'weighted':<14} {weighting['weighted_median_ms']:>9.1f}m "
        f"{weighting['weighted_mean_peak']:>11.4f}"
    )
    print(
        f"cost ratio {weighting['weighted_median_ms'] / weighting['plain_median_ms']:.2f}x   "
        f"largest surface difference {weighting['max_surface_difference']:.2e}"
    )

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "host": platform.platform(),
            "pairs": len(pairs),
            "stages": {
                name: {
                    "calls_per_pair": timer.calls[name] // n,
                    "ms_per_pair": timer.total[name] / n,
                    "percent": 100 * timer.total[name] / grand_total,
                }
                for name in timer.total
            },
            "localize": timings,
            "weighting": weighting,
        }
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
