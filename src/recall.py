"""Measure recall@K: was truth in the candidate list before disambiguation.

R4's number, R3's harness. Measured against the shortlist ``top_k_peaks``
produces, so it isolates the matcher from ``select_candidate``. Reuses
``_StageCache`` so the template, pose and PSF match what ``localize`` builds;
re-deriving them here would make recall@K and top-1 describe different
pipelines.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np

from src import config, matcher
from src.evaluate import load_cases, load_image
from src.localize import _resolve_pose, _StageCache
from src.types import FloatArray

DEFAULT_K_VALUES = (1, 5, 10, 30)
DEFAULT_RADII = (2, 4, 8, 16)


def _pose_for(search: FloatArray, reference: FloatArray) -> tuple[float, float]:
    """Rotation and scale for the fast tier, matching the real pipeline."""
    pose = _resolve_pose(search, reference, "fast")
    return float(pose.theta_deg), float(pose.scale)


def _rank_of_hit(
    peaks: list[Any],
    template_shape: tuple[int, int],
    gt_x: float,
    gt_y: float,
    tol: float,
) -> int:
    """1-based rank of the first peak within ``tol`` of truth, else -1."""
    for rank, peak in enumerate(peaks, start=1):
        x, y = peak.centre(template_shape)
        if math.hypot(float(x) - gt_x, float(y) - gt_y) <= tol:
            return rank
    return -1


def measure(
    cases: list[dict[str, Any]],
    data_dir: Path,
    radii: tuple[int, ...],
    max_k: int,
    tol: float,
) -> list[dict[str, Any]]:
    """One row per (case, nms_radius) with the rank at which truth appears."""
    rows: list[dict[str, Any]] = []
    for i, case in enumerate(cases, start=1):
        reference = np.asarray(load_image(data_dir / case["ref_path"]), dtype=np.float32)
        search = np.asarray(load_image(data_dir / case["search_path"]), dtype=np.float32)
        cache = _StageCache(search, reference)
        theta, scale = _pose_for(search, reference)
        psf = cache.psf_sigma("fast")
        for weighted in (False, True):
            template, surface, _ = cache.correlate(
                theta=theta, scale=scale, psf_sigma=psf, weighted=weighted
            )
            for radius in radii:
                peaks = matcher.top_k_peaks(surface, k=max_k, nms_radius=radius)
                rank = _rank_of_hit(
                    peaks,
                    template.shape,
                    float(case["gt_x"]),
                    float(case["gt_y"]),
                    tol,
                )
                rows.append(
                    {
                        "case_id": case["case_id"],
                        "arch": case["arch"],
                        "anchored": case["anchored"],
                        "pose": case["pose"],
                        "noise": case["noise"],
                        "weighted": weighted,
                        "nms_radius": radius,
                        "n_peaks": len(peaks),
                        "rank": rank,
                    }
                )
        print(f"  [{i:>3}/{len(cases)}] {case['case_id']:<30} done")
    return rows


def _recall(rows: list[dict[str, Any]], k: int) -> float:
    if not rows:
        return float("nan")
    hits = sum(1 for r in rows if 0 < r["rank"] <= k)
    return hits / len(rows)


def report(rows: list[dict[str, Any]], radii: tuple[int, ...], k_values: tuple[int, ...]) -> None:
    """Print recall@K per NMS radius, overall and by anchor stratum."""
    header = "wt     radius  stratum       n   " + "  ".join(f"r@{k:<4}" for k in k_values)
    print("\n" + header)
    print("-" * len(header))
    for weighted in (False, True):
        for radius in radii:
            at_r = [r for r in rows if r["nms_radius"] == radius and r["weighted"] == weighted]
            strata = [
                ("overall", at_r),
                ("anchored", [r for r in at_r if r["anchored"] in (True, "anchored")]),
                (
                    "unanchored",
                    [r for r in at_r if r["anchored"] not in (True, "anchored")],
                ),
            ]
            for name, subset in strata:
                cells = "  ".join(f"{_recall(subset, k):<6.3f}" for k in k_values)
                wt = "on " if weighted else "off"
                print(f"{wt:<6} {radius:<7} {name:<12} {len(subset):<3} {cells}")
            print()


def main() -> None:
    """Run the recall@K sweep from the command line."""
    parser = argparse.ArgumentParser(description="recall@K harness")
    parser.add_argument("--data", type=Path, default=Path("dataset_full"))
    parser.add_argument("--gt", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("results/recall.csv"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tol", type=float, default=1.0)
    parser.add_argument("--max-k", type=int, default=config.DEFAULT_TOP_K)
    args = parser.parse_args()

    gt_path = args.gt or (args.data / "ground_truth.jsonl")
    cases = load_cases(gt_path)
    if args.limit:
        cases = cases[: args.limit]
    print(f"Loaded {len(cases)} cases from {gt_path}")

    rows = measure(cases, args.data, DEFAULT_RADII, args.max_k, args.tol)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {args.out}")

    report(rows, DEFAULT_RADII, DEFAULT_K_VALUES)


if __name__ == "__main__":
    main()
