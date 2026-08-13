"""Stage-by-stage ablation: what each stage of the pipeline actually buys.

Each row adds exactly one stage to the row above, so every delta is
attributable to a single change:

  ncc         plain NCC, argmax of an unweighted surface
  weighted    uniqueness-weighted surface, argmax
  selected    weighted surface, then select_candidate with the centre
              tie-break

The sub-pixel row is the full ``localize()`` and is measured separately in
``results/uniqueness_on.csv``; it needs a refine toggle that lives in R4's
module.

Reuses ``_StageCache`` so pose, template and PSF match the real pipeline
rather than approximating it.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from src import config, disambiguate
from src.evaluate import load_cases, load_image
from src.localize import _resolve_pose, _StageCache
from src.types import FloatArray, Peak

STAGES = ("ncc", "weighted", "selected")

COLUMNS = [
    "case_id",
    "arch",
    "anchored",
    "pose",
    "noise",
    "stage",
    "gt_x",
    "gt_y",
    "pred_x",
    "pred_y",
    "err_px",
    "success_1px",
    "wall_ms",
]


def _score(
    peak: Peak,
    template_shape: tuple[int, int],
    case: dict[str, Any],
    tol: float,
) -> tuple[float, float, float, int]:
    """Return predicted centre, error and pass flag for one peak."""
    x, y = peak.centre(template_shape)
    err = math.hypot(x - float(case["gt_x"]), y - float(case["gt_y"]))
    return float(x), float(y), err, int(err <= tol)


def run_case(case: dict[str, Any], data_dir: Path, tol: float) -> list[dict[str, Any]]:
    """Run every ablation stage on one pair and return one row per stage."""
    reference: FloatArray = np.asarray(load_image(data_dir / case["ref_path"]), dtype=np.float32)
    search: FloatArray = np.asarray(load_image(data_dir / case["search_path"]), dtype=np.float32)
    cache = _StageCache(search, reference)
    pose = _resolve_pose(search, reference, "fast")
    psf = cache.psf_sigma("fast")

    rows: list[dict[str, Any]] = []
    surfaces: dict[bool, tuple[FloatArray, FloatArray, list[Peak], float]] = {}
    for weighted in (False, True):
        started = time.perf_counter()
        template, surface, peaks = cache.correlate(
            theta=pose.theta_deg,
            scale=pose.scale,
            psf_sigma=psf,
            weighted=weighted,
        )
        elapsed = (time.perf_counter() - started) * 1000.0
        surfaces[weighted] = (template, surface, peaks, elapsed)

    for stage in STAGES:
        weighted = stage != "ncc"
        template, surface, peaks, elapsed = surfaces[weighted]
        if not peaks:
            continue
        if stage == "selected":
            started = time.perf_counter()
            _, sidelobe_std = disambiguate.sidelobe_stats(
                surface,
                peaks[0],
                exclusion_radius=config.DEFAULT_NMS_RADIUS_PX,
            )
            tolerance = config.TIE_SIGMA * sidelobe_std
            best, _ = disambiguate.select_candidate(
                peaks,
                config.image_centre(search.shape),
                template.shape,
                tolerance=tolerance,
            )
            elapsed += (time.perf_counter() - started) * 1000.0
        else:
            best = peaks[0]

        x, y, err, ok = _score(best, template.shape, case, tol)
        rows.append(
            {
                "case_id": case["case_id"],
                "arch": case["arch"],
                "anchored": case["anchored"],
                "pose": case["pose"],
                "noise": case["noise"],
                "stage": stage,
                "gt_x": case["gt_x"],
                "gt_y": case["gt_y"],
                "pred_x": x,
                "pred_y": y,
                "err_px": err,
                "success_1px": ok,
                "wall_ms": elapsed,
            }
        )
    return rows


def report(rows: list[dict[str, Any]]) -> None:
    """Print accuracy and median error per stage and anchor stratum."""
    header = f"{'stage':<11} {'stratum':<12} {'n':<5} {'acc':<8} {'median err':<12} {'ms':<8}"
    print("\n" + header)
    print("-" * len(header))
    for stage in STAGES:
        at_s = [r for r in rows if r["stage"] == stage]
        strata = [
            ("overall", at_s),
            ("anchored", [r for r in at_s if r["anchored"] in (True, "anchored")]),
            (
                "unanchored",
                [r for r in at_s if r["anchored"] not in (True, "anchored")],
            ),
        ]
        for name, subset in strata:
            if not subset:
                continue
            acc = sum(r["success_1px"] for r in subset) / len(subset)
            med = float(np.median([r["err_px"] for r in subset]))
            ms = float(np.median([r["wall_ms"] for r in subset]))
            print(f"{stage:<11} {name:<12} {len(subset):<5} {acc:<8.3f} {med:<12.3f} {ms:<8.1f}")
        print()


def main() -> None:
    """Run the ablation sweep from the command line."""
    parser = argparse.ArgumentParser(description="stage ablation")
    parser.add_argument("--data", type=Path, default=Path("dataset_full"))
    parser.add_argument("--gt", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("results/ablation.csv"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tol", type=float, default=1.0)
    args = parser.parse_args()

    gt_path = args.gt or (args.data / "ground_truth.jsonl")
    cases = load_cases(gt_path)
    if args.limit:
        cases = cases[: args.limit]
    print(f"Loaded {len(cases)} cases from {gt_path}")

    rows: list[dict[str, Any]] = []
    for i, case in enumerate(cases, start=1):
        rows.extend(run_case(case, args.data, args.tol))
        print(f"  [{i:>3}/{len(cases)}] {case['case_id']:<30} done")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {args.out}")

    report(rows)


if __name__ == "__main__":
    main()
