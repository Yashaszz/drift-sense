"""Plain-NCC baseline: the incumbent the full pipeline must beat.

Single-scale ZNCC at the nominal 10x decimation with no pose estimation,
no uniqueness weighting, no disambiguation and no sub-pixel refinement --
the global argmax of one correlation surface. This is what a reasonable
first attempt looks like, and it is what ``src/disambiguate.py`` describes
as the behaviour its stub reduced to.

Reuses ``build_template`` and ``zncc_surface`` so that the gap against the
full pipeline isolates the added stages rather than confounding them with
a different correlator.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from src import config, matcher
from src.evaluate import load_cases, load_image

COLUMNS = [
    "case_id",
    "arch",
    "anchored",
    "pose",
    "noise",
    "seed",
    "gt_x",
    "gt_y",
    "pred_x",
    "pred_y",
    "err_px",
    "success_1px",
    "wall_ms",
    "error",
]


def run_case(case: dict[str, Any], data_dir: Path, tol: float) -> dict[str, Any]:
    """Localise one pair with plain NCC and return a flat CSV row."""
    row: dict[str, Any] = {col: "" for col in COLUMNS}
    row.update(
        {
            "case_id": case["case_id"],
            "arch": case["arch"],
            "anchored": case["anchored"],
            "pose": case["pose"],
            "noise": case["noise"],
            "seed": case["seed"],
            "gt_x": case["gt_x"],
            "gt_y": case["gt_y"],
        }
    )
    try:
        reference = np.asarray(load_image(data_dir / case["ref_path"]), dtype=np.float32)
        search = np.asarray(load_image(data_dir / case["search_path"]), dtype=np.float32)
    except Exception as exc:  # noqa: BLE001 - harness failure, keep going
        row["error"] = f"load: {exc}"
        return row

    started = time.perf_counter()
    template = matcher.build_template(
        reference,
        theta=0.0,
        scale=float(config.NOMINAL_SCALE),
        psf_sigma_px=float(config.DEFAULT_PSF_SIGMA_PX),
    )
    surface = matcher.zncc_surface(template, search, weight=None)
    peaks = matcher.top_k_peaks(surface, k=1, nms_radius=0)
    row["wall_ms"] = (time.perf_counter() - started) * 1000.0

    if not peaks:
        row["error"] = "no candidates"
        return row

    pred_x, pred_y = peaks[0].centre(template.shape)
    err = math.hypot(pred_x - float(case["gt_x"]), pred_y - float(case["gt_y"]))
    row["pred_x"] = pred_x
    row["pred_y"] = pred_y
    row["err_px"] = err
    row["success_1px"] = int(err <= tol)
    return row


def summarise(rows: list[dict[str, Any]]) -> None:
    """Print accuracy and median error, overall and by anchor stratum."""
    scored = [r for r in rows if r["success_1px"] != ""]
    strata = [
        ("overall", scored),
        ("anchored", [r for r in scored if r["anchored"] in (True, "anchored")]),
        (
            "unanchored",
            [r for r in scored if r["anchored"] not in (True, "anchored")],
        ),
    ]
    print(f"\n{'stratum':<12} {'n':<5} {'acc@1px':<9} {'median err':<12} {'ms':<8}")
    print("-" * 48)
    for name, subset in strata:
        if not subset:
            print(f"{name:<12} {0:<5} {'nan':<9} {'nan':<12} {'nan':<8}")
            continue
        acc = sum(r["success_1px"] for r in subset) / len(subset)
        med = float(np.median([r["err_px"] for r in subset]))
        ms = float(np.median([r["wall_ms"] for r in subset]))
        print(f"{name:<12} {len(subset):<5} {acc:<9.3f} {med:<12.3f} {ms:<8.1f}")


def main() -> None:
    """Run the plain-NCC baseline from the command line."""
    parser = argparse.ArgumentParser(description="plain-NCC baseline")
    parser.add_argument("--data", type=Path, default=Path("dataset_full"))
    parser.add_argument("--gt", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("results/baseline.csv"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tol", type=float, default=1.0)
    args = parser.parse_args()

    gt_path = args.gt or (args.data / "ground_truth.jsonl")
    cases = load_cases(gt_path)
    if args.limit:
        cases = cases[: args.limit]
    print(f"Loaded {len(cases)} cases from {gt_path}")

    rows = []
    for i, case in enumerate(cases, start=1):
        row = run_case(case, args.data, args.tol)
        rows.append(row)
        status = row["error"] or f"err={row['err_px']:.3f}px"
        print(f"  [{i:>3}/{len(cases)}] {row['case_id']:<30} {status}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {args.out}")

    summarise(rows)


if __name__ == "__main__":
    main()
