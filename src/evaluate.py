"""R3 evaluation harness.

Runs :func:`src.localize.localize` over every pair in the dataset and writes one
CSV row per case, then prints per-stratum aggregates.

This module is the measurement authority for the team. It is deliberately dumb:
no plots, no config file, no clever caching. Its only job is to turn 36 image
pairs into a table of numbers that R1, R2 and R4 can each point at.

Usage
-----
    python -m src.evaluate
    python -m src.evaluate --data data --out results.csv --limit 4

Notes
-----
``localize()`` returns a single answer, so this harness measures **top-1
accuracy only**. ``recall@K`` requires the candidate list *before*
disambiguation and must call the matcher directly -- see
:func:`recall_at_k_pass` at the bottom of this file. Keeping the two numbers
separate is deliberate: top-1 failing while recall@K passes means the
disambiguator picked wrong; both failing means the matcher never had the answer.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import subprocess
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.localize import localize
from src.types import Mode

# --------------------------------------------------------------------------
# Ground-truth loading
# --------------------------------------------------------------------------

# R1's ground_truth.jsonl nests coordinates under "ground_truth" and stratum
# labels under "strata". Paths below are dotted; plain names are top-level.
# First hit wins, so alternate spellings can be appended without reordering.
_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "case_id": ("pair_id", "case_id", "id"),
    "arch": ("strata.architecture", "architecture", "arch"),
    "anchored": ("strata.anchor", "anchored", "anchor_state"),
    "pose": ("strata.pose_condition", "pose_condition", "pose"),
    "noise": ("strata.noise_level", "noise_level"),
    "seed": ("seed", "rng_seed"),
    "gt_x": ("ground_truth.x", "gt_x", "x"),
    "gt_y": ("ground_truth.y", "gt_y", "y"),
    "gt_rotation_deg": ("ground_truth.rotation_deg", "rotation_deg"),
    "gt_scale": ("ground_truth.scale", "scale"),
    "ref_path": ("reference_path", "ref_path", "reference"),
    "search_path": ("search_path", "search"),
}

_REQUIRED = ("gt_x", "gt_y", "ref_path", "search_path")


def _pick(record: dict[str, object], field: str) -> object:
    """Return the first alias of ``field`` found in ``record``, else None.

    Aliases containing dots are walked as nested keys, so ``ground_truth.x``
    reads ``record["ground_truth"]["x"]``. A missing or non-dict level is
    treated as a miss rather than an error, letting the next alias be tried.
    """
    for alias in _KEY_ALIASES[field]:
        cursor: Any = record
        for part in alias.split("."):
            if not isinstance(cursor, dict) or part not in cursor:
                cursor = None
                break
            cursor = cursor[part]
        if cursor is not None:
            return cursor
    return None


def load_cases(gt_path: Path) -> list[dict[str, Any]]:
    """Read ``ground_truth.jsonl`` into normalised case dictionaries.

    Raises
    ------
    FileNotFoundError
        If the ground-truth file is absent. ``data/`` is gitignored, so this
        means R1's dataset has not been copied onto this machine.
    KeyError
        If a required field cannot be resolved under any known alias. The error
        names the record's actual keys so the alias table can be extended.
    """
    if not gt_path.exists():
        raise FileNotFoundError(
            f"{gt_path} not found. data/ is gitignored, so the dataset does not "
            "arrive with a git pull -- get the 36 pairs from R1, or regenerate "
            "them locally with generate_dataset.py."
        )

    cases: list[dict[str, Any]] = []
    with gt_path.open() as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            case = {field: _pick(record, field) for field in _KEY_ALIASES}

            missing = [f for f in _REQUIRED if case[f] is None]
            if missing:
                raise KeyError(
                    f"{gt_path}:{line_no} -- could not resolve {missing}. "
                    f"Record keys are {sorted(record)}. Add the spelling to "
                    "_KEY_ALIASES."
                )

            if case["case_id"] is None:
                case["case_id"] = f"case_{line_no:03d}"
            cases.append(case)

    return cases


def load_image(path: Path) -> np.ndarray:
    """Load a greyscale image as float32.

    RGB inputs are collapsed to luminance so the optical-bonus pairs run through
    the same harness without a separate code path.
    """
    arr = np.asarray(Image.open(path), dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., :3] @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return arr


# --------------------------------------------------------------------------
# Per-case execution
# --------------------------------------------------------------------------

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
    "gt_rotation_deg",
    "gt_scale",
    "theta_err_deg",
    "scale_err",
    "confidence",
    "low_confidence_flag",
    "ncc_peak",
    "psr",
    "n_tied",
    "tie_break_used",
    "uniqueness_score",
    "theta_est",
    "scale_est",
    "subpixel_error",
    "subpixel_method",
    "mode_used",
    "failure_mode",
    "elapsed_ms",
    "wall_ms",
    "error",
]


def _diag(diagnostics: object, field: str) -> object:
    """Read a diagnostics field, tolerating absence during the build-out."""
    return getattr(diagnostics, field, float("nan"))


def _signed_gap(estimate: object, truth: object) -> float:
    """Return ``estimate - truth`` as a float, or NaN if either is unusable.

    Signed rather than absolute: a pose stage that is consistently biased in one
    direction is a different bug from one that is merely noisy, and averaging
    absolute values hides that distinction.
    """
    try:
        gap = float(estimate) - float(truth)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    return gap if math.isfinite(gap) else float("nan")


def run_case(case: dict[str, Any], data_dir: Path, mode: Mode = "auto") -> dict[str, Any]:
    """Run one pair and return a flat CSV row.

    ``localize()`` is contracted never to raise on valid input, so a non-empty
    ``error`` column means the harness itself broke -- a missing file or a bad
    path -- not the algorithm.
    """
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
            "gt_rotation_deg": case["gt_rotation_deg"],
            "gt_scale": case["gt_scale"],
        }
    )

    try:
        reference = load_image(data_dir / case["ref_path"])
        search = load_image(data_dir / case["search_path"])
    except Exception as exc:  # noqa: BLE001 - harness-level failure, keep going
        row["error"] = f"load: {exc}"
        return row

    started = time.perf_counter()
    try:
        result = localize(search, reference, mode=mode)
    except Exception as exc:  # noqa: BLE001 - contract says this cannot happen
        row["error"] = f"localize: {type(exc).__name__}: {exc}"
        row["wall_ms"] = (time.perf_counter() - started) * 1000.0
        return row
    wall_ms = (time.perf_counter() - started) * 1000.0

    err = math.hypot(result.x - float(case["gt_x"]), result.y - float(case["gt_y"]))
    diagnostics = getattr(result, "diagnostics", None)

    # R1 records true rotation and scale alongside the centre, so the pose
    # stages can be scored directly instead of inferred from position error.
    theta_err = _signed_gap(_diag(diagnostics, "theta_est"), case["gt_rotation_deg"])
    scale_err = _signed_gap(_diag(diagnostics, "scale_est"), case["gt_scale"])

    row.update(
        {
            "pred_x": result.x,
            "pred_y": result.y,
            "err_px": err,
            "success_1px": int(err <= 1.0),
            "theta_err_deg": theta_err,
            "scale_err": scale_err,
            "confidence": result.confidence,
            "low_confidence_flag": int(bool(result.low_confidence_flag)),
            "ncc_peak": _diag(diagnostics, "ncc_peak"),
            "psr": _diag(diagnostics, "psr"),
            "n_tied": _diag(diagnostics, "n_tied"),
            "tie_break_used": _diag(diagnostics, "tie_break_used"),
            "uniqueness_score": _diag(diagnostics, "uniqueness_score"),
            "theta_est": _diag(diagnostics, "theta_est"),
            "scale_est": _diag(diagnostics, "scale_est"),
            "subpixel_error": _diag(diagnostics, "subpixel_error"),
            "subpixel_method": _diag(diagnostics, "subpixel_method"),
            "mode_used": _diag(diagnostics, "mode_used"),
            "failure_mode": _diag(diagnostics, "failure_mode"),
            "elapsed_ms": _diag(diagnostics, "elapsed_ms"),
            "wall_ms": wall_ms,
        }
    )
    return row


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


def _mean(values: list[float]) -> float:
    finite = [v for v in values if isinstance(v, float) and math.isfinite(v)]
    return sum(finite) / len(finite) if finite else float("nan")


def summarise(rows: list[dict[str, Any]], group_by: str) -> list[tuple[Any, ...]]:
    """Aggregate top-1 metrics over one stratum column."""
    buckets: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[row[group_by]].append(row)

    out = []
    for key in sorted(buckets, key=str):
        group = [r for r in buckets[key] if r["success_1px"] != ""]
        if not group:
            continue
        n = len(group)
        successes = sum(int(r["success_1px"]) for r in group)
        errs = [float(r["err_px"]) for r in group]
        finite_errs = sorted(e for e in errs if math.isfinite(e))
        median = finite_errs[len(finite_errs) // 2] if finite_errs else float("nan")
        tie_rate = _mean([float(bool(r["tie_break_used"])) for r in group])
        out.append(
            (
                key,
                n,
                successes / n,
                median,
                _mean([float(r["psr"]) for r in group]),
                tie_rate,
                _mean([float(r["wall_ms"]) for r in group]),
            )
        )
    return out


def print_table(title: str, rows: list[tuple[Any, ...]]) -> None:
    """Print a titled table of per-stratum metrics."""
    print(f"\n{title}")
    print(
        f"{'group':<18}{'n':>4}{'succ@1px':>10}{'med_err':>10}"
        f"{'mean_psr':>10}{'tie_rate':>10}{'ms':>9}"
    )
    print("-" * 71)
    for key, n, succ, med, psr, tie, ms in rows:
        print(f"{key!s:<18}{n:>4}{succ:>10.3f}{med:>10.3f}{psr:>10.3f}{tie:>10.3f}{ms:>9.1f}")


# --------------------------------------------------------------------------
# recall@K -- SECOND PASS, not wired yet
# --------------------------------------------------------------------------


def recall_at_k_pass(cases: list[dict[str, Any]], data_dir: Path, k: int = 30) -> None:
    """Measure whether the true answer was in the top-K *before* disambiguation.

    Not implemented yet: this must call R4's ``zncc_surface()`` and
    ``top_k_peaks()`` directly rather than ``localize()``, because by the time
    ``localize()`` returns, disambiguation has already collapsed the candidate
    list to one answer.

    Confirm with R4 before writing:
      1. exact signatures of ``zncc_surface`` and ``top_k_peaks``
      2. whether the surface is computed at search-image scale or on a
         downsampled reference (the 10x ratio has to be applied somewhere)
      3. whether NMS is applied inside ``top_k_peaks`` -- with
         DEFAULT_NMS_RADIUS_PX = 8 on a 16px lattice, NMS sits exactly at the
         half-pitch tie spacing and may suppress the true peak before it is
         ever counted, which would make recall@K look worse than the matcher
         really is.

    A peak counts as a hit when ``config.window_topleft_to_centre()`` applied to
    it lands within 1px of ground truth. Do not reimplement that offset by hand.
    """
    raise NotImplementedError("Confirm matcher signatures with R4 first.")


# --------------------------------------------------------------------------
# Run provenance
# --------------------------------------------------------------------------


def _git_commit() -> str:
    """Return the current commit, or ``"unknown"`` outside a git checkout."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def write_run_metadata(out: Path, *, cases: int, data_dir: Path, mode: str) -> Path:
    """Record which machine produced a results CSV, beside the CSV.

    Parameters
    ----------
    out
        Path the results CSV was written to. The sidecar takes the same stem
        with a ``.meta.json`` suffix.
    cases
        Number of rows written.
    data_dir
        Dataset the run measured.
    mode
        Mode passed to :func:`~src.localize.localize`.

    Returns
    -------
    Path
        The sidecar that was written.

    Notes
    -----
    Latency is the one headline number that is a property of the machine rather
    than of the code. Two of us measured the same 324 pairs and reported 206 ms
    and 356 ms; both were correct, and neither said on what. A number nobody can
    attribute cannot be defended, so every CSV now carries its machine with it.
    """
    meta = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "commit": _git_commit(),
        "cases": cases,
        "dataset": str(data_dir),
        "mode": mode,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
    }
    meta_path = out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta_path


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main() -> None:
    """Run the evaluation harness from the command line."""
    parser = argparse.ArgumentParser(description="R3 evaluation harness")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument(
        "--gt", type=Path, default=None, help="defaults to <data>/ground_truth.jsonl"
    )
    parser.add_argument("--out", type=Path, default=Path("results.csv"))
    parser.add_argument("--mode", default="auto")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="run only the first N cases (smoke test)",
    )
    args = parser.parse_args()

    gt_path = args.gt or (args.data / "ground_truth.jsonl")
    cases = load_cases(gt_path)
    if args.limit:
        cases = cases[: args.limit]
    print(f"Loaded {len(cases)} cases from {gt_path}")

    rows = []
    for i, case in enumerate(cases, start=1):
        row = run_case(case, args.data, mode=args.mode)
        rows.append(row)
        status = row["error"] or f"err={row['err_px']:.3f}px"
        print(f"  [{i:>3}/{len(cases)}] {row['case_id']:<24} {status}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {args.out}")

    meta_path = write_run_metadata(args.out, cases=len(rows), data_dir=args.data, mode=args.mode)
    print(f"Wrote run metadata to {meta_path}")

    broken = [r for r in rows if r["error"]]
    if broken:
        print(f"WARNING: {len(broken)} cases failed to run (see error column)")

    scored = [r for r in rows if r["success_1px"] != ""]
    if not scored:
        print("No cases scored.")
        return

    overall = sum(int(r["success_1px"]) for r in scored) / len(scored)
    print(f"\nOVERALL success@1px: {overall:.3f}  ({len(scored)} cases)")
    for column in ("arch", "anchored", "pose", "noise"):
        if any(r[column] not in ("", None) for r in scored):
            print_table(f"By {column}", summarise(scored, column))

    print(
        "\nNOTE: top-1 only. recall@K needs the pre-disambiguation candidate "
        "list -- see recall_at_k_pass()."
    )


if __name__ == "__main__":
    main()
