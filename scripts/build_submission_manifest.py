"""Build deliverable 6: the per-pair CSV/manifest the submission requires.

Owned by R1.

Section 5 of the problem statement asks for one CSV carrying "Reference path,
search-image path, ground-truth x/y for generated cases, predicted x/y and
per-pair generation metadata", and the final checklist repeats it as "CSV/
manifest contains paths, true coordinates, predictions and metadata".

Every one of those fields already exists. None of them are in the same file:
the paths and the generation metadata live in ``<dataset>/ground_truth.jsonl``,
and the predictions live in ``results/*.csv`` keyed by ``case_id``. This joins
them on ``pair_id`` and writes the single artefact the grader is told to look
for.

Predictions are optional. Run it without ``--results`` and you get the generator
half alone, with the prediction columns present but empty -- which is the honest
shape for a dataset that has been generated but not yet evaluated.

Image paths are emitted relative to the repository root (``<dataset>/reference/
x.png``) rather than as stored in the ground truth (``reference/x.png``), so a
grader can follow them from where the README tells them to stand. Nothing is
written as an absolute path: the checklist forbids hard-coded local paths, and
an absolute path is what broke `pip freeze` for R4 earlier.

Run from the repository root:

    python -m scripts.build_submission_manifest \\
        --dataset dataset --results results/full_324.csv \\
        --out results/submission_manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

COLUMNS: tuple[str, ...] = (
    # Identity and the two image paths the grader follows.
    "pair_id",
    "reference_path",
    "search_path",
    # Ground truth.
    "gt_x",
    "gt_y",
    "gt_rotation_deg",
    "gt_scale",
    # Prediction. Empty when no results file was supplied.
    "pred_x",
    "pred_y",
    "err_px",
    "confidence",
    "low_confidence_flag",
    "mode_used",
    "elapsed_ms",
    # Per-pair generation metadata.
    "seed",
    "architecture",
    "anchor",
    "noise_level",
    "pose_condition",
    "anchors_in_reference",
    "layout_params_json",
    "physics_params_json",
)

_PREDICTION_COLUMNS: tuple[str, ...] = (
    "pred_x",
    "pred_y",
    "err_px",
    "confidence",
    "low_confidence_flag",
    "mode_used",
    "elapsed_ms",
)


def load_ground_truth(dataset_dir: Path) -> list[dict[str, Any]]:
    """Read every ground-truth record for ``dataset_dir``."""
    gt_path = dataset_dir / "ground_truth.jsonl"

    if not gt_path.exists():
        msg = f"no ground truth at {gt_path}"
        raise FileNotFoundError(msg)

    with gt_path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_predictions(results_csv: Path) -> dict[str, dict[str, str]]:
    """Index a results CSV by ``case_id``."""
    if not results_csv.exists():
        msg = f"no results file at {results_csv}"
        raise FileNotFoundError(msg)

    with results_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    return {row["case_id"]: row for row in rows}


def build_rows(
    records: list[dict[str, Any]],
    predictions: dict[str, dict[str, str]],
    dataset_dir: Path,
) -> list[dict[str, Any]]:
    """Join generator records against predictions, one row per pair.

    The join is left-outer on the generator side on purpose. The dataset is the
    thing being described; a pair with no prediction is a real state worth
    reporting, whereas a prediction with no pair is a bug and is reported as
    one by :func:`main`.
    """
    # Always relative, whatever the caller passed. `--dataset` given as an
    # absolute path would otherwise write this machine's directory layout into
    # the deliverable, which is the "no hard-coded local paths" checklist item
    # failing silently -- the manifest still looks complete, and only resolves
    # on the machine that built it. `relpath` never returns an absolute path.
    prefix = Path(os.path.relpath(dataset_dir, Path.cwd())).as_posix()
    rows: list[dict[str, Any]] = []

    for record in records:
        truth = record["ground_truth"]
        strata = record["strata"]
        predicted = predictions.get(record["pair_id"], {})

        row: dict[str, Any] = {
            "pair_id": record["pair_id"],
            "reference_path": f"{prefix}/{record['reference_path']}",
            "search_path": f"{prefix}/{record['search_path']}",
            "gt_x": truth["x"],
            "gt_y": truth["y"],
            "gt_rotation_deg": truth["rotation_deg"],
            "gt_scale": truth["scale"],
            "seed": record["seed"],
            "architecture": strata["architecture"],
            "anchor": strata["anchor"],
            "noise_level": strata["noise_level"],
            "pose_condition": strata["pose_condition"],
            "anchors_in_reference": record["anchors_in_reference"],
            # Kept as JSON rather than exploded into columns: the two
            # architectures carry different layout keys (DRAM has pitch_nm,
            # FinFET has fin_pitch_nm and gate_pitch_nm), so a flat schema
            # would be half-empty on every row.
            "layout_params_json": json.dumps(record["layout_params"], sort_keys=True),
            "physics_params_json": json.dumps(record["physics_params"], sort_keys=True),
        }

        for column in _PREDICTION_COLUMNS:
            row[column] = predicted.get(column, "")

        rows.append(row)

    return rows


def write_manifest(rows: list[dict[str, Any]], out_path: Path) -> Path:
    """Write ``rows`` to ``out_path`` and return it."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    return out_path


def main(argv: list[str] | None = None) -> int:
    """Build the submission manifest from the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help="results CSV to join predictions from; omit for the generator half alone",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/submission_manifest.csv"),
    )
    args = parser.parse_args(argv)

    records = load_ground_truth(args.dataset)
    predictions = load_predictions(args.results) if args.results else {}

    known = {record["pair_id"] for record in records}
    orphans = sorted(set(predictions) - known)

    if orphans:
        # A prediction whose pair is not in this dataset means the results file
        # was measured on different data. Silently dropping those rows would
        # produce a manifest that looks complete and describes two datasets.
        print(
            f"FAILED: {len(orphans)} prediction(s) name pairs absent from "
            f"{args.dataset}, starting with {orphans[0]!r}. The results file "
            "was measured on a different dataset.",
            file=sys.stderr,
        )
        return 1

    rows = build_rows(records, predictions, args.dataset)
    out_path = write_manifest(rows, args.out)

    matched = sum(1 for row in rows if row["pred_x"] != "")
    print(f"Wrote {len(rows)} rows to {out_path}")
    print(f"  predictions joined: {matched}/{len(rows)}")

    if args.results and matched < len(rows):
        print(f"  {len(rows) - matched} pair(s) have no prediction in {args.results}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
