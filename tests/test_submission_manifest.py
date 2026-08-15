"""The submission manifest must satisfy deliverable 6 on its own.

Section 5 names one CSV holding "Reference path, search-image path, ground-truth
x/y for generated cases, predicted x/y and per-pair generation metadata", and
the final checklist repeats it. Every field existed before this script; they
were split across ``ground_truth.jsonl`` and ``results/*.csv`` and nothing
joined them, which is a deliverable that fails on filing rather than on content.

These tests pin the shape a grader is told to expect, and the one silent failure
that would matter: a results file measured on a *different* dataset joining
cleanly against this one and producing a manifest that looks complete.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.build_submission_manifest import COLUMNS, main

_REQUIRED_BY_DELIVERABLE_6 = (
    "reference_path",
    "search_path",
    "gt_x",
    "gt_y",
    "pred_x",
    "pred_y",
    "seed",
)


def _record(pair_id: str, *, x: float, y: float) -> dict[str, object]:
    """One ground-truth record, shaped like the generator's."""
    return {
        "pair_id": pair_id,
        "reference_path": f"reference/{pair_id}.png",
        "search_path": f"search/{pair_id}.png",
        "ground_truth": {"x": x, "y": y, "rotation_deg": 1.5, "scale": 10.0},
        "strata": {
            "architecture": "dram",
            "anchor": "anchored",
            "noise_level": "low",
            "pose_condition": "small",
        },
        "layout_params": {"kind": "dram", "pitch_nm": 180.0},
        "anchors_in_reference": 3,
        "physics_params": {"noise_level": "low", "chain": "src.sem_physics.apply_sem_chain"},
        "seed": 20260807,
    }


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    """A two-pair dataset carrying only its provenance records."""
    root = tmp_path / "dataset_tiny"
    root.mkdir()
    records = [_record("pair_0000", x=100.0, y=200.0), _record("pair_0001", x=300.0, y=400.0)]
    with (root / "ground_truth.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return root


def _write_results(path: Path, case_ids: list[str]) -> Path:
    """A results CSV with the columns the manifest reads."""
    fields = [
        "case_id",
        "pred_x",
        "pred_y",
        "err_px",
        "confidence",
        "low_confidence_flag",
        "mode_used",
        "elapsed_ms",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case_id in case_ids:
            writer.writerow(
                {
                    "case_id": case_id,
                    "pred_x": "101.5",
                    "pred_y": "201.5",
                    "err_px": "2.12",
                    "confidence": "0.8",
                    "low_confidence_flag": "0",
                    "mode_used": "auto",
                    "elapsed_ms": "212.0",
                }
            )
    return path


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_manifest_carries_every_field_deliverable_6_names(dataset_dir: Path, tmp_path: Path):
    """Paths, true coordinates, predictions and metadata, in one file."""
    results = _write_results(tmp_path / "results.csv", ["pair_0000", "pair_0001"])
    out = tmp_path / "manifest.csv"

    assert main(["--dataset", str(dataset_dir), "--results", str(results), "--out", str(out)]) == 0

    rows = _read(out)
    assert len(rows) == 2
    for field in _REQUIRED_BY_DELIVERABLE_6:
        assert field in COLUMNS
        assert rows[0][field] != "", f"{field} is empty"


def test_predictions_join_on_pair_id(dataset_dir: Path, tmp_path: Path):
    """The prediction lands on the right row, not merely on some row."""
    results = _write_results(tmp_path / "results.csv", ["pair_0001"])
    out = tmp_path / "manifest.csv"

    main(["--dataset", str(dataset_dir), "--results", str(results), "--out", str(out)])

    by_id = {row["pair_id"]: row for row in _read(out)}
    assert by_id["pair_0001"]["pred_x"] == "101.5"
    # Left-outer on the generator side: the unevaluated pair still appears.
    assert by_id["pair_0000"]["pred_x"] == ""
    assert by_id["pair_0000"]["gt_x"] == "100.0"


def test_generator_half_alone_is_valid(dataset_dir: Path, tmp_path: Path):
    """Without --results every pair is still described, predictions blank."""
    out = tmp_path / "manifest.csv"

    assert main(["--dataset", str(dataset_dir), "--out", str(out)]) == 0

    rows = _read(out)
    assert len(rows) == 2
    assert all(row["pred_x"] == "" for row in rows)
    assert all(row["gt_x"] != "" for row in rows)


def test_results_from_another_dataset_are_rejected(dataset_dir: Path, tmp_path: Path):
    """A manifest describing two datasets must fail loudly, not join quietly."""
    results = _write_results(tmp_path / "results.csv", ["pair_0000", "pair_from_elsewhere"])
    out = tmp_path / "manifest.csv"

    assert main(["--dataset", str(dataset_dir), "--results", str(results), "--out", str(out)]) == 1
    assert not out.exists(), "a rejected run must not leave a partial manifest behind"


def test_image_paths_are_relative_and_dataset_qualified(
    dataset_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The checklist forbids hard-coded local paths.

    Run the way the README says to: from the directory the dataset sits in.
    """
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "manifest.csv"
    main(["--dataset", str(dataset_dir), "--out", str(out)])

    for row in _read(out):
        for column in ("reference_path", "search_path"):
            value = row[column]
            assert not Path(value).is_absolute(), f"{column} is absolute: {value}"
            # Qualified by the dataset directory, so it resolves from the root
            # the README tells a grader to stand in.
            assert value.startswith(dataset_dir.name + "/"), value


def test_an_absolute_dataset_path_never_reaches_the_manifest(
    dataset_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Passing an absolute --dataset must not leak this machine's layout."""
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "manifest.csv"

    main(["--dataset", str(dataset_dir.resolve()), "--out", str(out)])

    for row in _read(out):
        assert not Path(row["reference_path"]).is_absolute(), row["reference_path"]
        assert not Path(row["search_path"]).is_absolute(), row["search_path"]


_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_tracked_manifest_matches_the_tracked_results():
    """The shipped deliverable must quote the predictions the repo ships.

    This file is downstream of ``results/full_324.csv`` and nothing regenerates
    it when that changes, so it goes stale silently. It already did once: it was
    built the morning Stage 1 landed and merged after the 324-pair results were
    regenerated against pose, leaving 232 of 324 rows carrying pre-pose
    predictions. The manifest implied anchored accuracy of 0.852 while the
    README, the deck and the failure analysis all said 0.938 -- a contradiction
    a grader finds by joining the one CSV they are handed against our headline.

    Reads only tracked CSVs, never the dataset, so it runs on a clean checkout.
    Regenerate with::

        python -m scripts.build_submission_manifest --dataset dataset \\
            --results results/full_324.csv --out results/submission_manifest.csv
    """
    manifest_path = _REPO_ROOT / "results" / "submission_manifest.csv"
    results_path = _REPO_ROOT / "results" / "full_324.csv"
    assert manifest_path.exists(), f"deliverable 6 is missing: {manifest_path}"
    assert results_path.exists(), f"canonical results are missing: {results_path}"

    predictions = {row["case_id"]: row for row in _read(results_path)}
    stale = [
        row["pair_id"]
        for row in _read(manifest_path)
        if row["pair_id"] in predictions
        and (
            row["pred_x"] != predictions[row["pair_id"]]["pred_x"]
            or row["pred_y"] != predictions[row["pair_id"]]["pred_y"]
        )
    ]

    assert not stale, (
        f"{len(stale)} of {len(predictions)} rows disagree with results/full_324.csv; "
        f"rebuild the manifest (first: {stale[:3]})"
    )
