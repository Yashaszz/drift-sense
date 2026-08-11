"""Fit and evaluate the Stage 6 confidence model on the generated dataset.

Runs ``localize`` over every labelled pair, turns the diagnostics into the
feature matrix, fits a logistic calibrator, and reports what it is actually
worth: cross-validated discrimination, feature importance, and the trade-off
between trusting a wrong answer and escalating a right one.

Cross-validated rather than in-sample. With around a hundred pairs and six
features, an in-sample score would mostly measure memorisation, and a confidence
model that is itself overconfident is worse than no model at all.

Usage
-----
    python -m benchmarks.fit_confidence --data data
    python -m benchmarks.fit_confidence --data data --out models/confidence.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.confidence import FEATURE_NAMES, ConfidenceModel, extract_features  # noqa: E402
from src.localize import localize  # noqa: E402
from src.types import Diagnostics  # noqa: E402

FOLDS = 5


def collect(root: Path) -> tuple[list[Diagnostics], np.ndarray, list[dict[str, Any]]]:
    """Run the pipeline over every labelled pair.

    Parameters
    ----------
    root
        Dataset directory containing ``ground_truth.jsonl``.

    Returns
    -------
    tuple
        Diagnostics per pair, a boolean correctness array, and the raw records.
    """
    records = [json.loads(line) for line in (root / "ground_truth.jsonl").read_text().splitlines()]
    diagnostics: list[Diagnostics] = []
    correct: list[bool] = []

    for record in records:
        reference = np.asarray(Image.open(root / record["reference_path"]).convert("L"))
        search = np.asarray(Image.open(root / record["search_path"]).convert("L"))
        truth = record["ground_truth"]

        result = localize(search, reference, mode="auto")
        error = math.dist((result.x, result.y), (truth["x"], truth["y"]))

        diagnostics.append(result.diagnostics)
        correct.append(error <= config.HEADLINE_TOLERANCE_PX)
        record["error_px"] = error

    return diagnostics, np.asarray(correct, dtype=bool), records


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Return the area under the ROC curve, by rank.

    Parameters
    ----------
    scores
        Predicted confidence per sample.
    labels
        True correctness per sample.

    Returns
    -------
    float
        AUC in ``[0, 1]``, or NaN when one class is absent.
    """
    positives = scores[labels]
    negatives = scores[~labels]
    if positives.size == 0 or negatives.size == 0:
        return float("nan")
    order = np.concatenate([positives, negatives]).argsort().argsort().astype(float) + 1.0
    rank_sum = order[: positives.size].sum()
    return float(
        (rank_sum - positives.size * (positives.size + 1) / 2) / (positives.size * negatives.size)
    )


def cross_validated_scores(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Return out-of-fold confidence predictions.

    Parameters
    ----------
    features
        Feature matrix.
    labels
        Correctness labels.

    Returns
    -------
    numpy.ndarray
        One out-of-fold prediction per sample.
    """
    from sklearn.model_selection import StratifiedKFold

    predictions = np.zeros(labels.shape[0], dtype=np.float64)
    splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=0)

    for train_index, test_index in splitter.split(features, labels):
        model = ConfidenceModel().fit(features[train_index], labels[train_index].astype(float))
        for position in test_index:
            predictions[position] = _predict_from_row(model, features[position])
    return predictions


def _predict_from_row(model: ConfidenceModel, row: np.ndarray) -> float:
    """Score one already-extracted feature row.

    Parameters
    ----------
    model
        A fitted calibrator.
    row
        One row of the feature matrix.

    Returns
    -------
    float
        Confidence in ``[0, 1]``.
    """
    if model.centre is None or model.spread is None or model.coefficients is None:
        return 0.0
    standardised = (row - np.asarray(model.centre)) / np.asarray(model.spread)
    logit = float(np.dot(standardised, np.asarray(model.coefficients))) + model.intercept
    return float(1.0 / (1.0 + np.exp(-logit)))


def sweep_threshold(scores: np.ndarray, labels: np.ndarray) -> list[dict[str, float]]:
    """Tabulate the confidence/escalation trade-off across thresholds.

    Parameters
    ----------
    scores
        Out-of-fold confidence per sample.
    labels
        True correctness per sample.

    Returns
    -------
    list of dict
        One row per threshold.
    """
    rows = []
    for threshold in np.round(np.arange(0.05, 1.0, 0.05), 2):
        trusted = scores >= threshold
        trusted_count = int(trusted.sum())
        false_confidence = int((trusted & ~labels).sum())
        escalated_but_right = int((~trusted & labels).sum())
        rows.append(
            {
                "threshold": float(threshold),
                "trusted": trusted_count,
                "precision": float((trusted & labels).sum() / trusted_count)
                if trusted_count
                else float("nan"),
                "recall": float((trusted & labels).sum() / max(1, int(labels.sum()))),
                "false_confidence": false_confidence,
                "wasted_escalation": escalated_but_right,
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    """Fit, evaluate and optionally save the confidence model.

    Parameters
    ----------
    argv
        Argument list; defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        ``0`` on success, ``1`` when the dataset is missing.
    """
    parser = argparse.ArgumentParser(description="Fit the Stage 6 confidence model.")
    parser.add_argument("--data", type=Path, default=Path("data"), help="dataset root")
    parser.add_argument("--out", type=Path, default=None, help="where to write the fitted model")
    args = parser.parse_args(argv)

    if not (args.data / "ground_truth.jsonl").is_file():
        parser.exit(status=1, message=f"error: no ground_truth.jsonl under {args.data}\n")

    diagnostics, labels, records = collect(args.data)
    features = np.vstack([extract_features(d) for d in diagnostics])

    print(f"{len(labels)} pairs | {int(labels.sum())} correct | {int((~labels).sum())} wrong")
    print(f"pipeline accuracy @ {config.HEADLINE_TOLERANCE_PX} px: {100 * labels.mean():.1f}%")

    print()
    print("Per-feature discrimination")
    print(f"{'feature':<20} {'variance':>12} {'AUC':>8}  status")
    print("-" * 60)
    for index, name in enumerate(FEATURE_NAMES):
        column = features[:, index]
        variance = float(np.var(column))
        note = "dead - constant" if variance < 1e-12 else ""
        print(f"{name:<20} {variance:>12.3e} {roc_auc(column, labels):>8.3f}  {note}")

    scores = cross_validated_scores(features, labels)
    print()
    print(f"Cross-validated AUC ({FOLDS}-fold): {roc_auc(scores, labels):.3f}")

    model = ConfidenceModel().fit(features, labels.astype(float))
    print()
    print("Feature importance (standardised coefficients)")
    print(f"{'feature':<20} {'coefficient':>12}")
    print("-" * 34)
    assert model.coefficients is not None
    for name, weight in sorted(
        zip(FEATURE_NAMES, model.coefficients, strict=True),
        key=lambda pair: -abs(pair[1]),
    ):
        print(f"{name:<20} {weight:>12.4f}")

    print()
    print("Threshold trade-off (out-of-fold)")
    print(
        f"{'thr':>5} {'trusted':>8} {'precision':>10} {'recall':>8} "
        f"{'false conf':>11} {'wasted esc':>11}"
    )
    print("-" * 58)
    sweep = sweep_threshold(scores, labels)
    for row in sweep:
        print(
            f"{row['threshold']:>5.2f} {row['trusted']:>8} {row['precision']:>10.3f} "
            f"{row['recall']:>8.3f} {row['false_confidence']:>11} {row['wasted_escalation']:>11}"
        )

    # Pick the threshold that admits no false confidence, preferring the one
    # that then wastes the fewest escalations. Trusting a wrong answer is the
    # expensive error on a wafer tool: it silently corrupts a measurement,
    # whereas an unnecessary escalation only costs time.
    clean = [row for row in sweep if row["false_confidence"] == 0 and row["trusted"] > 0]
    chosen = min(clean, key=lambda row: row["wasted_escalation"]) if clean else None
    if chosen is None:
        print("\nNo threshold admits zero false confidence on this data.")
    else:
        model.threshold = chosen["threshold"]
        print(
            f"\nCalibrated threshold {model.threshold:.2f}: "
            f"trusts {chosen['trusted']} answers, {chosen['false_confidence']} of them wrong, "
            f"{chosen['wasted_escalation']} correct answers unnecessarily escalated."
        )

    if args.out is not None:
        model.save(args.out)
        print(f"Wrote {args.out}")

    by_anchor: dict[str, list[bool]] = {}
    for record, is_correct in zip(records, labels, strict=True):
        by_anchor.setdefault(record["strata"]["anchor"], []).append(bool(is_correct))
    print()
    print("Accuracy by anchor stratum")
    for anchor, values in sorted(by_anchor.items()):
        print(f"  {anchor:<12} {100 * float(np.mean(values)):>5.1f}%  (n={len(values)})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
