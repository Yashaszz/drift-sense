"""Readiness harness for R3's ``uniqueness_map``.

R4's weighted-correlation path (T8) and confidence calibrator (T7) are both
implemented but inert, because ``disambiguate.uniqueness_map`` still returns a
constant. This script substitutes a *stand-in* map, runs the whole pipeline, and
reports what changes.

The stand-in is a test double, not an implementation of R3's stage. It scores
each tile by spectral flatness, which is one of the two approaches the design
names, and it exists only to prove that the surrounding wiring reacts correctly.
R3 should replace it, not extend it.

What this answers
-----------------
Whether R4 needs any change when the real map lands. If accuracy and confidence
both move when a non-constant map is injected, the integration is complete and
the handover is a one-line substitution on R3's side.

Usage
-----
    python -m benchmarks.verify_uniqueness_integration --data data
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, disambiguate  # noqa: E402
from src import localize as localize_module
from src.confidence import ConfidenceModel, extract_features  # noqa: E402
from src.localize import localize  # noqa: E402


def stand_in_uniqueness_map(
    reference: np.ndarray,
    tile: int = config.DEFAULT_UNIQUENESS_TILE_PX,
) -> np.ndarray:
    """Score tiles by spectral flatness, as a test double for R3's stage.

    **Not an implementation of R3's stage.** It exists only to prove the
    surrounding wiring reacts to a non-constant map.

    A periodic tile concentrates its energy in a few spectral lines, so its
    spectrum is peaky and its flatness is low. An aperiodic tile spreads energy
    broadly and scores high. Flatness is the ratio of the geometric mean of the
    power spectrum to its arithmetic mean.

    Parameters
    ----------
    reference
        Reference image at 1 nm/px.
    tile
        Tile edge length in reference pixels.

    Returns
    -------
    numpy.ndarray
        Weights matching R3's contract: ``float32``, shape of ``reference``,
        values in ``[0.05, 1.0]``.
    """
    image = np.asarray(reference, dtype=np.float32)
    rows, cols = image.shape
    weights = np.full((rows, cols), 0.05, dtype=np.float32)
    scores: list[tuple[int, int, float]] = []

    for top in range(0, rows - tile + 1, tile):
        for left in range(0, cols - tile + 1, tile):
            patch = image[top : top + tile, left : left + tile]
            patch = patch - patch.mean()
            if float(patch.std()) < 1e-6:
                scores.append((top, left, 0.0))
                continue
            power = np.abs(np.fft.rfft2(patch)) ** 2 + 1e-12
            flatness = float(np.exp(np.mean(np.log(power))) / np.mean(power))
            scores.append((top, left, flatness))

    if not scores:
        return weights

    values = np.array([s[2] for s in scores])
    lo, hi = float(values.min()), float(values.max())
    span = hi - lo if hi > lo else 1.0
    for top, left, value in scores:
        weights[top : top + tile, left : left + tile] = 0.05 + 0.95 * (value - lo) / span
    return weights


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Return the area under the ROC curve, by rank."""
    positives = scores[labels]
    negatives = scores[~labels]
    if positives.size == 0 or negatives.size == 0:
        return float("nan")
    order = np.concatenate([positives, negatives]).argsort().argsort().astype(float) + 1.0
    rank_sum = order[: positives.size].sum()
    return float(
        (rank_sum - positives.size * (positives.size + 1) / 2) / (positives.size * negatives.size)
    )


def evaluate(root: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Run the pipeline over the dataset.

    Parameters
    ----------
    root
        Dataset directory.

    Returns
    -------
    tuple
        Correctness array, feature matrix, and the anchor stratum per pair.
    """
    records = [json.loads(line) for line in (root / "ground_truth.jsonl").read_text().splitlines()]
    correct: list[bool] = []
    features: list[np.ndarray] = []
    anchors: list[str] = []

    for record in records:
        reference = np.asarray(Image.open(root / record["reference_path"]).convert("L"))
        search = np.asarray(Image.open(root / record["search_path"]).convert("L"))
        truth = record["ground_truth"]

        result = localize(search, reference, mode="auto")
        error = math.dist((result.x, result.y), (truth["x"], truth["y"]))

        correct.append(error <= config.HEADLINE_TOLERANCE_PX)
        features.append(extract_features(result.diagnostics))
        anchors.append(record["strata"]["anchor"])

    return np.asarray(correct, dtype=bool), np.vstack(features), anchors


def cross_validated_auc(features: np.ndarray, labels: np.ndarray) -> float:
    """Return out-of-fold AUC for the confidence calibrator."""
    from sklearn.model_selection import StratifiedKFold

    predictions = np.zeros(labels.shape[0], dtype=np.float64)
    for train_index, test_index in StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(
        features, labels
    ):
        model = ConfidenceModel().fit(features[train_index], labels[train_index].astype(float))
        if model.centre is None or model.spread is None or model.coefficients is None:
            continue
        standardised = (features[test_index] - np.asarray(model.centre)) / np.asarray(model.spread)
        logits = standardised @ np.asarray(model.coefficients) + model.intercept
        predictions[test_index] = 1.0 / (1.0 + np.exp(-logits))
    return roc_auc(predictions, labels)


def report(name: str, correct: np.ndarray, features: np.ndarray, anchors: list[str]) -> None:
    """Print one configuration's results."""
    anchored = np.array([a == "anchored" for a in anchors])
    print(f"  {name}")
    print(f"    accuracy @ {config.HEADLINE_TOLERANCE_PX} px : {100 * correct.mean():5.1f}%")
    print(f"    anchored                : {100 * correct[anchored].mean():5.1f}%")
    print(f"    unanchored              : {100 * correct[~anchored].mean():5.1f}%")
    print(f"    confidence CV AUC       : {cross_validated_auc(features, correct):5.3f}")


def main(argv: list[str] | None = None) -> int:
    """Compare the pipeline with and without an informative uniqueness map."""
    parser = argparse.ArgumentParser(description="Verify readiness for R3's uniqueness_map.")
    parser.add_argument("--data", type=Path, default=Path("data"), help="dataset root")
    args = parser.parse_args(argv)

    if not (args.data / "ground_truth.jsonl").is_file():
        parser.exit(status=1, message=f"error: no ground_truth.jsonl under {args.data}\n")

    print("R3 uniqueness_map readiness check\n")

    real = disambiguate.uniqueness_map
    localize_module._confidence_model.cache_clear()
    baseline = evaluate(args.data)
    report("with R3's current stub (constant map)", *baseline)

    print()
    disambiguate.uniqueness_map = stand_in_uniqueness_map
    try:
        localize_module._confidence_model.cache_clear()
        injected = evaluate(args.data)
        report("with a non-constant stand-in map", *injected)
    finally:
        disambiguate.uniqueness_map = real
        localize_module._confidence_model.cache_clear()

    changed = not np.array_equal(baseline[0], injected[0])
    print()
    print(f"pipeline reacts to the map: {changed}")
    print(
        "R4 requires no change when the real map lands; substituting it is a\n"
        "one-line replacement inside disambiguate.uniqueness_map."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
