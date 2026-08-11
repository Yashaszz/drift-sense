"""T7 — the Stage 6 confidence calibrator.

The model is fitted, serialised and wired, but on the current data it does not
discriminate: cross-validated AUC is 0.504, which is chance. That is not a
defect in the calibrator, it is a consequence of four of its six features being
constant because the diagnostics feeding them are still stubs upstream.

These tests therefore assert the mechanics — determinism, NaN handling,
round-tripping, threshold semantics — and separately assert that the model
*would* learn from an informative feature, so the handover is covered when R2
and R3 land.
"""

import numpy as np
import pytest

from src import config
from src.confidence import (
    FEATURE_NAMES,
    ConfidenceModel,
    extract_features,
    heuristic_confidence,
    is_low_confidence,
)
from src.types import Diagnostics


def diag(**kwargs) -> Diagnostics:
    """Build a Diagnostics with the given overrides."""
    return Diagnostics(**kwargs)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def test_feature_vector_matches_the_declared_names():
    assert extract_features(diag()).shape == (len(FEATURE_NAMES),)


def test_features_are_deterministic():
    d = diag(ncc_peak=0.9, psr=5.0, n_tied=3, uniqueness_score=0.4)
    np.testing.assert_array_equal(extract_features(d), extract_features(d))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_diagnostics_never_reach_the_model(bad):
    """One NaN would poison a whole fitted model."""
    features = extract_features(diag(psr=bad, uniqueness_score=bad, scale_est=bad))
    assert np.all(np.isfinite(features))


def test_negative_tie_counts_do_not_break_the_log():
    assert np.all(np.isfinite(extract_features(diag(n_tied=-5))))


# ---------------------------------------------------------------------------
# Heuristic
# ---------------------------------------------------------------------------


def test_heuristic_is_monotone_in_psr():
    scores = [heuristic_confidence(diag(psr=p)) for p in (0.0, 2.0, 5.0, 8.0, 12.0, 30.0)]
    assert scores == sorted(scores)


def test_heuristic_is_bounded():
    for psr in (-100.0, 0.0, 1e6):
        assert 0.0 <= heuristic_confidence(diag(psr=psr)) <= 1.0


def test_heuristic_crosses_half_at_the_acceptance_threshold():
    """The heuristic and the escalation logic must agree by construction."""
    assert heuristic_confidence(diag(psr=config.PSR_ACCEPT_THRESHOLD)) == pytest.approx(0.5)


def test_heuristic_scores_unmeasurable_psr_at_zero():
    assert heuristic_confidence(diag(psr=float("nan"))) == 0.0


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def synthetic_dataset(n: int = 200, informative: bool = True):
    """Build a feature matrix where one column predicts the label, or none does."""
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, size=n).astype(float)
    features = rng.normal(0.0, 1.0, size=(n, len(FEATURE_NAMES)))
    if informative:
        features[:, 1] = labels * 3.0 + rng.normal(0.0, 0.3, size=n)
    return features, labels


def test_fit_learns_an_informative_feature():
    """The calibrator works; the current data is what lacks signal.

    Guards against concluding from the chance-level result on real data that the
    model itself is broken.
    """
    features, labels = synthetic_dataset(informative=True)
    model = ConfidenceModel().fit(features, labels)

    assert model.is_fitted
    assert model.coefficients is not None
    assert abs(model.coefficients[1]) > 1.0

    high = model.predict(diag(psr=10.0))
    low = model.predict(diag(psr=-10.0))
    assert high > low


def test_fit_is_deterministic():
    features, labels = synthetic_dataset()
    first = ConfidenceModel().fit(features, labels)
    second = ConfidenceModel().fit(features, labels)
    assert first.coefficients == second.coefficients
    assert first.intercept == second.intercept


def test_fit_tolerates_constant_features():
    """Four of six features are constant today; fitting must not divide by zero."""
    features, labels = synthetic_dataset()
    features[:, 2] = 0.0
    features[:, 4] = 7.0

    model = ConfidenceModel().fit(features, labels)
    assert model.spread is not None
    assert all(np.isfinite(model.spread))
    assert np.isfinite(model.predict(diag()))


def test_fit_gives_a_constant_feature_no_weight():
    features, labels = synthetic_dataset()
    features[:, 4] = 3.0
    model = ConfidenceModel().fit(features, labels)
    assert model.coefficients is not None
    assert model.coefficients[4] == pytest.approx(0.0, abs=1e-9)


def test_fit_rejects_mismatched_shapes():
    features, labels = synthetic_dataset()
    with pytest.raises(ValueError, match="sample count"):
        ConfidenceModel().fit(features, labels[:-1])


# ---------------------------------------------------------------------------
# Prediction and serialisation
# ---------------------------------------------------------------------------


def test_prediction_is_bounded_and_deterministic():
    features, labels = synthetic_dataset()
    model = ConfidenceModel().fit(features, labels)
    d = diag(ncc_peak=0.8, psr=4.0)
    assert 0.0 <= model.predict(d) <= 1.0
    assert model.predict(d) == model.predict(d)


def test_unfitted_model_falls_back_to_the_heuristic():
    model = ConfidenceModel()
    assert not model.is_fitted
    assert model.predict(diag(psr=6.0)) == heuristic_confidence(diag(psr=6.0))


def test_fitted_model_round_trips(tmp_path):
    features, labels = synthetic_dataset()
    original = ConfidenceModel().fit(features, labels)
    original.threshold = 0.37

    path = tmp_path / "confidence.json"
    original.save(path)
    restored = ConfidenceModel.load(path)

    assert restored.coefficients == original.coefficients
    assert restored.centre == original.centre
    assert restored.spread == original.spread
    assert restored.threshold == pytest.approx(0.37)

    d = diag(ncc_peak=0.7, psr=3.0)
    assert restored.predict(d) == pytest.approx(original.predict(d))


def test_load_rejects_a_stale_feature_order(tmp_path):
    import json

    path = tmp_path / "stale.json"
    path.write_text(json.dumps({"feature_names": ["only_one"], "coefficients": [1.0]}))
    with pytest.raises(ValueError, match="different feature order"):
        ConfidenceModel.load(path)


def test_missing_model_is_not_an_error(tmp_path):
    """The deliverable must run from a clean unzip with no trained artefacts."""
    model = ConfidenceModel.load_or_default(tmp_path / "absent.json")
    assert not model.is_fitted
    assert 0.0 <= model.predict(diag(psr=5.0)) <= 1.0


# ---------------------------------------------------------------------------
# Threshold semantics
# ---------------------------------------------------------------------------


def test_threshold_governs_the_flag():
    confident = diag(psr=config.PSR_AMBIGUOUS_THRESHOLD + 5.0)
    assert is_low_confidence(0.4, confident, threshold=0.3) is False
    assert is_low_confidence(0.4, confident, threshold=0.6) is True


def test_hard_signals_override_the_threshold():
    """A miscalibrated model must not be able to clear a known-bad case."""
    assert is_low_confidence(1.0, diag(failure_mode="internal_error"), threshold=0.0) is True
    assert is_low_confidence(1.0, diag(psr=float("nan")), threshold=0.0) is True
    assert is_low_confidence(1.0, diag(psr=0.0), threshold=0.0) is True


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------


def test_localize_confidence_is_in_range(two_scale_pair):
    from src.localize import localize

    reference, search, _ = two_scale_pair
    result = localize(search, reference, mode="auto")
    assert 0.0 <= result.confidence <= 1.0
    assert isinstance(result.low_confidence_flag, bool)


def test_localize_uses_a_fitted_model_when_one_is_present(tmp_path, monkeypatch, two_scale_pair):
    """Dropping a model file in must change the reported confidence."""
    from src import localize as localize_module

    features, labels = synthetic_dataset()
    model = ConfidenceModel().fit(features, labels)
    path = tmp_path / "confidence.json"
    model.save(path)

    reference, search, _ = two_scale_pair
    baseline = localize_module.localize(search, reference, mode="fast").confidence

    monkeypatch.setattr(localize_module.config, "CONFIDENCE_MODEL_PATH", str(path))
    localize_module._confidence_model.cache_clear()
    try:
        fitted = localize_module.localize(search, reference, mode="fast").confidence
    finally:
        localize_module._confidence_model.cache_clear()

    assert fitted != baseline


def test_confidence_is_deterministic_across_calls(two_scale_pair):
    from src.localize import localize

    reference, search, _ = two_scale_pair
    first = localize(search, reference, mode="auto")
    second = localize(search, reference, mode="auto")
    assert first.confidence == second.confidence
    assert first.low_confidence_flag == second.low_confidence_flag
