"""T8 — uniqueness-weighted correlation.

The weighted path is checkable today even though ``uniqueness_map`` still
returns a constant, because the constant case has an exact expected answer: it
must reproduce the unweighted surface. Everything else here uses synthetic
non-constant weights, so the weighting is exercised for real rather than only
through the identity case.
"""

import numpy as np
import pytest

from src import config, matcher
from src.localize import _StageCache
from tests.conftest import TEMPLATE_SIZE, crop, make_periodic_field


def argmax_colrow(surface: np.ndarray) -> tuple[int, int]:
    """Return the surface argmax as ``(col, row)``."""
    row, col = np.unravel_index(int(np.argmax(surface)), surface.shape)
    return int(col), int(row)


@pytest.fixture
def field_and_template():
    field = make_periodic_field()
    return field, crop(field, 250, 120, TEMPLATE_SIZE), (250, 120)


def contract_weights(shape, seed: int = 0) -> np.ndarray:
    """Random weights obeying R3's contract: float32, in ``[0.05, 1.0]``."""
    return np.random.default_rng(seed).uniform(0.05, 1.0, shape).astype(np.float32)


# ---------------------------------------------------------------------------
# Constant-weight equivalence — the contract between the two paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("level", [1.0, 0.05, 0.5, 7.3, 1e-3])
def test_constant_weight_of_any_level_reproduces_unweighted(field_and_template, level):
    """Normalising by the weight sum makes the level irrelevant.

    Every weighted mean collapses to the ordinary mean and the common factor
    cancels between numerator and denominator, so a constant map of *any*
    magnitude must give back plain ZNCC.
    """
    field, template, _ = field_and_template
    unweighted = matcher.zncc_surface(template, field)
    weighted = matcher.zncc_surface(template, field, weight=np.full_like(template, level))
    np.testing.assert_allclose(weighted, unweighted, atol=1e-5)


def test_constant_weight_equivalence_is_tight(field_and_template):
    """Records how close the two paths actually agree, not merely that they do."""
    field, template, _ = field_and_template
    unweighted = matcher.zncc_surface(template, field)
    weighted = matcher.zncc_surface(template, field, weight=np.ones_like(template))
    assert float(np.abs(weighted - unweighted).max()) < 1e-6


def test_uniqueness_stub_is_still_constant():
    """Guards the assumption the equivalence tests rest on.

    While ``uniqueness_map`` returns a constant, weighted correlation provably
    equals unweighted, so the weighted path cannot yet improve anything. When R3
    lands a real map this test fails, which is the signal that the weighting has
    started doing work and the accuracy comparison becomes meaningful.
    """
    from src.disambiguate import uniqueness_map

    weights = uniqueness_map(np.zeros((256, 256), dtype=np.float32))
    assert float(weights.min()) == float(weights.max())


# ---------------------------------------------------------------------------
# Weighted behaviour with genuinely non-constant weights
# ---------------------------------------------------------------------------


def test_non_constant_weights_change_the_surface(field_and_template):
    field, template, _ = field_and_template
    unweighted = matcher.zncc_surface(template, field)
    weighted = matcher.zncc_surface(template, field, weight=contract_weights(template.shape))
    assert float(np.abs(weighted - unweighted).max()) > 1e-3


def test_weighted_surface_still_finds_the_truth(field_and_template):
    """Re-weighting must not move a genuinely unique match."""
    field, template, truth = field_and_template
    weighted = matcher.zncc_surface(template, field, weight=contract_weights(template.shape))
    assert argmax_colrow(weighted) == truth
    assert weighted.max() == pytest.approx(1.0, abs=1e-3)


def test_weighted_surface_stays_in_unit_range(field_and_template):
    field, template, _ = field_and_template
    weighted = matcher.zncc_surface(template, field, weight=contract_weights(template.shape))
    assert weighted.min() >= -1.0
    assert weighted.max() <= 1.0
    assert np.all(np.isfinite(weighted))


def test_weighting_can_suppress_a_distractor():
    """The point of the weighting, demonstrated on a constructed case.

    A template whose left half is unique and right half is a repeating stripe.
    Down-weighting the periodic half must not damage the match at the true
    position, because the unique half still carries it.
    """
    field = make_periodic_field()
    template = crop(field, 250, 120, TEMPLATE_SIZE)

    weights = np.full_like(template, 0.05)
    weights[:, : TEMPLATE_SIZE // 2] = 1.0

    weighted = matcher.zncc_surface(template, field, weight=weights)
    assert argmax_colrow(weighted) == (250, 120)


def test_zero_weights_outside_a_patch_use_only_that_patch(field_and_template):
    """Weighting to a sub-window must match correlating that sub-window alone."""
    field, template, truth = field_and_template
    weights = np.zeros_like(template)
    weights[8:40, 8:40] = 1.0

    weighted = matcher.zncc_surface(template, field, weight=weights)
    assert argmax_colrow(weighted) == truth


def test_all_zero_weight_yields_a_zero_surface(field_and_template):
    field, template, _ = field_and_template
    weighted = matcher.zncc_surface(template, field, weight=np.zeros_like(template))
    assert np.all(weighted == 0.0)


def test_negative_weights_are_clipped_not_propagated(field_and_template):
    """A negative weight would make the variance term meaningless."""
    field, template, truth = field_and_template
    weights = contract_weights(template.shape)
    weights[0, 0] = -5.0
    weighted = matcher.zncc_surface(template, field, weight=weights)
    assert np.all(np.isfinite(weighted))
    assert argmax_colrow(weighted) == truth


def test_weighted_path_is_deterministic(field_and_template):
    field, template, _ = field_and_template
    weights = contract_weights(template.shape)
    first = matcher.zncc_surface(template, field, weight=weights)
    second = matcher.zncc_surface(template, field, weight=weights)
    np.testing.assert_array_equal(first, second)


def test_weighted_path_is_invariant_to_intensity_change(field_and_template):
    """The property that justified ZNCC must survive the weighting."""
    field, template, truth = field_and_template
    weights = contract_weights(template.shape)

    baseline = matcher.zncc_surface(template, field, weight=weights)
    altered = matcher.zncc_surface(template * 3.0 - 12.0, field * 2.0 + 40.0, weight=weights)

    assert argmax_colrow(altered) == truth
    np.testing.assert_allclose(altered, baseline, atol=1e-4)


def test_weighted_path_does_not_mutate_its_inputs(field_and_template):
    field, template, _ = field_and_template
    weights = contract_weights(template.shape)
    before = (template.copy(), field.copy(), weights.copy())

    matcher.zncc_surface(template, field, weight=weights)

    np.testing.assert_array_equal(template, before[0])
    np.testing.assert_array_equal(field, before[1])
    np.testing.assert_array_equal(weights, before[2])


def test_constant_template_still_scores_zero(field_and_template):
    field, template, _ = field_and_template
    flat = np.full_like(template, 0.5)
    weighted = matcher.zncc_surface(flat, field, weight=contract_weights(template.shape))
    assert np.all(weighted == 0.0)


# ---------------------------------------------------------------------------
# build_weight — carrying the map onto the template grid
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("theta, scale", [(0.0, 10.0), (0.0, 9.7), (5.0, 10.0), (-8.0, 10.3)])
def test_weight_lands_on_the_template_grid(theta, scale):
    """Alignment is the whole point: shapes must match exactly.

    The weight goes through the same rotation, the same valid-area crop and the
    same decimation as the image, so a shape mismatch means they are no longer
    describing the same pixels.
    """
    reference = make_periodic_field(shape=(1000, 1000))
    weights = np.full_like(reference, 0.5)

    template = matcher.build_template(reference, theta, scale, psf_sigma_px=1.0)
    carried = matcher.build_weight(weights, theta, scale)
    assert carried.shape == template.shape


def test_weight_is_float32():
    reference = make_periodic_field(shape=(500, 500))
    assert matcher.build_weight(np.full_like(reference, 0.5), 0.0, 10.0).dtype == np.float32


def test_weight_is_not_standardised():
    """A non-negative importance profile must stay non-negative.

    Centring it would make roughly half the weights negative, which has no
    meaning in a weighted variance.
    """
    reference = make_periodic_field(shape=(500, 500))
    weights = np.random.default_rng(1).uniform(0.05, 1.0, reference.shape).astype(np.float32)
    carried = matcher.build_weight(weights, 0.0, 10.0)

    assert float(carried.min()) >= 0.0
    assert float(carried.mean()) == pytest.approx(float(weights.mean()), abs=0.02)


def test_weight_is_not_blurred_by_the_psf():
    """No PSF on the weights: an anchor's importance must not bleed outward.

    A sharp block of high weight stays sharp. If the PSF were applied the block
    would spread into the surrounding low-weight region.
    """
    weights = np.full((500, 500), 0.05, dtype=np.float32)
    weights[200:300, 200:300] = 1.0

    carried = matcher.build_weight(weights, 0.0, 10.0)
    high = carried > 0.5
    assert int(high.sum()) == pytest.approx(100, abs=40)
    assert float(carried[0, 0]) == pytest.approx(0.05, abs=1e-3)


def test_weight_decimation_is_an_area_average():
    weights = np.zeros((100, 100), dtype=np.float32)
    weights[:50, :] = 1.0
    carried = matcher.build_weight(weights, 0.0, 10.0)
    assert float(carried[:5, :].mean()) == pytest.approx(1.0, abs=1e-3)
    assert float(carried[5:, :].mean()) == pytest.approx(0.0, abs=1e-3)


def test_weight_rotation_matches_the_template_rotation():
    """A marker in the weight map must land where the same marker in the image does."""
    reference = np.zeros((1000, 1000), dtype=np.float32)
    reference[480:520, 300:340] = 1.0

    weights = np.full_like(reference, 0.05)
    weights[480:520, 300:340] = 1.0

    for theta in (0.0, 6.0, -6.0):
        template = matcher.build_template(reference, theta, 10.0, psf_sigma_px=0.2)
        carried = matcher.build_weight(weights, theta, 10.0)
        assert np.unravel_index(int(template.argmax()), template.shape) == np.unravel_index(
            int(carried.argmax()), carried.shape
        )


def test_build_weight_rejects_non_positive_scale():
    with pytest.raises(ValueError, match="scale must be strictly positive"):
        matcher.build_weight(np.ones((100, 100), dtype=np.float32), 0.0, 0.0)


def test_build_weight_is_deterministic():
    weights = np.random.default_rng(2).uniform(0.05, 1.0, (500, 500)).astype(np.float32)
    first = matcher.build_weight(weights, 4.0, 10.0)
    second = matcher.build_weight(weights, 4.0, 10.0)
    np.testing.assert_array_equal(first, second)


# ---------------------------------------------------------------------------
# Integration through the stage cache
# ---------------------------------------------------------------------------


def test_weighted_and_unweighted_are_cached_separately(two_scale_pair):
    reference, search, _ = two_scale_pair
    cache = _StageCache(
        np.ascontiguousarray(search, dtype=np.float32),
        np.ascontiguousarray(reference, dtype=np.float32),
    )
    plain = cache.correlate(0.0, config.NOMINAL_SCALE, 1.0, weighted=False)
    weighted = cache.correlate(0.0, config.NOMINAL_SCALE, 1.0, weighted=True)
    assert plain[1] is not weighted[1]


def test_uniqueness_map_is_computed_once(two_scale_pair, monkeypatch):
    from src import disambiguate

    reference, search, _ = two_scale_pair
    cache = _StageCache(
        np.ascontiguousarray(search, dtype=np.float32),
        np.ascontiguousarray(reference, dtype=np.float32),
    )
    calls = {"n": 0}
    real = disambiguate.uniqueness_map

    def counted(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(disambiguate, "uniqueness_map", counted)

    cache.correlate(0.0, config.NOMINAL_SCALE, 1.0, weighted=True)
    cache.correlate(0.0, config.NOMINAL_SCALE, 2.0, weighted=True)
    assert calls["n"] == 1


def test_template_weight_is_memoised_per_pose(two_scale_pair):
    reference, search, _ = two_scale_pair
    cache = _StageCache(
        np.ascontiguousarray(search, dtype=np.float32),
        np.ascontiguousarray(reference, dtype=np.float32),
    )
    assert cache.template_weight(0.0, 10.0) is cache.template_weight(0.0, 10.0)
    assert cache.template_weight(0.0, 10.0) is not cache.template_weight(3.0, 10.0)


def test_localize_reports_the_uniqueness_score(two_scale_pair):
    from src.localize import localize

    reference, search, _ = two_scale_pair
    robust = localize(search, reference, mode="robust")
    assert np.isfinite(robust.diagnostics.uniqueness_score)
    assert robust.diagnostics.uniqueness_score > 0.0


def test_fast_tier_stays_unweighted(two_scale_pair):
    """Weighting is the payload of escalation, not the default."""
    from src.localize import localize

    reference, search, _ = two_scale_pair
    assert localize(search, reference, mode="fast").diagnostics.uniqueness_score == 0.0


def test_weighting_engages_once_the_map_stops_being_constant(two_scale_pair, monkeypatch):
    """The handover test: nothing here changes when R3 lands a real map.

    Simulates a non-constant uniqueness map and checks that the weighted path
    actually engages, rather than the constant-map short-circuit swallowing it.
    Without this, the skip added for speed could silently disable the whole
    feature the moment it became useful.
    """
    from src import disambiguate
    from src.localize import localize

    reference, search, _ = two_scale_pair
    baseline = localize(search, reference, mode="robust")

    def informative(reference_image, tile=config.DEFAULT_UNIQUENESS_TILE_PX):
        weights = np.full(reference_image.shape, 0.05, dtype=np.float32)
        weights[:, : reference_image.shape[1] // 3] = 1.0
        return weights

    monkeypatch.setattr(disambiguate, "uniqueness_map", informative)
    weighted = localize(search, reference, mode="robust")

    assert weighted.diagnostics.ncc_peak != baseline.diagnostics.ncc_peak
    assert weighted.diagnostics.uniqueness_score < baseline.diagnostics.uniqueness_score
