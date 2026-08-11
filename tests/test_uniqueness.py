"""Contract and behaviour tests for Stage 4a uniqueness weighting.

Fixtures here are synthetic sine lattices, materially softer than R1's
hard-edged renders. Nothing measured here may be quoted in failure_analysis.md.
"""

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from src.uniqueness import uniqueness_map, uniqueness_score


def _periodic(shape=(512, 512), pitch=48):
    rows, cols = shape
    rr, cc = np.ogrid[:rows, :cols]
    field = np.sin(2 * np.pi * cc / pitch) + np.sin(2 * np.pi * rr / pitch)
    return field.astype(np.float32)


def _periodic_with_anchor(shape=(512, 512), pitch=48, anchor=(352, 352), size=96):
    """Periodic field with an aperiodic textured patch.

    The anchor must carry *structure that does not repeat*, not merely a
    constant block. A flat block is an absence of features, and a matched
    filter cannot localise on it either — so it is not a valid stand-in for
    the aperiodic marks R1's layouts place.
    """
    field = _periodic(shape, pitch)
    rng = np.random.default_rng(11)
    r0, c0 = anchor
    patch = rng.normal(0.0, 1.0, (size, size)).astype(np.float32)
    field[r0 : r0 + size, c0 : c0 + size] = gaussian_filter(patch, sigma=3.0) * 12.0
    return field


# --- Contract: R4 codes against these ---------------------------------------


@pytest.mark.parametrize("shape", [(512, 512), (300, 700), (64, 64), (10, 10)])
def test_shape_and_dtype_match_reference(shape):
    weights = uniqueness_map(_periodic(shape, pitch=24))
    assert weights.shape == shape
    assert weights.dtype == np.float32


def test_values_bounded_and_finite():
    weights = uniqueness_map(_periodic_with_anchor())
    assert np.all(np.isfinite(weights))
    assert weights.min() > 0.0
    assert weights.max() <= 1.0


def test_never_degenerate():
    for field in (np.zeros((256, 256), np.float32), np.ones((256, 256), np.float32)):
        weights = uniqueness_map(field)
        assert np.all(np.isfinite(weights))
        assert weights.sum() > 0.0


def test_deterministic():
    field = _periodic_with_anchor()
    assert np.array_equal(uniqueness_map(field), uniqueness_map(field))


def test_tiny_reference_returns_uniform():
    weights = uniqueness_map(np.random.default_rng(0).random((12, 12)).astype(np.float32))
    assert np.allclose(weights, 1.0)


# --- Behaviour --------------------------------------------------------------


def test_anchor_scores_above_periodic_background():
    weights = uniqueness_map(_periodic_with_anchor(anchor=(352, 352), size=96))
    assert weights[352:448, 352:448].mean() > weights[:200, :200].mean()


def test_pure_periodic_map_is_near_flat():
    """No anchor in frame means no region is more informative. Do not invent one."""
    weights = uniqueness_map(_periodic())
    assert float(weights.max() - weights.min()) < 0.25


def test_score_separates_anchored_from_unanchored():
    """The validation target. If this fails, the map is not working."""
    unanchored = uniqueness_score(uniqueness_map(_periodic()))
    anchored = uniqueness_score(uniqueness_map(_periodic_with_anchor()))
    assert anchored > unanchored


def test_score_is_nan_on_empty_map():
    assert np.isnan(uniqueness_score(np.array([], dtype=np.float32)))


def test_score_is_nan_on_all_nan_map():
    assert np.isnan(uniqueness_score(np.full((16, 16), np.nan, dtype=np.float32)))


def test_prefilter_makes_the_map_track_structure_not_noise():
    """Without the low-pass, sensor noise dominates every tile\'s autocorrelation.

    Noise is broadband, so it drives all sidelobes down uniformly and every
    tile scores alike — the map stops discriminating. The prefilter, set near
    the 10x decimation Nyquist, removes the frequencies the matcher cannot see
    anyway and lets real structure drive the score. Test the discrimination,
    not the level: the anchored/unanchored gap must survive the noise.
    """
    rng = np.random.default_rng(7)
    noise = rng.normal(0.0, 0.5, (512, 512)).astype(np.float32)
    plain = _periodic() + noise
    anchored = _periodic_with_anchor() + noise

    filtered_gap = uniqueness_score(uniqueness_map(anchored)) - uniqueness_score(
        uniqueness_map(plain)
    )
    unfiltered_gap = uniqueness_score(
        uniqueness_map(anchored, prefilter_sigma_px=0.0)
    ) - uniqueness_score(uniqueness_map(plain, prefilter_sigma_px=0.0))

    assert filtered_gap > unfiltered_gap
