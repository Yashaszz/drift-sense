"""Stage 3 (T2) — behaviour tests for the ZNCC correlation surface.

The T0 contract tests already lock shapes and error behaviour. These assert what
the implementation must actually *do*: find the right place, and keep finding it
when the two captures differ in the ways two real captures differ.
"""

import numpy as np
import pytest

from src import matcher
from src.config import window_topleft_to_centre
from tests.conftest import TEMPLATE_SIZE, crop, make_periodic_field, rescale_to_dtype


def argmax_colrow(surface: np.ndarray) -> tuple[int, int]:
    """Return the surface argmax as ``(col, row)``."""
    row, col = np.unravel_index(int(np.argmax(surface)), surface.shape)
    return int(col), int(row)


# ---------------------------------------------------------------------------
# Core behaviour
# ---------------------------------------------------------------------------


def test_finds_exact_position_of_an_extracted_patch(search_image, template_and_truth):
    template, true_col, true_row = template_and_truth
    surface = matcher.zncc_surface(template, search_image)
    assert argmax_colrow(surface) == (true_col, true_row)


def test_perfect_match_scores_one(search_image, template_and_truth):
    template, true_col, true_row = template_and_truth
    surface = matcher.zncc_surface(template, search_image)
    assert surface[true_row, true_col] == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize(
    "col, row",
    [(0, 0), (5, 300), (300, 5), (336, 336), (137, 201)],
)
def test_finds_patches_at_many_positions(search_image, col, row):
    """Includes both corners of the valid range, where off-by-one bugs live."""
    template = crop(search_image, col, row, TEMPLATE_SIZE)
    surface = matcher.zncc_surface(template, search_image)
    assert argmax_colrow(surface) == (col, row)


def test_surface_values_stay_within_unit_range(search_image, template_and_truth):
    template, _, _ = template_and_truth
    surface = matcher.zncc_surface(template, search_image)
    assert surface.min() >= -1.0
    assert surface.max() <= 1.0


def test_surface_is_float32(search_image, template_and_truth):
    template, _, _ = template_and_truth
    assert matcher.zncc_surface(template, search_image).dtype == np.float32


def test_surface_is_finite_everywhere(search_image, template_and_truth):
    template, _, _ = template_and_truth
    assert np.all(np.isfinite(matcher.zncc_surface(template, search_image)))


def test_is_deterministic(search_image, template_and_truth):
    template, _, _ = template_and_truth
    first = matcher.zncc_surface(template, search_image)
    second = matcher.zncc_surface(template, search_image)
    np.testing.assert_array_equal(first, second)


# ---------------------------------------------------------------------------
# Invariance — the property that justifies choosing ZNCC over SSD
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "gain, offset",
    [(1.0, 0.0), (2.0, 0.0), (0.5, 0.0), (1.0, 50.0), (3.0, -20.0), (0.25, 100.0)],
)
def test_invariant_to_affine_intensity_change(search_image, template_and_truth, gain, offset):
    """The reference and search are separate captures with different gain/offset.

    ZNCC must be invariant to ``I -> a*I + b``. If this fails the similarity
    measure is wrong, not merely inaccurate: SSD fails exactly here, which is
    why it was rejected.
    """
    template, true_col, true_row = template_and_truth
    baseline = matcher.zncc_surface(template, search_image)
    altered = matcher.zncc_surface(template, search_image * gain + offset)

    assert argmax_colrow(altered) == (true_col, true_row)
    np.testing.assert_allclose(altered, baseline, atol=1e-4)


def test_invariant_when_the_template_is_rescaled(search_image, template_and_truth):
    """Gain/offset may differ on the template side too, not only the search side."""
    template, true_col, true_row = template_and_truth
    surface = matcher.zncc_surface(template * 4.0 - 7.0, search_image)
    assert argmax_colrow(surface) == (true_col, true_row)
    assert surface[true_row, true_col] == pytest.approx(1.0, abs=1e-4)


def test_inverted_template_gives_strong_negative_correlation(search_image, template_and_truth):
    template, true_col, true_row = template_and_truth
    surface = matcher.zncc_surface(-template, search_image)
    assert surface[true_row, true_col] == pytest.approx(-1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("noise", [0.05, 0.2, 0.5])
def test_survives_independent_noise_on_the_search_image(noise):
    """The two captures have independent noise realizations, never a shared one.

    Structure seed is held constant and only the noise seed varies: same wafer,
    two captures.
    """
    template = crop(make_periodic_field(noise=0.0), 250, 120, TEMPLATE_SIZE)
    noisy = make_periodic_field(noise=noise, noise_seed=99)

    surface = matcher.zncc_surface(template, noisy)
    assert argmax_colrow(surface) == (250, 120)


def test_peak_degrades_gracefully_as_noise_rises():
    """Correlation quality should fall monotonically, not collapse discontinuously."""
    template = crop(make_periodic_field(noise=0.0), 250, 120, TEMPLATE_SIZE)

    peaks = [
        float(matcher.zncc_surface(template, make_periodic_field(noise=n, noise_seed=7)).max())
        for n in (0.0, 0.1, 0.3, 0.6)
    ]
    assert peaks == sorted(peaks, reverse=True)


def test_unanchored_field_produces_a_lattice_of_tied_peaks():
    """Documents the core difficulty that Stage 4 exists to resolve.

    With no aperiodic anchor in frame, a template correlates essentially as well
    at every lattice-aligned position. Stage 3's recall is perfect here — the
    true position is in the candidate set — but top-1 selection is impossible
    from correlation evidence alone. That distinction is exactly why R4 is
    measured on recall@K and R3 on top-1-given-recall.
    """
    bare = make_periodic_field(anchored=False)
    template = crop(bare, 250, 120, TEMPLATE_SIZE)
    surface = matcher.zncc_surface(template, bare)

    assert int(np.sum(surface > surface.max() - 0.15)) > 1000
    assert surface[120, 250] == pytest.approx(surface.max(), abs=1e-4)


def test_anchored_field_resolves_the_ambiguity():
    """The same template in an anchored field has one clear winner.

    Paired with the test above, this is the evidence that aperiodic content is
    what makes the answer unique — the argument the failure analysis rests on.
    """
    anchored = make_periodic_field(anchored=True)
    template = crop(anchored, 250, 120, TEMPLATE_SIZE)
    surface = matcher.zncc_surface(template, anchored)

    assert int(np.sum(surface > surface.max() - 0.15)) < 100
    assert argmax_colrow(surface) == (250, 120)


# ---------------------------------------------------------------------------
# Degenerate inputs
# ---------------------------------------------------------------------------


def test_constant_template_scores_zero_not_one(search_image):
    """OpenCV returns 1.0 everywhere for a zero-variance template.

    That is the most dangerous possible failure: wrong *and* maximally
    confident. A constant patch carries no information about where it sits, so
    the honest surface is all zeros.
    """
    template = np.full((TEMPLATE_SIZE, TEMPLATE_SIZE), 0.5, dtype=np.float32)
    surface = matcher.zncc_surface(template, search_image)
    assert np.all(surface == 0.0)


@pytest.mark.parametrize("fill", [0.0, 1.0, -3.5, 255.0])
def test_constant_template_of_any_level_scores_zero(search_image, fill):
    template = np.full((TEMPLATE_SIZE, TEMPLATE_SIZE), fill, dtype=np.float32)
    assert np.all(matcher.zncc_surface(template, search_image) == 0.0)


def test_constant_search_image_yields_finite_surface(template_and_truth):
    template, _, _ = template_and_truth
    search = np.full((400, 400), 7.0, dtype=np.float32)
    surface = matcher.zncc_surface(template, search)
    assert np.all(np.isfinite(surface))
    assert np.all(np.abs(surface) <= 1.0)


def test_both_constant_yields_zero_surface():
    template = np.zeros((10, 10), dtype=np.float32)
    search = np.zeros((50, 50), dtype=np.float32)
    assert np.all(matcher.zncc_surface(template, search) == 0.0)


def test_template_equal_to_search_gives_single_perfect_score(search_image):
    surface = matcher.zncc_surface(search_image, search_image)
    assert surface.shape == (1, 1)
    assert surface[0, 0] == pytest.approx(1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# Input coercion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.float32, np.float64])
def test_accepts_any_real_dtype(dtype):
    """cv2.matchTemplate itself rejects float64; coercion happens in our layer."""
    scaled = rescale_to_dtype(make_periodic_field(shape=(200, 200)), dtype)
    template = np.ascontiguousarray(scaled[60:124, 90:154])

    surface = matcher.zncc_surface(template, scaled)
    assert argmax_colrow(surface) == (90, 60)


def test_dtypes_agree_with_each_other():
    base = make_periodic_field(shape=(200, 200))
    template_slice = (slice(60, 124), slice(90, 154))

    surfaces = [
        matcher.zncc_surface(
            np.ascontiguousarray(rescale_to_dtype(base, dtype)[template_slice]),
            rescale_to_dtype(base, dtype),
        )
        for dtype in (np.float32, np.float64)
    ]
    np.testing.assert_allclose(surfaces[0], surfaces[1], atol=1e-5)


def test_accepts_non_contiguous_input(search_image, template_and_truth):
    template, true_col, true_row = template_and_truth
    surface = matcher.zncc_surface(template, np.asfortranarray(search_image))
    assert argmax_colrow(surface) == (true_col, true_row)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_rejects_non_finite_search(template_and_truth, bad):
    template, _, _ = template_and_truth
    search = make_periodic_field()
    search[10, 10] = bad
    with pytest.raises(ValueError, match="non-finite"):
        matcher.zncc_surface(template, search)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_rejects_non_finite_template(search_image, template_and_truth, bad):
    template, _, _ = template_and_truth
    template = template.copy()
    template[0, 0] = bad
    with pytest.raises(ValueError, match="non-finite"):
        matcher.zncc_surface(template, search_image)


def test_rejects_non_two_dimensional_input(search_image):
    with pytest.raises(ValueError, match="two-dimensional"):
        matcher.zncc_surface(np.zeros((4, 4, 3), dtype=np.float32), search_image)


def test_does_not_mutate_its_inputs(search_image, template_and_truth):
    template, _, _ = template_and_truth
    template_before = template.copy()
    search_before = search_image.copy()

    matcher.zncc_surface(template, search_image)

    np.testing.assert_array_equal(template, template_before)
    np.testing.assert_array_equal(search_image, search_before)


# ---------------------------------------------------------------------------
# Masked path — still deferred to T8
# ---------------------------------------------------------------------------


def test_weight_no_longer_warns(search_image, template_and_truth):
    """The masked path is implemented; passing a weight must be silent."""
    import warnings

    template, _, _ = template_and_truth
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        matcher.zncc_surface(template, search_image, weight=np.ones_like(template))


def test_constant_weight_reproduces_the_unweighted_surface(search_image, template_and_truth):
    """The contract between the two paths, asserted directly.

    Normalising the weights by their sum makes every weighted mean collapse to
    the ordinary mean, so a constant map must reproduce plain ZNCC. This is what
    makes the weighted path checkable while ``uniqueness_map`` still returns a
    constant.
    """
    template, _, _ = template_and_truth
    unweighted = matcher.zncc_surface(template, search_image)
    weighted = matcher.zncc_surface(template, search_image, weight=np.ones_like(template))
    np.testing.assert_allclose(weighted, unweighted, atol=1e-5)


# ---------------------------------------------------------------------------
# Integration with the coordinate convention
# ---------------------------------------------------------------------------


def test_peak_converts_to_the_expected_search_centre(search_image, template_and_truth):
    """End-to-end check of the corner-to-centre conversion against known truth."""
    template, true_col, true_row = template_and_truth
    surface = matcher.zncc_surface(template, search_image)
    col, row = argmax_colrow(surface)

    centre = window_topleft_to_centre(col, row, template.shape)
    expected = (
        true_col + (TEMPLATE_SIZE - 1) / 2,
        true_row + (TEMPLATE_SIZE - 1) / 2,
    )
    assert centre == expected
