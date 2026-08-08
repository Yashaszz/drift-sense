"""Stage 2 (T4) — behaviour tests for template construction.

Covers the four operations individually and then the composed chain against a
two-scale pair with exact ground truth. The individual tests catch the mistakes
that are invisible end-to-end — a sign error in rotation, a scale factor applied
on the wrong grid — because those show up as a slightly worse peak rather than
an obvious failure.
"""

import cv2
import numpy as np
import pytest

from src import config, matcher
from tests.conftest import (
    PAIR_SCALE,
    make_periodic_field,
    make_two_scale_pair,
)


def argmax_colrow(surface: np.ndarray) -> tuple[int, int]:
    """Return the surface argmax as ``(col, row)``."""
    row, col = np.unravel_index(int(np.argmax(surface)), surface.shape)
    return int(col), int(row)


# ---------------------------------------------------------------------------
# area_average_downsample
# ---------------------------------------------------------------------------


def test_constant_image_survives_downsampling_unchanged():
    """The regression test for ringing.

    An interpolating kernel with negative lobes undershoots at edges — the
    measured 121 against a 128 background that motivated area-averaging. A box
    average cannot: the mean of equal values is that value.
    """
    image = np.full((1000, 1000), 128.0, dtype=np.float32)
    out = matcher.area_average_downsample(image, 10.0)
    assert np.allclose(out, 128.0, atol=1e-4)


def test_downsampling_computes_the_block_mean():
    image = np.arange(64, dtype=np.float32).reshape(8, 8)
    out = matcher.area_average_downsample(image, 2.0)
    expected = image.reshape(4, 2, 4, 2).mean(axis=(1, 3))
    np.testing.assert_allclose(out, expected, atol=1e-4)


def test_downsampling_does_not_overshoot_a_step_edge():
    """A ringing kernel would push values outside the input's range."""
    image = np.zeros((200, 200), dtype=np.float32)
    image[:, 100:] = 100.0
    out = matcher.area_average_downsample(image, 10.0)
    assert out.min() >= -1e-4
    assert out.max() <= 100.0 + 1e-4


@pytest.mark.parametrize("factor", [2.0, 5.0, 10.0, 9.7, 10.3])
def test_downsampling_shape(factor):
    image = np.zeros((1000, 1000), dtype=np.float32)
    out = matcher.area_average_downsample(image, factor)
    assert out.shape == (round(1000 / factor), round(1000 / factor))


def test_downsampling_by_one_is_a_no_op():
    image = make_periodic_field(shape=(64, 64))
    np.testing.assert_array_equal(matcher.area_average_downsample(image, 1.0), image)


def test_downsampling_preserves_the_mean():
    image = make_periodic_field(shape=(500, 500))
    out = matcher.area_average_downsample(image, 10.0)
    assert float(out.mean()) == pytest.approx(float(image.mean()), abs=1e-3)


def test_downsampling_rejects_non_positive_factor():
    with pytest.raises(ValueError, match="strictly positive"):
        matcher.area_average_downsample(np.zeros((10, 10), dtype=np.float32), 0.0)


# ---------------------------------------------------------------------------
# match_psf
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sigma", [0.0, -1.0])
def test_non_positive_sigma_leaves_the_image_unchanged(sigma):
    image = make_periodic_field(shape=(64, 64))
    np.testing.assert_array_equal(matcher.match_psf(image, sigma), image)


def test_blur_preserves_shape_and_dtype():
    image = make_periodic_field(shape=(64, 48))
    out = matcher.match_psf(image, 2.0)
    assert out.shape == image.shape
    assert out.dtype == np.float32


def test_blur_leaves_a_constant_image_constant():
    """Catches zero-padded borders, which would darken the rim.

    A rim artefact sits exactly where the correlation is most sensitive to it,
    so border handling has to replicate rather than pad.
    """
    image = np.full((128, 128), 50.0, dtype=np.float32)
    assert np.allclose(matcher.match_psf(image, 3.0), 50.0, atol=1e-3)


def test_blur_reduces_high_frequency_energy_monotonically():
    image = make_periodic_field(shape=(256, 256))
    energies = [float(np.var(np.diff(matcher.match_psf(image, s), axis=1))) for s in (0.5, 1, 2, 4)]
    assert energies == sorted(energies, reverse=True)


def test_blur_width_matches_the_requested_sigma():
    """Blur an impulse and recover sigma from the second moment of the result."""
    image = np.zeros((129, 129), dtype=np.float32)
    image[64, 64] = 1.0
    sigma = 3.0

    blurred = matcher.match_psf(image, sigma)
    coords = np.arange(129) - 64
    profile = blurred.sum(axis=0)
    recovered = float(np.sqrt((profile * coords**2).sum() / profile.sum()))

    assert recovered == pytest.approx(sigma, rel=0.05)


def test_blur_preserves_total_intensity():
    image = make_periodic_field(shape=(128, 128))
    assert float(matcher.match_psf(image, 2.0).mean()) == pytest.approx(
        float(image.mean()), abs=1e-3
    )


# ---------------------------------------------------------------------------
# rotate_image
# ---------------------------------------------------------------------------


def test_zero_rotation_is_the_identity():
    image = make_periodic_field(shape=(64, 64))
    np.testing.assert_array_equal(matcher.rotate_image(image, 0.0), image)


def test_rotation_preserves_shape():
    image = make_periodic_field(shape=(80, 60))
    assert matcher.rotate_image(image, 17.5).shape == (80, 60)


def test_positive_theta_is_counter_clockwise():
    """Sign convention, checked against numpy rather than assumed.

    PoseEstimate documents positive theta as counter-clockwise. A sign error
    here is invisible on symmetric layouts and catastrophic on asymmetric ones.
    """
    image = np.zeros((101, 101), dtype=np.float32)
    image[10:20, 45:56] = 1.0  # a bar above centre

    rotated = matcher.rotate_image(image, 90.0)
    expected = np.rot90(image)

    assert float(np.corrcoef(rotated.ravel(), expected.ravel())[0, 1]) > 0.95


def test_rotation_fill_is_the_image_mean_not_zero():
    """Zero fill would act as a synthetic dark edge for the matcher to latch on."""
    image = np.full((101, 101), 200.0, dtype=np.float32)
    image[40:60, 40:60] = 100.0

    rotated = matcher.rotate_image(image, 30.0)
    assert float(rotated[0, 0]) == pytest.approx(float(image.mean()), abs=1.0)


def test_rotation_is_about_the_centre():
    """The centre pixel is the fixed point of the rotation."""
    image = make_periodic_field(shape=(201, 201))
    rotated = matcher.rotate_image(image, 40.0)
    assert float(rotated[100, 100]) == pytest.approx(float(image[100, 100]), abs=0.1)


def test_rotating_back_recovers_the_interior():
    image = make_periodic_field(shape=(201, 201))
    round_trip = matcher.rotate_image(matcher.rotate_image(image, 12.0), -12.0)

    interior = (slice(60, 140), slice(60, 140))
    correlation = np.corrcoef(round_trip[interior].ravel(), image[interior].ravel())[0, 1]
    assert float(correlation) > 0.98


# ---------------------------------------------------------------------------
# _crop_to_valid_rotation
# ---------------------------------------------------------------------------


def test_crop_is_a_no_op_at_zero_rotation():
    image = make_periodic_field(shape=(64, 64))
    assert matcher._crop_to_valid_rotation(image, 0.0) is image


@pytest.mark.parametrize("theta", [1.0, 5.0, 15.0, 30.0])
def test_crop_removes_every_filled_pixel(theta):
    """Rotate a marker image with zero fill; the crop must contain no zeros."""
    ones = np.ones((401, 401), dtype=np.float32)
    matrix = cv2.getRotationMatrix2D((200.0, 200.0), theta, 1.0)
    rotated = cv2.warpAffine(
        ones, matrix, (401, 401), borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
    )

    cropped = matcher._crop_to_valid_rotation(rotated, theta)
    assert float(cropped.min()) > 0.99


@pytest.mark.parametrize("theta", [3.0, 10.0])
def test_crop_stays_centred(theta):
    """An asymmetric crop would shift the reported centre systematically."""
    image = make_periodic_field(shape=(401, 401))
    cropped = matcher._crop_to_valid_rotation(image, theta)

    removed_rows = image.shape[0] - cropped.shape[0]
    removed_cols = image.shape[1] - cropped.shape[1]
    assert removed_rows % 2 == 0
    assert removed_cols % 2 == 0


def test_larger_rotation_crops_more():
    image = make_periodic_field(shape=(401, 401))
    sizes = [matcher._crop_to_valid_rotation(image, t).shape[0] for t in (1.0, 5.0, 15.0, 30.0)]
    assert sizes == sorted(sizes, reverse=True)


# ---------------------------------------------------------------------------
# _reference_grid_sigma
# ---------------------------------------------------------------------------


def test_sigma_converts_from_search_grid_to_reference_grid():
    """A 1 px blur in the search image is roughly a 10 px blur at 1 nm/px."""
    sigma = matcher._reference_grid_sigma(1.0, 10.0)
    assert 9.0 < sigma < 10.0


def test_sigma_discounts_the_blur_decimation_supplies():
    """Averaging over a box of width s contributes variance s^2/12 for free."""
    naive = 1.0 * 10.0
    actual = matcher._reference_grid_sigma(1.0, 10.0)
    assert actual == pytest.approx(np.sqrt(naive**2 - 100.0 / 12.0), rel=1e-6)


def test_sigma_is_zero_when_decimation_already_over_blurs():
    assert matcher._reference_grid_sigma(0.1, 10.0) == 0.0


def test_sigma_is_monotone_in_the_target():
    widths = [matcher._reference_grid_sigma(t, 10.0) for t in (0.5, 1.0, 2.0, 4.0)]
    assert widths == sorted(widths)


# ---------------------------------------------------------------------------
# estimate_psf_sigma
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sigma", [1.0, 1.5, 2.0])
def test_estimates_a_known_blur_on_broadband_content(sigma):
    base = np.random.default_rng(0).normal(0, 1, (512, 512)).astype(np.float32)
    assert matcher.estimate_psf_sigma(matcher.match_psf(base, sigma)) == pytest.approx(
        sigma, abs=0.15
    )


def test_declines_to_estimate_on_a_lattice_dominated_spectrum():
    """A periodic layout's spectrum is a comb, not a rolloff.

    Fitting a line through it yields a confident, meaningless number. Measured
    fit quality separates the two cases cleanly, so the estimator refuses and
    returns the documented default instead of guessing.
    """
    field = make_periodic_field(shape=(512, 512))
    for applied in (0.0, 1.0, 2.0):
        blurred = matcher.match_psf(field, applied) if applied else field
        assert matcher.estimate_psf_sigma(blurred) == config.DEFAULT_PSF_SIGMA_PX


def test_estimate_is_always_positive_and_finite():
    for image in (
        np.zeros((128, 128), dtype=np.float32),
        np.full((128, 128), 7.0, dtype=np.float32),
        np.random.default_rng(1).normal(0, 1, (128, 128)).astype(np.float32),
    ):
        sigma = matcher.estimate_psf_sigma(image)
        assert np.isfinite(sigma)
        assert sigma > 0.0


def test_estimate_snaps_to_supplied_candidates():
    base = np.random.default_rng(2).normal(0, 1, (512, 512)).astype(np.float32)
    blurred = matcher.match_psf(base, 1.4)
    assert matcher.estimate_psf_sigma(blurred, candidates=(0.5, 1.5, 3.0)) == 1.5


def test_estimate_never_raises_on_degenerate_input():
    for image in (np.zeros((4, 4), dtype=np.float32), np.zeros((1, 1), dtype=np.float32)):
        assert matcher.estimate_psf_sigma(image) > 0.0


# ---------------------------------------------------------------------------
# build_template
# ---------------------------------------------------------------------------


def test_template_is_nominal_size_at_nominal_scale():
    reference = make_periodic_field(shape=(1000, 1000))
    template = matcher.build_template(reference, theta=0.0, scale=10.0, psf_sigma_px=1.0)
    assert template.shape == (config.TEMPLATE_NOMINAL_PX, config.TEMPLATE_NOMINAL_PX)


@pytest.mark.parametrize("scale", [9.7, 10.0, 10.3])
def test_template_size_tracks_the_scale_residual(scale):
    reference = make_periodic_field(shape=(1000, 1000))
    template = matcher.build_template(reference, theta=0.0, scale=scale, psf_sigma_px=1.0)
    assert template.shape == (round(1000 / scale), round(1000 / scale))


def test_template_shrinks_with_rotation():
    """The rotation crop trades size for containing only real data."""
    reference = make_periodic_field(shape=(1000, 1000))
    sizes = [
        matcher.build_template(reference, theta=t, scale=10.0, psf_sigma_px=1.0).shape[0]
        for t in (0.0, 2.0, 5.0, 10.0)
    ]
    assert sizes == sorted(sizes, reverse=True)


def test_template_is_float32():
    reference = make_periodic_field(shape=(500, 500))
    assert (
        matcher.build_template(reference, theta=0.0, scale=10.0, psf_sigma_px=1.0).dtype
        == np.float32
    )


def test_template_preserves_the_reference_centre():
    """No step may translate the content, or the reported match centre shifts.

    A bright marker at the reference centre must land at the template centre.
    """
    reference = np.zeros((1000, 1000), dtype=np.float32)
    reference[480:520, 480:520] = 1.0

    template = matcher.build_template(reference, theta=0.0, scale=10.0, psf_sigma_px=0.5)
    row, col = np.unravel_index(int(np.argmax(template)), template.shape)

    centre = (template.shape[0] - 1) / 2
    assert abs(row - centre) <= 1
    assert abs(col - centre) <= 1


def test_template_centre_is_preserved_under_rotation():
    reference = np.zeros((1000, 1000), dtype=np.float32)
    reference[480:520, 480:520] = 1.0

    template = matcher.build_template(reference, theta=8.0, scale=10.0, psf_sigma_px=0.5)
    row, col = np.unravel_index(int(np.argmax(template)), template.shape)

    centre = (template.shape[0] - 1) / 2
    assert abs(row - centre) <= 1
    assert abs(col - centre) <= 1


def test_template_rejects_non_positive_scale():
    reference = make_periodic_field(shape=(100, 100))
    with pytest.raises(ValueError, match="scale must be strictly positive"):
        matcher.build_template(reference, theta=0.0, scale=0.0, psf_sigma_px=1.0)


def test_template_rejects_non_finite_reference():
    reference = make_periodic_field(shape=(100, 100)).copy()
    reference[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        matcher.build_template(reference, theta=0.0, scale=10.0, psf_sigma_px=1.0)


def test_template_does_not_mutate_the_reference():
    reference = make_periodic_field(shape=(500, 500))
    before = reference.copy()
    matcher.build_template(reference, theta=3.0, scale=10.0, psf_sigma_px=1.0)
    np.testing.assert_array_equal(reference, before)


# ---------------------------------------------------------------------------
# Stage 2 + Stage 3 against a real two-scale pair
# ---------------------------------------------------------------------------


def test_locates_the_reference_in_a_two_scale_pair(two_scale_pair):
    """The payoff test: one scene, two magnifications, exact ground truth."""
    reference, search, (true_col, true_row) = two_scale_pair

    template = matcher.build_template(reference, theta=0.0, scale=PAIR_SCALE, psf_sigma_px=1.0)
    surface = matcher.zncc_surface(template, search)

    assert argmax_colrow(surface) == (true_col, true_row)
    assert float(surface.max()) > 0.8


@pytest.mark.parametrize("noise", [0.05, 0.15])
def test_two_scale_localization_survives_noise(noise):
    reference, search, truth = make_two_scale_pair(noise=noise)
    template = matcher.build_template(reference, theta=0.0, scale=PAIR_SCALE, psf_sigma_px=1.0)
    assert argmax_colrow(matcher.zncc_surface(template, search)) == truth


def test_psf_matching_improves_the_correlation_peak(two_scale_pair):
    """The whole justification for Stage 2's blur step, measured.

    Decimating without matching the search image's blur injects high-frequency
    content that does not exist there, and the peak drops.
    """
    reference, search, _ = two_scale_pair

    unmatched = matcher.build_template(reference, theta=0.0, scale=PAIR_SCALE, psf_sigma_px=0.0)
    matched = matcher.build_template(reference, theta=0.0, scale=PAIR_SCALE, psf_sigma_px=1.0)

    peak_unmatched = float(matcher.zncc_surface(unmatched, search).max())
    peak_matched = float(matcher.zncc_surface(matched, search).max())

    assert peak_matched > peak_unmatched


def test_wrong_scale_degrades_the_peak(two_scale_pair):
    """Confirms the scale factor is genuinely being applied, not ignored."""
    reference, search, _ = two_scale_pair

    correct = float(
        matcher.zncc_surface(
            matcher.build_template(reference, theta=0.0, scale=PAIR_SCALE, psf_sigma_px=1.0),
            search,
        ).max()
    )
    wrong = float(
        matcher.zncc_surface(
            matcher.build_template(reference, theta=0.0, scale=PAIR_SCALE * 1.3, psf_sigma_px=1.0),
            search,
        ).max()
    )
    assert correct > wrong


def test_rotation_is_undone_with_the_matching_sign(two_scale_pair):
    """Closed-loop check of the rotation sign convention.

    Rotating the reference by +theta and asking build_template for -theta must
    recover the original alignment. If the sign were inverted this would fail
    while every symmetric-pattern test still passed.
    """
    reference, search, (true_col, true_row) = two_scale_pair
    theta = 6.0

    tilted = matcher.rotate_image(reference, theta)
    template = matcher.build_template(tilted, theta=-theta, scale=PAIR_SCALE, psf_sigma_px=1.0)
    surface = matcher.zncc_surface(template, search)
    found_col, found_row = argmax_colrow(surface)

    # The crop shrinks the template, so the window's top-left moves inward by
    # half the size difference; compare centres instead, which are invariant.
    found_centre = config.window_topleft_to_centre(found_col, found_row, template.shape)
    reference_template = matcher.build_template(
        reference, theta=0.0, scale=PAIR_SCALE, psf_sigma_px=1.0
    )
    true_centre = config.window_topleft_to_centre(true_col, true_row, reference_template.shape)

    assert abs(found_centre[0] - true_centre[0]) <= 2.0
    assert abs(found_centre[1] - true_centre[1]) <= 2.0
