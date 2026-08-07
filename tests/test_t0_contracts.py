"""T0 contract tests.

These lock the interfaces other roles build against. They deliberately assert
*contracts* — shapes, types, ranges, invariants — and not algorithm behaviour,
because no algorithm is implemented yet. Every assertion here must keep passing
once the real implementations land; if one starts failing in T2-T8, an interface
was broken and the break is the bug.

Numerical accuracy tests arrive with each stage's implementation task.
"""

import json

import numpy as np
import pytest

from src import config, matcher
from src.confidence import FEATURE_NAMES, ConfidenceModel, extract_features, is_low_confidence
from src.localize import localize
from src.types import Diagnostics, LocalizationResult, Peak, PoseEstimate

# ---------------------------------------------------------------------------
# config — coordinate conventions
# ---------------------------------------------------------------------------


def test_image_centre_matches_documented_convention():
    assert config.image_centre((1000, 1000)) == (499.5, 499.5)


def test_image_centre_handles_non_square():
    assert config.image_centre((100, 200)) == (99.5, 49.5)


def test_window_topleft_to_centre_at_origin():
    assert config.window_topleft_to_centre(0, 0, (100, 100)) == (49.5, 49.5)


def test_window_topleft_to_centre_offsets_linearly():
    assert config.window_topleft_to_centre(200, 350, (100, 100)) == (249.5, 399.5)


def test_window_topleft_to_centre_respects_actual_template_shape():
    """The template is not exactly 100x100 once the scale residual is applied."""
    assert config.window_topleft_to_centre(0, 0, (97, 103)) == (51.0, 48.0)


def test_full_window_centre_agrees_with_image_centre():
    """A window filling the whole image must have the image's own centre."""
    shape = (1000, 1000)
    assert config.window_topleft_to_centre(0, 0, shape) == config.image_centre(shape)


def test_nominal_scale_is_consistent_with_pixel_sizes():
    assert config.NOMINAL_SCALE == config.SEARCH_PX_NM / config.REF_PX_NM


@pytest.mark.parametrize("pixels", [0.0, 1.0, 2.5, 100.0])
def test_pixel_nm_round_trip(pixels):
    assert config.nm_to_search_px(config.search_px_to_nm(pixels)) == pytest.approx(pixels)


# ---------------------------------------------------------------------------
# types
# ---------------------------------------------------------------------------


def test_peak_centre_delegates_to_config():
    peak = Peak(col=200, row=350, score=0.9)
    assert peak.centre((100, 100)) == config.window_topleft_to_centre(200, 350, (100, 100))


def test_peak_is_immutable():
    peak = Peak(col=1, row=2, score=0.5)
    with pytest.raises(AttributeError):
        peak.col = 5


def test_pose_estimate_nominal():
    pose = PoseEstimate.nominal()
    assert pose.theta_deg == 0.0
    assert pose.scale == config.NOMINAL_SCALE
    assert pose.quality == 0.0


def test_diagnostics_notes_accumulate():
    diagnostics = Diagnostics()
    assert diagnostics.notes == ()
    diagnostics.with_note("first")
    diagnostics.with_note("second")
    assert diagnostics.notes == ("first", "second")


def test_diagnostics_to_dict_is_json_serialisable():
    diagnostics = Diagnostics()
    diagnostics.with_note("note")
    payload = diagnostics.to_dict()
    assert json.loads(json.dumps(payload))["notes"] == ["note"]


@pytest.mark.parametrize("confidence", [-0.01, 1.01, 2.0])
def test_localization_result_rejects_out_of_range_confidence(confidence):
    with pytest.raises(ValueError, match="confidence"):
        LocalizationResult(x=0.0, y=0.0, confidence=confidence, low_confidence_flag=True)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_localization_result_rejects_non_finite_coordinates(bad):
    with pytest.raises(ValueError, match="finite"):
        LocalizationResult(x=bad, y=0.0, confidence=0.0, low_confidence_flag=True)


def test_localization_result_to_dict_is_json_serialisable():
    result = LocalizationResult(x=1.5, y=2.5, confidence=0.25, low_confidence_flag=False)
    payload = json.loads(json.dumps(result.to_dict()))
    assert payload["x"] == 1.5
    assert payload["diagnostics"]["mode_used"] == "fast"


def test_localization_result_centre_property():
    result = LocalizationResult(x=3.0, y=4.0, confidence=0.0, low_confidence_flag=True)
    assert result.centre == (3.0, 4.0)


# ---------------------------------------------------------------------------
# matcher — shape contracts
# ---------------------------------------------------------------------------


def test_zncc_surface_shape():
    template = np.zeros((100, 100), dtype=np.float32)
    search = np.zeros((1000, 1000), dtype=np.float32)
    assert matcher.zncc_surface(template, search).shape == (901, 901)


def test_zncc_surface_shape_matches_helper():
    template = np.zeros((97, 103), dtype=np.float32)
    search = np.zeros((500, 400), dtype=np.float32)
    expected = matcher.surface_shape(template.shape, search.shape)
    assert matcher.zncc_surface(template, search).shape == expected


def test_zncc_surface_rejects_oversized_template():
    template = np.zeros((200, 200), dtype=np.float32)
    search = np.zeros((100, 100), dtype=np.float32)
    with pytest.raises(ValueError, match="exceeds"):
        matcher.zncc_surface(template, search)


def test_zncc_surface_rejects_mismatched_weight():
    template = np.zeros((10, 10), dtype=np.float32)
    search = np.zeros((100, 100), dtype=np.float32)
    weight = np.ones((5, 5), dtype=np.float32)
    with pytest.raises(ValueError, match="weight shape"):
        matcher.zncc_surface(template, search, weight=weight)


def test_zncc_surface_accepts_matching_weight():
    template = np.zeros((10, 10), dtype=np.float32)
    search = np.zeros((100, 100), dtype=np.float32)
    weight = np.ones((10, 10), dtype=np.float32)
    with pytest.warns(UserWarning, match="masked ZNCC is not implemented"):
        assert matcher.zncc_surface(template, search, weight=weight).shape == (91, 91)


@pytest.mark.parametrize("factor", [2.0, 10.0, 9.97])
def test_area_average_downsample_shape(factor):
    image = np.zeros((1000, 1000), dtype=np.float32)
    out = matcher.area_average_downsample(image, factor)
    assert out.shape == (round(1000 / factor), round(1000 / factor))
    assert out.dtype == np.float32


def test_area_average_downsample_rejects_non_positive_factor():
    with pytest.raises(ValueError, match="strictly positive"):
        matcher.area_average_downsample(np.zeros((10, 10), dtype=np.float32), 0.0)


def test_build_template_is_roughly_nominal_size():
    reference = np.zeros((1000, 1000), dtype=np.float32)
    template = matcher.build_template(reference, theta=0.0, scale=10.0, psf_sigma_px=1.0)
    assert template.shape == (config.TEMPLATE_NOMINAL_PX, config.TEMPLATE_NOMINAL_PX)


def test_build_template_tracks_scale_residual():
    """A non-nominal scale must change the template size, not be rounded away."""
    reference = np.zeros((1000, 1000), dtype=np.float32)
    template = matcher.build_template(reference, theta=0.0, scale=10.3, psf_sigma_px=1.0)
    assert template.shape != (config.TEMPLATE_NOMINAL_PX, config.TEMPLATE_NOMINAL_PX)


def test_rotate_image_preserves_shape():
    image = np.zeros((64, 48), dtype=np.float32)
    assert matcher.rotate_image(image, 12.5).shape == image.shape


def test_match_psf_preserves_shape():
    image = np.zeros((64, 48), dtype=np.float32)
    assert matcher.match_psf(image, 1.5).shape == image.shape


def test_estimate_psf_sigma_returns_positive_float():
    sigma = matcher.estimate_psf_sigma(np.zeros((100, 100), dtype=np.float32))
    assert isinstance(sigma, float)
    assert sigma > 0.0


def test_top_k_peaks_returns_peaks_within_surface():
    surface = np.zeros((91, 91), dtype=np.float32)
    peaks = matcher.top_k_peaks(surface, k=10, nms_radius=4)
    assert all(isinstance(p, Peak) for p in peaks)
    assert all(0 <= p.col < 91 and 0 <= p.row < 91 for p in peaks)


def test_top_k_peaks_never_exceeds_k():
    surface = np.zeros((50, 50), dtype=np.float32)
    assert len(matcher.top_k_peaks(surface, k=3, nms_radius=2)) <= 3


@pytest.mark.parametrize("k", [0, -1])
def test_top_k_peaks_empty_for_non_positive_k(k):
    surface = np.zeros((50, 50), dtype=np.float32)
    assert matcher.top_k_peaks(surface, k=k, nms_radius=2) == []


def test_top_k_peaks_rejects_negative_radius():
    with pytest.raises(ValueError, match="nms_radius"):
        matcher.top_k_peaks(np.zeros((10, 10), dtype=np.float32), k=1, nms_radius=-1)


@pytest.mark.parametrize(
    "refine",
    [
        lambda s, p: matcher.refine_subpixel(s, p),
        lambda s, p: matcher.refine_subpixel_crop(
            np.zeros((10, 10), dtype=np.float32), np.zeros((100, 100), dtype=np.float32), p
        ),
    ],
    ids=["surface", "crop"],
)
def test_refine_subpixel_returns_finite_offset_pair(refine):
    surface = np.zeros((91, 91), dtype=np.float32)
    offset = refine(surface, Peak(col=45, row=45, score=1.0))
    assert len(offset) == 2
    assert all(np.isfinite(value) for value in offset)


def test_refine_subpixel_rejects_invalid_upsample():
    with pytest.raises(ValueError, match="upsample"):
        matcher.refine_subpixel(
            np.zeros((10, 10), dtype=np.float32), Peak(col=5, row=5, score=1.0), upsample=0
        )


def test_surface_shape_rejects_oversized_template():
    with pytest.raises(ValueError, match="exceeds"):
        matcher.surface_shape((200, 200), (100, 100))


# ---------------------------------------------------------------------------
# confidence
# ---------------------------------------------------------------------------


def test_extract_features_length_matches_feature_names():
    assert extract_features(Diagnostics()).shape == (len(FEATURE_NAMES),)


def test_extract_features_are_finite():
    diagnostics = Diagnostics(ncc_peak=0.9, psr=12.0, n_tied=2025, scale_est=10.4, theta_est=-3.2)
    assert np.all(np.isfinite(extract_features(diagnostics)))


def test_unfitted_model_still_predicts_in_range():
    model = ConfidenceModel()
    assert not model.is_fitted
    assert 0.0 <= model.predict(Diagnostics()) <= 1.0


def test_model_rejects_wrong_coefficient_count():
    with pytest.raises(ValueError, match="coefficients"):
        ConfidenceModel(coefficients=(1.0, 2.0))


def test_model_fit_rejects_mismatched_sample_counts():
    features = np.zeros((5, len(FEATURE_NAMES)), dtype=np.float64)
    correct = np.zeros((3,), dtype=np.float64)
    with pytest.raises(ValueError, match="sample count"):
        ConfidenceModel().fit(features, correct)


def test_model_fit_rejects_wrong_feature_width():
    features = np.zeros((5, len(FEATURE_NAMES) + 1), dtype=np.float64)
    correct = np.zeros((5,), dtype=np.float64)
    with pytest.raises(ValueError, match="features must have shape"):
        ConfidenceModel().fit(features, correct)


def test_model_save_load_round_trip(tmp_path):
    path = tmp_path / "nested" / "model.json"
    original = ConfidenceModel(coefficients=tuple(range(len(FEATURE_NAMES))), intercept=-1.5)
    original.save(path)
    restored = ConfidenceModel.load(path)
    assert restored.coefficients == original.coefficients
    assert restored.intercept == original.intercept


def test_model_load_rejects_stale_feature_order(tmp_path):
    path = tmp_path / "stale.json"
    path.write_text(json.dumps({"feature_names": ["only_one"], "coefficients": [1.0]}))
    with pytest.raises(ValueError, match="different feature order"):
        ConfidenceModel.load(path)


def test_load_or_default_tolerates_missing_file(tmp_path):
    assert not ConfidenceModel.load_or_default(tmp_path / "absent.json").is_fitted


def test_load_or_default_tolerates_none():
    assert not ConfidenceModel.load_or_default(None).is_fitted


def test_load_or_default_tolerates_corrupt_file(tmp_path):
    path = tmp_path / "corrupt.json"
    path.write_text("{not valid json")
    assert not ConfidenceModel.load_or_default(path).is_fitted


def test_internal_error_always_flags_low_confidence():
    diagnostics = Diagnostics(psr=1e6, failure_mode="internal_error")
    assert is_low_confidence(1.0, diagnostics) is True


def test_weak_psr_flags_low_confidence():
    diagnostics = Diagnostics(psr=config.PSR_AMBIGUOUS_THRESHOLD - 1.0)
    assert is_low_confidence(1.0, diagnostics) is True


# ---------------------------------------------------------------------------
# localize — the never-raises contract
# ---------------------------------------------------------------------------


def _valid_pair():
    rng = np.random.default_rng(0)
    return (
        rng.integers(0, 255, size=(400, 400), dtype=np.uint8),
        rng.integers(0, 255, size=(1000, 1000), dtype=np.uint8),
    )


def test_localize_returns_result_type():
    search, reference = _valid_pair()
    assert isinstance(localize(search, reference), LocalizationResult)


def test_localize_result_fields_are_valid():
    search, reference = _valid_pair()
    result = localize(search, reference)
    assert np.isfinite(result.x)
    assert np.isfinite(result.y)
    assert 0.0 <= result.confidence <= 1.0
    assert isinstance(result.low_confidence_flag, bool)
    assert result.diagnostics.elapsed_ms >= 0.0


@pytest.mark.parametrize("mode", ["auto", "fast", "robust", "ambiguous"])
def test_localize_accepts_every_mode(mode):
    search, reference = _valid_pair()
    assert isinstance(localize(search, reference, mode=mode), LocalizationResult)


@pytest.mark.parametrize(
    "search",
    [
        np.zeros((200, 200), dtype=np.uint8),
        np.full((200, 200), 128, dtype=np.uint8),
        np.zeros((200, 200), dtype=np.float64),
        np.zeros((200, 200), dtype=np.float32).T,
        np.random.default_rng(1).normal(0, 1e6, size=(200, 200)),
    ],
    ids=["zeros", "constant", "float64", "non-contiguous", "extreme-noise"],
)
def test_localize_never_raises_on_awkward_but_valid_input(search):
    reference = np.zeros((100, 100), dtype=np.uint8)
    assert isinstance(localize(search, reference), LocalizationResult)


@pytest.mark.parametrize(
    "search, reference",
    [
        (np.zeros((10,), dtype=np.uint8), np.zeros((5,), dtype=np.uint8)),
        (np.zeros((0, 0), dtype=np.uint8), np.zeros((5, 5), dtype=np.uint8)),
        (np.zeros((3, 3), dtype=np.uint8), np.zeros((100, 100), dtype=np.uint8)),
        (np.zeros((4, 4, 3), dtype=np.uint8), np.zeros((2, 2), dtype=np.uint8)),
    ],
    ids=["one-dimensional", "empty", "template-exceeds-search", "three-dimensional"],
)
def test_localize_degrades_instead_of_raising(search, reference):
    result = localize(search, reference)
    assert isinstance(result, LocalizationResult)
    assert result.low_confidence_flag is True
    assert result.confidence == 0.0
    assert result.diagnostics.failure_mode == "internal_error"
    assert result.diagnostics.notes


def test_localize_fallback_uses_image_centre_when_shape_is_usable():
    """A search image too small to hold the template still gets a centre answer."""
    search = np.zeros((3, 3), dtype=np.uint8)
    reference = np.zeros((100, 100), dtype=np.uint8)
    result = localize(search, reference)
    assert result.diagnostics.failure_mode == "internal_error"
    assert result.centre == config.image_centre((3, 3))


def test_reference_larger_than_search_is_the_normal_case():
    """The reference is 10x finer, so in raw pixels it is *expected* to be bigger.

    A 1000x1000 reference at 1 nm/px decimates to roughly 100x100 in the search
    grid. Treating "reference bigger than search" as an error would reject every
    legitimate input.
    """
    search = np.zeros((1000, 1000), dtype=np.uint8)
    reference = np.zeros((1000, 1000), dtype=np.uint8)
    result = localize(search, reference)
    assert result.diagnostics.failure_mode == "none"


def test_localize_records_requested_mode_in_diagnostics():
    search, reference = _valid_pair()
    assert localize(search, reference, mode="fast").diagnostics.mode_used == "fast"
