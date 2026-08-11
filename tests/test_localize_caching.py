"""T9 — the per-call memo that collapses repeated escalation tiers.

The optimisation claim is narrow and testable: a tier whose parameters match an
earlier tier's produces an identical outcome, so it may reuse it. These tests
assert that equivalence directly, and guard the two ways a cache can go wrong —
leaking between calls, and going stale when parameters actually change.
"""

import numpy as np
import pytest

from src import config, matcher
from src.localize import _StageCache, _TierOutcome, localize
from src.types import Diagnostics, Peak
from tests.conftest import make_two_scale_pair


@pytest.fixture
def cache(two_scale_pair):
    reference, search, _ = two_scale_pair
    return _StageCache(
        np.ascontiguousarray(search, dtype=np.float32),
        np.ascontiguousarray(reference, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# correlate()
# ---------------------------------------------------------------------------


def test_repeat_key_returns_the_same_arrays(cache):
    """A hit must reuse, not recompute — identity is the observable proof."""
    first = cache.correlate(0.0, config.NOMINAL_SCALE, 1.0)
    second = cache.correlate(0.0, config.NOMINAL_SCALE, 1.0)
    assert first[0] is second[0]
    assert first[1] is second[1]
    assert first[2] is second[2]


@pytest.mark.parametrize(
    "changed",
    [(1.0, config.NOMINAL_SCALE, 1.0), (0.0, 10.3, 1.0), (0.0, config.NOMINAL_SCALE, 2.0)],
)
def test_any_parameter_change_invalidates(cache, changed):
    baseline = cache.correlate(0.0, config.NOMINAL_SCALE, 1.0)
    assert cache.correlate(*changed)[1] is not baseline[1]


def test_recomputation_matches_the_uncached_path(cache, two_scale_pair):
    """A cache miss must produce exactly what calling the stages directly does."""
    reference, search, _ = two_scale_pair
    template, surface, peaks = cache.correlate(0.0, config.NOMINAL_SCALE, 1.0)

    direct_template = matcher.build_template(
        np.ascontiguousarray(reference, dtype=np.float32),
        theta=0.0,
        scale=config.NOMINAL_SCALE,
        psf_sigma_px=1.0,
    )
    direct_surface = matcher.zncc_surface(
        direct_template, np.ascontiguousarray(search, dtype=np.float32)
    )

    np.testing.assert_array_equal(template, direct_template)
    np.testing.assert_array_equal(surface, direct_surface)
    assert peaks == matcher.top_k_peaks(
        direct_surface, k=config.DEFAULT_TOP_K, nms_radius=config.DEFAULT_NMS_RADIUS_PX
    )


# ---------------------------------------------------------------------------
# psf_sigma()
# ---------------------------------------------------------------------------


def test_fast_tier_skips_estimation(cache):
    assert cache.psf_sigma("fast") == config.DEFAULT_PSF_SIGMA_PX


def test_estimate_is_computed_once_and_reused(cache, monkeypatch):
    calls = {"n": 0}
    real = matcher.estimate_psf_sigma

    def counted(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(matcher, "estimate_psf_sigma", counted)

    first = cache.psf_sigma("robust")
    second = cache.psf_sigma("ambiguous")
    assert calls["n"] == 1
    assert first == second


# ---------------------------------------------------------------------------
# Outcome memo
# ---------------------------------------------------------------------------


def test_outcome_round_trips(cache):
    key = (0.0, 10.0, 1.0, False)
    assert cache.outcome(key) is None

    outcome = _TierOutcome(
        centre=(1.5, 2.5),
        n_tied=3,
        tie_break_used=True,
        ncc_peak=0.9,
        theta_est=0.0,
        scale_est=10.0,
        psr=4.2,
        uniqueness_score=0.5,
        subpixel_error=0.01,
        subpixel_method="phase_cross_correlation",
    )
    cache.store(key, outcome)
    assert cache.outcome(key) is outcome
    assert cache.outcome((0.0, 10.0, 2.0, False)) is None


def test_outcome_apply_populates_every_field_but_mode():
    diagnostics = Diagnostics(mode_used="ambiguous")
    outcome = _TierOutcome(
        centre=(4.0, 5.0),
        n_tied=7,
        tie_break_used=True,
        ncc_peak=0.8,
        theta_est=1.5,
        scale_est=10.2,
        psr=3.3,
        uniqueness_score=0.75,
        subpixel_error=0.02,
        subpixel_method="surface_upsampling",
    )
    assert outcome.apply(diagnostics) == (4.0, 5.0)
    assert diagnostics.n_tied == 7
    assert diagnostics.tie_break_used is True
    assert diagnostics.psr == 3.3
    assert diagnostics.uniqueness_score == 0.75
    assert diagnostics.subpixel_method == "surface_upsampling"
    assert diagnostics.mode_used == "ambiguous"


# ---------------------------------------------------------------------------
# The property the optimisation rests on
# ---------------------------------------------------------------------------


def test_tiers_sharing_parameters_share_an_outcome(two_scale_pair):
    """Escalating without changing pose or PSF cannot change the answer.

    This is the equivalence the memo exploits, asserted end to end rather than
    inferred from the implementation. ``robust`` and ``ambiguous`` both use the
    estimated PSF and the nominal pose, so they resolve to the same key.
    """
    reference, search, _ = two_scale_pair
    robust = localize(search, reference, mode="robust")
    ambiguous = localize(search, reference, mode="ambiguous")

    assert (robust.x, robust.y) == (ambiguous.x, ambiguous.y)
    assert robust.diagnostics.ncc_peak == ambiguous.diagnostics.ncc_peak
    assert robust.diagnostics.n_tied == ambiguous.diagnostics.n_tied
    assert robust.diagnostics.psr == pytest.approx(ambiguous.diagnostics.psr, nan_ok=True)


def test_fast_is_not_merged_with_the_later_tiers(two_scale_pair):
    """The key must separate tiers that genuinely differ.

    ``fast`` assumes the default PSF while ``robust`` measures it, so on this
    fixture the estimator returns 1.17 against a default of 1.0 and the two
    tiers build different templates. A key that ignored the PSF width would
    wrongly reuse fast's surface and silently disable the robust path.
    """
    reference, search, _ = two_scale_pair
    fast = localize(search, reference, mode="fast")
    robust = localize(search, reference, mode="robust")
    assert fast.diagnostics.ncc_peak != robust.diagnostics.ncc_peak


def test_mode_used_still_reports_the_final_tier(two_scale_pair):
    """Reuse must not disguise how far the ladder actually climbed."""
    reference, search, _ = two_scale_pair
    assert localize(search, reference, mode="auto").diagnostics.mode_used == "ambiguous"


def test_repeated_calls_are_deterministic(two_scale_pair):
    reference, search, _ = two_scale_pair
    first = localize(search, reference, mode="auto").diagnostics.to_dict()
    second = localize(search, reference, mode="auto").diagnostics.to_dict()

    # Wall-clock timing is the one field that legitimately varies run to run.
    first.pop("elapsed_ms")
    second.pop("elapsed_ms")
    assert first == second


def test_cache_does_not_leak_between_pairs():
    """The memo is per call. A shared one would answer for the wrong image."""
    reference_a, search_a, truth_a = make_two_scale_pair()
    reference_b, search_b, _ = make_two_scale_pair(noise=0.3)

    localize(search_b, reference_b, mode="auto")
    result = localize(search_a, reference_a, mode="auto")

    expected = config.window_topleft_to_centre(truth_a[0], truth_a[1], (64, 64))
    assert abs(result.x - expected[0]) < 1.0
    assert abs(result.y - expected[1]) < 1.0


def test_caching_did_not_change_the_never_raises_contract():
    for search in (
        np.zeros((200, 200), dtype=np.uint8),
        np.full((200, 200), 128, dtype=np.uint8),
    ):
        result = localize(search, np.zeros((100, 100), dtype=np.uint8), mode="auto")
        assert np.isfinite(result.x)
        assert np.isfinite(result.y)


# ---------------------------------------------------------------------------
# PSR shortcut
# ---------------------------------------------------------------------------


def test_psr_from_stats_matches_r3(two_scale_pair):
    """Guards the duplicated arithmetic against drifting from R3's definition."""
    from src.disambiguate import peak_to_sidelobe, sidelobe_stats
    from src.localize import _psr_from_stats

    reference, search, _ = two_scale_pair
    template = matcher.build_template(reference, 0.0, config.NOMINAL_SCALE, 1.0)
    surface = matcher.zncc_surface(template, search)
    peak = matcher.top_k_peaks(surface, config.DEFAULT_TOP_K, config.DEFAULT_NMS_RADIUS_PX)[0]

    mean, std = sidelobe_stats(surface, peak, config.DEFAULT_NMS_RADIUS_PX)
    assert _psr_from_stats(surface, peak, mean, std) == pytest.approx(
        peak_to_sidelobe(surface, peak, config.DEFAULT_NMS_RADIUS_PX)
    )


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_psr_from_stats_returns_nan_on_undefined_statistics(bad):
    from src.localize import _psr_from_stats

    surface = np.zeros((10, 10), dtype=np.float32)
    peak = Peak(col=5, row=5, score=1.0)
    assert np.isnan(_psr_from_stats(surface, peak, bad, 1.0))
    assert np.isnan(_psr_from_stats(surface, peak, 0.0, bad))
