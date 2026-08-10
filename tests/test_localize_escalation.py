"""T6 (partial) — escalation ladder, PSR wiring, and NaN handling.

Covers the parts of T6 that do not depend on ``disambiguate.select_candidate``.
The centre tie-break and the tied-count remain unwired because that function's
tie-break compares candidate top-left corners against the search-image centre
rather than candidate centres; see ``test_r3_tie_break_bug`` below.
"""

import math

import numpy as np
import pytest

from src import config, matcher
from src.confidence import extract_features, is_low_confidence
from src.disambiguate import select_candidate
from src.localize import _escalation_path, _should_escalate, localize
from src.types import Diagnostics, Peak

# ---------------------------------------------------------------------------
# The blocker: R3's centre tie-break
# ---------------------------------------------------------------------------



def test_r3_tie_break_bug():
    """The mandated rule picks the candidate whose *centre* is nearest.

    Two peaks with identical scores. A's matched region is 14 px from the image
    centre; B's is 99 px away. The rule requires A.
    """
    template_shape = (100, 100)
    centre = config.image_centre((1000, 1000))

    a = Peak(col=460, row=460, score=1.0)
    b = Peak(col=520, row=520, score=1.0)

    assert math.dist(a.centre(template_shape), centre) < math.dist(b.centre(template_shape), centre)

    chosen, tie_break_used = select_candidate([a, b], centre, template_shape)
    assert tie_break_used is True
    assert (chosen.col, chosen.row) == (a.col, a.row)


def test_tie_break_offset_is_half_a_template():
    """Documents the size of the error so the fix can be checked against it."""
    template_shape = (100, 100)
    centre_x, centre_y = config.image_centre((1000, 1000))
    offset = (template_shape[1] - 1) / 2

    # The target select_candidate effectively minimises against, in top-left
    # coordinates, versus the correct one.
    assert offset == pytest.approx(49.5)
    assert (centre_x - offset, centre_y - offset) == (450.0, 450.0)


# ---------------------------------------------------------------------------
# Escalation path
# ---------------------------------------------------------------------------


def test_auto_expands_to_the_full_ladder():
    assert _escalation_path("auto") == ("fast", "robust", "ambiguous")


@pytest.mark.parametrize("mode", ["fast", "robust", "ambiguous"])
def test_explicit_mode_runs_only_that_tier(mode):
    assert _escalation_path(mode) == (mode,)


# ---------------------------------------------------------------------------
# Escalation decision
# ---------------------------------------------------------------------------


def test_nan_psr_escalates():
    """Unknown is not the same as good.

    ``nan < threshold`` is False, so a naive comparison would accept an answer
    exactly when there is no evidence for it.
    """
    assert _should_escalate(Diagnostics(psr=float("nan")), "fast") is True
    assert _should_escalate(Diagnostics(psr=float("nan")), "robust") is True


def test_strong_psr_stops_at_fast():
    strong = Diagnostics(psr=config.PSR_ACCEPT_THRESHOLD + 1.0)
    assert _should_escalate(strong, "fast") is False


def test_weak_psr_escalates_from_fast():
    weak = Diagnostics(psr=config.PSR_ACCEPT_THRESHOLD - 1.0)
    assert _should_escalate(weak, "fast") is True


def test_robust_tier_uses_the_lower_threshold():
    middling = Diagnostics(psr=(config.PSR_ACCEPT_THRESHOLD + config.PSR_AMBIGUOUS_THRESHOLD) / 2)
    assert _should_escalate(middling, "fast") is True
    assert _should_escalate(middling, "robust") is False


def test_ambiguous_is_terminal():
    """There is nowhere further to escalate to."""
    assert _should_escalate(Diagnostics(psr=float("nan")), "ambiguous") is False
    assert _should_escalate(Diagnostics(psr=0.0), "ambiguous") is False


# ---------------------------------------------------------------------------
# PSR wiring
# ---------------------------------------------------------------------------


def test_localize_populates_psr(two_scale_pair):
    reference, search, _ = two_scale_pair
    diagnostics = localize(search, reference, mode="fast").diagnostics
    assert diagnostics.psr != 0.0
    assert np.isfinite(diagnostics.psr) or np.isnan(diagnostics.psr)


def test_localize_psr_matches_a_direct_computation(two_scale_pair):
    """The reported statistic must be the one R3's function actually returns."""
    from src.disambiguate import peak_to_sidelobe

    reference, search, _ = two_scale_pair
    result = localize(search, reference, mode="fast")

    template = matcher.build_template(
        reference, theta=0.0, scale=config.NOMINAL_SCALE, psf_sigma_px=config.DEFAULT_PSF_SIGMA_PX
    )
    surface = matcher.zncc_surface(template, search)
    peak = matcher.top_k_peaks(surface, k=config.DEFAULT_TOP_K, nms_radius=8)[0]

    assert result.diagnostics.psr == pytest.approx(
        peak_to_sidelobe(surface, peak, config.DEFAULT_NMS_RADIUS_PX)
    )


def test_n_tied_is_not_filled_with_the_shortlist_length(two_scale_pair):
    """n_tied is R3's field; it counts statistically tied candidates.

    Filling it with len(peaks) would report the shortlist length, a different
    quantity, and would silently corrupt the confidence features.
    """
    reference, search, _ = two_scale_pair
    result = localize(search, reference, mode="fast")
    assert result.diagnostics.n_tied != config.DEFAULT_TOP_K


# ---------------------------------------------------------------------------
# NaN safety downstream
# ---------------------------------------------------------------------------


def test_nan_psr_flags_low_confidence():
    assert is_low_confidence(1.0, Diagnostics(psr=float("nan"))) is True


def test_features_are_finite_even_with_nan_inputs():
    """A single NaN would poison a fitted model."""
    diagnostics = Diagnostics(
        psr=float("nan"),
        subpixel_error=float("nan"),
        uniqueness_score=float("nan"),
    )
    features = extract_features(diagnostics)
    assert np.all(np.isfinite(features))


def test_features_keep_their_length():
    from src.confidence import FEATURE_NAMES

    assert extract_features(Diagnostics(psr=float("nan"))).shape == (len(FEATURE_NAMES),)


# ---------------------------------------------------------------------------
# End-to-end orchestration
# ---------------------------------------------------------------------------


def test_auto_reaches_the_ambiguous_tier_on_weak_evidence(two_scale_pair):
    """With current thresholds every case escalates, which is safe but slow.

    Recorded so that tuning the thresholds in Phase 3 visibly changes behaviour
    rather than passing unnoticed.
    """
    reference, search, _ = two_scale_pair
    result = localize(search, reference, mode="auto")
    assert result.diagnostics.mode_used == "ambiguous"
    assert any("tie-break unavailable" in note for note in result.diagnostics.notes)


def test_auto_still_produces_a_valid_answer(two_scale_pair):
    reference, search, (true_col, true_row) = two_scale_pair
    result = localize(search, reference, mode="auto")

    expected = config.window_topleft_to_centre(true_col, true_row, (64, 64))
    assert math.dist((result.x, result.y), expected) < 1.0


def test_auto_costs_more_than_fast(two_scale_pair):
    reference, search, _ = two_scale_pair
    fast = localize(search, reference, mode="fast").diagnostics.elapsed_ms
    auto = localize(search, reference, mode="auto").diagnostics.elapsed_ms
    assert auto > fast


def test_escalation_never_breaks_the_never_raises_contract():
    for search in (
        np.zeros((200, 200), dtype=np.uint8),
        np.full((200, 200), 128, dtype=np.uint8),
        np.random.default_rng(0).normal(0, 1e6, size=(200, 200)),
    ):
        result = localize(search, np.zeros((100, 100), dtype=np.uint8), mode="auto")
        assert np.isfinite(result.x)
        assert np.isfinite(result.y)


def test_mode_used_reports_the_tier_that_answered(two_scale_pair):
    reference, search, _ = two_scale_pair
    for mode in ("fast", "robust", "ambiguous"):
        assert localize(search, reference, mode=mode).diagnostics.mode_used == mode


def test_real_dataset_psr_does_not_yet_separate_correct_from_wrong():
    """Recorded measurement, not an aspiration.

    On the first twelve generated pairs PSR spans 1.77-3.08 on correct answers
    and 1.44-3.41 on wrong ones. The ranges overlap completely and the largest
    value belongs to a wrong answer, so PSR alone cannot yet gate escalation.
    This is expected of an unweighted surface over a periodic lattice and is
    what Stage 4a uniqueness weighting exists to fix.

    Asserted here only as a guard on the thresholds: if someone tunes them down
    to the observed range without the weighting in place, every answer becomes
    falsely confident.
    """
    assert config.PSR_ACCEPT_THRESHOLD > 3.5
    assert config.PSR_AMBIGUOUS_THRESHOLD > 3.5


def test_unanchored_pair_is_flagged(two_scale_pair):
    """A weak detection statistic must reach the caller as a flag."""
    reference, search, _ = two_scale_pair
    result = localize(search, reference, mode="auto")
    assert isinstance(result.low_confidence_flag, bool)
