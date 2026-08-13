"""The ``refine`` flag on :func:`src.localize.localize`.

Requested by R3 for the Phase 3 ablations table. The other rows — weighting and
selection — can be measured from ``_StageCache`` directly; Stage 5 is the only
one that lives behind ``localize()``, so it needed a switch rather than being
derived from the difference between two other numbers.

These tests are mostly about what the flag *does not* touch. The value of the
ablation depends on ``refine=False`` isolating Stage 5 exactly: if it also
perturbed selection or confidence, the row would measure a mixture and the table
would quietly overstate what refinement buys.
"""

import dataclasses
import inspect

import numpy as np
import pytest

from src.localize import localize
from src.types import Diagnostics

# Fields that are expected to differ between a refined and an unrefined run.
# Everything else must be identical, which is the point of the flag.
_STAGE_FIVE_FIELDS = frozenset({"subpixel_error", "subpixel_method"})

# Wall-clock, so it differs run to run regardless of what the flag does.
_NONDETERMINISTIC_FIELDS = frozenset({"elapsed_ms"})


def _comparable(diagnostics: Diagnostics) -> dict[str, object]:
    """Return the diagnostics fields that ``refine`` must leave alone."""
    return {
        f.name: getattr(diagnostics, f.name)
        for f in dataclasses.fields(diagnostics)
        if f.name not in _STAGE_FIVE_FIELDS | _NONDETERMINISTIC_FIELDS
    }


# ---------------------------------------------------------------------------
# Default behaviour is unchanged
# ---------------------------------------------------------------------------


def test_refine_defaults_to_true():
    """Nothing existing moves: the parameter defaults to the old behaviour."""
    assert inspect.signature(localize).parameters["refine"].default is True


def test_omitting_the_flag_matches_passing_true(two_scale_pair):
    """An unchanged call site produces an unchanged answer, field for field."""
    reference, search, _ = two_scale_pair

    implicit = localize(search, reference)
    explicit = localize(search, reference, refine=True)

    assert (implicit.x, implicit.y) == (explicit.x, explicit.y)
    assert implicit.confidence == explicit.confidence
    assert _comparable(implicit.diagnostics) == _comparable(explicit.diagnostics)
    assert implicit.diagnostics.subpixel_error == pytest.approx(
        explicit.diagnostics.subpixel_error, nan_ok=True
    )
    assert implicit.diagnostics.subpixel_method == explicit.diagnostics.subpixel_method


# ---------------------------------------------------------------------------
# What refine=False changes
# ---------------------------------------------------------------------------


def test_refine_false_returns_the_integer_peak_centre(two_scale_pair):
    """The unrefined answer sits on the peak grid.

    ``Peak.centre`` offsets a top-left corner by ``(edge - 1) / 2``, so an
    unrefined centre is always an integer or a half-integer. A surviving
    sub-pixel shift would land off that grid with probability one.
    """
    reference, search, _ = two_scale_pair

    result = localize(search, reference, refine=False)

    assert float(result.x * 2).is_integer()
    assert float(result.y * 2).is_integer()


def test_refine_false_reports_the_refinement_as_absent(two_scale_pair):
    """Skipped is reported as absent, not as zero error.

    ``nan`` is this codebase's absent-measurement convention — the same one
    ``peak_to_sidelobe`` uses — and ``"none"`` is already the ``Diagnostics``
    default for a stage that did not run. Reporting ``0.0`` would claim a
    perfect refinement had happened.
    """
    reference, search, _ = two_scale_pair

    result = localize(search, reference, refine=False)

    assert np.isnan(result.diagnostics.subpixel_error)
    assert result.diagnostics.subpixel_method == "none"


def test_refine_only_moves_the_answer_by_a_sub_pixel_amount(two_scale_pair):
    """The flag removes a sub-pixel correction, not a different candidate.

    If the two paths ever disagreed by more than a pixel it would mean
    disambiguation had selected differently, and the ablation row would be
    measuring selection rather than refinement.
    """
    reference, search, _ = two_scale_pair

    refined = localize(search, reference, refine=True)
    unrefined = localize(search, reference, refine=False)

    assert abs(refined.x - unrefined.x) < 1.0
    assert abs(refined.y - unrefined.y) < 1.0


# ---------------------------------------------------------------------------
# What refine=False must not change
# ---------------------------------------------------------------------------


def test_refine_leaves_every_other_diagnostic_untouched(two_scale_pair):
    """Stage 5 is isolated: selection, PSR, pose and mode all agree.

    This is the assumption the ablations table rests on.
    """
    reference, search, _ = two_scale_pair

    refined = localize(search, reference, refine=True)
    unrefined = localize(search, reference, refine=False)

    assert _comparable(refined.diagnostics) == _comparable(unrefined.diagnostics)


def test_refine_does_not_change_confidence(two_scale_pair):
    """Calibration is untouched.

    ``extract_features`` reads ``ncc_peak``, ``psr``, ``n_tied``,
    ``uniqueness_score``, ``scale_est`` and ``theta_est`` — not
    ``subpixel_error`` — so an unrefined run must score identically. If this
    ever fails, the feature vector gained a Stage 5 term and the ablation is no
    longer clean.
    """
    reference, search, _ = two_scale_pair

    refined = localize(search, reference, refine=True)
    unrefined = localize(search, reference, refine=False)

    assert refined.confidence == unrefined.confidence
    assert refined.low_confidence_flag == unrefined.low_confidence_flag


def test_the_diagnostics_record_keeps_its_shape(two_scale_pair):
    """No field appears or disappears — the same record, differently filled."""
    reference, search, _ = two_scale_pair

    refined = localize(search, reference, refine=True)
    unrefined = localize(search, reference, refine=False)

    refined_fields = [f.name for f in dataclasses.fields(refined.diagnostics)]
    unrefined_fields = [f.name for f in dataclasses.fields(unrefined.diagnostics)]

    assert refined_fields == unrefined_fields


def test_a_skipped_stage_five_records_no_timing(two_scale_pair, monkeypatch):
    """The stage timer sits inside the gate, not around it.

    A skipped Stage 5 should be *absent* from the breakdown rather than present
    at zero: an ablation row comparing the two paths must be able to tell "did
    not run" from "ran and cost nothing". This pins the interaction between the
    refine gate and the timing instrumentation, which is the seam a later
    refactor is most likely to get wrong.
    """
    from src import config

    reference, search, _ = two_scale_pair
    monkeypatch.setattr(config, "COLLECT_STAGE_TIMINGS", True)

    refined = localize(search, reference, refine=True)
    unrefined = localize(search, reference, refine=False)

    assert "refine_subpixel" in refined.diagnostics.stage_ms
    assert "refine_subpixel" not in unrefined.diagnostics.stage_ms
    # The stages either side of the gate still report, so this is the gate
    # working rather than the instrumentation being off.
    assert "correlate" in unrefined.diagnostics.stage_ms


# ---------------------------------------------------------------------------
# The never-raises contract still holds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "search,reference",
    [
        (np.zeros((5, 5)), np.zeros((10, 10))),  # template larger than search
        (np.zeros((1, 1)), np.zeros((1, 1))),
        ([[0.0, 1.0], [1.0, 0.0]], [[0.0, 1.0], [1.0, 0.0]]),  # plain lists
        (np.zeros((40, 40), dtype=np.uint16), np.zeros((10, 10), dtype=np.int8)),
    ],
)
def test_refine_false_never_raises(search, reference):
    """Adversarial input degrades to a flagged answer, exactly as with True."""
    result = localize(search, reference, refine=False)

    assert np.isfinite(result.x)
    assert np.isfinite(result.y)
    assert 0.0 <= result.confidence <= 1.0


def test_fallback_path_is_unaffected_by_the_flag():
    """A pair that cannot be matched fails identically either way."""
    refined = localize(np.zeros((5, 5)), np.zeros((10, 10)), refine=True)
    unrefined = localize(np.zeros((5, 5)), np.zeros((10, 10)), refine=False)

    assert (refined.x, refined.y) == (unrefined.x, unrefined.y)
    assert refined.diagnostics.failure_mode == unrefined.diagnostics.failure_mode
    assert refined.low_confidence_flag == unrefined.low_confidence_flag
