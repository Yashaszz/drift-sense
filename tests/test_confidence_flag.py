"""The low-confidence flag as a safety property.

The flag is what a real tool acts on: it decides whether to trust an answer or
widen the search. Its value therefore does not come from being accurate on
average, it comes from never being *falsely clear*. Every gate is
one-directional — each can raise the flag, none can clear it — so a calibrator
that is unfitted, at chance, or actively wrong degrades the confidence number
without degrading the safety property.

These tests are written against that invariant rather than against the current
model, so they keep holding when the calibrator is re-fitted.
"""

import numpy as np
import pytest

from src import config
from src.confidence import is_low_confidence
from src.types import Diagnostics

_GOOD_PSR = 20.0
"""Comfortably above any threshold, so each test isolates one gate."""


def _clear() -> Diagnostics:
    """Diagnostics that no gate objects to."""
    return Diagnostics(psr=_GOOD_PSR, n_tied=1, tie_break_used=False, failure_mode="none")


def test_the_baseline_is_not_flagged():
    """Without this, every test below would pass vacuously."""
    assert is_low_confidence(0.99, _clear(), threshold=0.5) is False


# ---------------------------------------------------------------------------
# Gates that must raise the flag regardless of the score
# ---------------------------------------------------------------------------


def test_a_failure_mode_flags_even_at_full_confidence():
    """An internal failure is not something a high score may overrule."""
    diagnostics = _clear()
    diagnostics.failure_mode = "snr_collapse"

    assert is_low_confidence(1.0, diagnostics, threshold=0.5) is True


def test_an_unmeasurable_psr_flags():
    """NaN means the ambiguity could not be measured, not that there is none."""
    diagnostics = _clear()
    diagnostics.psr = float("nan")

    assert is_low_confidence(1.0, diagnostics, threshold=0.5) is True


def test_an_ambiguous_peak_flags():
    """Below the ambiguity bar the peak structure itself is the evidence."""
    diagnostics = _clear()
    diagnostics.psr = config.get_thresholds().psr_ambiguous - 0.1

    assert is_low_confidence(1.0, diagnostics, threshold=0.5) is True


def test_a_tie_break_decided_answer_flags():
    """The centre rule is a prior, not evidence.

    When the tie-break chose between genuinely tied candidates, the correlation
    surface ranked nothing and the answer rests on "the stage aimed here". That
    is the unanchored case, which measures 0% accuracy — precisely where a
    confident-looking score is most harmful.
    """
    diagnostics = _clear()
    diagnostics.tie_break_used = True
    diagnostics.n_tied = 7

    assert is_low_confidence(1.0, diagnostics, threshold=0.5) is True


def test_a_tie_break_over_a_single_candidate_does_not_flag():
    """A one-candidate shortlist is not a tie, whatever the flag says.

    select_candidate reports no tie-break for a single candidate, so this
    combination should not arise; asserting it keeps the gate from widening into
    "any answer at all" if that ever changes.
    """
    diagnostics = _clear()
    diagnostics.tie_break_used = True
    diagnostics.n_tied = 1

    assert is_low_confidence(0.99, diagnostics, threshold=0.5) is False


def test_a_non_finite_confidence_flags():
    """A NaN score would sail through a bare ``confidence < threshold``.

    Same failure class as the NaN psr gate, one level up: the comparison answers
    False for NaN, clearing the flag exactly when the model has told us nothing.
    """
    assert is_low_confidence(float("nan"), _clear(), threshold=0.5) is True


# ---------------------------------------------------------------------------
# The invariant itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("confidence", [0.0, 0.25, 0.5, 0.75, 1.0, float("nan")])
@pytest.mark.parametrize(
    "field,value",
    [
        ("failure_mode", "snr_collapse"),
        ("psr", float("nan")),
        ("psr", 0.0),
    ],
)
def test_no_score_can_clear_a_raised_gate(confidence, field, value):
    """The safety property: gates are one-directional, at every score."""
    diagnostics = _clear()
    setattr(diagnostics, field, value)

    assert is_low_confidence(confidence, diagnostics, threshold=0.5) is True


def test_the_score_still_decides_when_no_gate_objects():
    """Hardening must not collapse the flag into "always true"."""
    assert is_low_confidence(0.9, _clear(), threshold=0.5) is False
    assert is_low_confidence(0.1, _clear(), threshold=0.5) is True


def test_the_ambiguity_gate_follows_a_retuned_threshold():
    """Tuning the ladder retunes the flag with it, by construction."""
    diagnostics = _clear()
    diagnostics.psr = 10.0

    assert is_low_confidence(0.99, diagnostics, threshold=0.5) is False
    with config.override_thresholds(psr_accept=12.0, psr_ambiguous=12.0):
        assert is_low_confidence(0.99, diagnostics, threshold=0.5) is True


def test_flag_is_a_plain_bool():
    """It is serialised into the CLI's JSON output and read by evaluate.py."""
    assert isinstance(is_low_confidence(0.9, _clear(), threshold=0.5), bool)
    assert isinstance(is_low_confidence(np.float64(0.9), _clear(), threshold=0.5), bool)
