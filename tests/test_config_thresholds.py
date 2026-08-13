"""Threshold infrastructure for Phase 3 tuning.

The escalation ladder and the low-confidence flag both key off PSR decision
points. Tuning them used to mean editing ``config.py`` between runs, which makes
a sweep unreproducible and leaves the process mistuned if it fails partway.
These tests pin the contract a sweep depends on: overrides take effect at the
decision point, and they always unwind.
"""

import numpy as np
import pytest

from src import config
from src.confidence import is_low_confidence
from src.localize import _should_escalate, localize
from src.types import Diagnostics


@pytest.fixture(autouse=True)
def _restore_thresholds():
    """No test may leak a threshold change into the next one."""
    previous = config.get_thresholds()
    yield
    config.set_thresholds(previous)


# ---------------------------------------------------------------------------
# The value object
# ---------------------------------------------------------------------------


def test_defaults_match_the_module_constants():
    """The refactor changes where the numbers are read, not what they are."""
    thresholds = config.Thresholds()

    assert thresholds.psr_accept == config.PSR_ACCEPT_THRESHOLD
    assert thresholds.psr_ambiguous == config.PSR_AMBIGUOUS_THRESHOLD


def test_the_bar_drops_as_the_pipeline_spends_more():
    """Having paid for the expensive path, a weaker peak is the best on offer."""
    thresholds = config.Thresholds(psr_accept=8.0, psr_ambiguous=4.0)

    assert thresholds.for_tier("fast") == 8.0
    assert thresholds.for_tier("robust") == 4.0
    assert thresholds.for_tier("ambiguous") == 4.0


def test_an_unorderable_ladder_is_rejected():
    """accept < ambiguous would make robust accept what fast escalated."""
    with pytest.raises(ValueError, match="psr_accept"):
        config.Thresholds(psr_accept=2.0, psr_ambiguous=6.0)


def test_equal_thresholds_are_allowed():
    """A degenerate but coherent ladder: one bar for every tier."""
    assert config.Thresholds(psr_accept=5.0, psr_ambiguous=5.0).for_tier("fast") == 5.0


# ---------------------------------------------------------------------------
# Overrides reach the decision points
# ---------------------------------------------------------------------------


def test_override_changes_the_escalation_decision():
    """The point of the infrastructure: no source edit between sweep steps."""
    diagnostics = Diagnostics(psr=5.0)

    assert _should_escalate(diagnostics, "fast") is True  # 5.0 < default 8.0
    with config.override_thresholds(psr_accept=4.0):
        assert _should_escalate(diagnostics, "fast") is False


def test_override_changes_the_low_confidence_flag():
    """The flag reads the same live thresholds as the ladder."""
    diagnostics = Diagnostics(psr=5.0)

    assert is_low_confidence(0.9, diagnostics, threshold=0.5) is False
    with config.override_thresholds(psr_ambiguous=6.0):
        assert is_low_confidence(0.9, diagnostics, threshold=0.5) is True


def test_override_restores_on_exit():
    """A sweep step must not leak into the next one."""
    before = config.get_thresholds()

    with config.override_thresholds(psr_accept=99.0):
        assert config.get_thresholds().psr_accept == 99.0

    assert config.get_thresholds() == before


def test_override_restores_even_when_the_body_raises():
    """A sweep that dies partway must not mistune the rest of the process."""
    before = config.get_thresholds()

    with pytest.raises(RuntimeError), config.override_thresholds(psr_accept=99.0):
        raise RuntimeError("sweep step failed")

    assert config.get_thresholds() == before


def test_override_rejects_an_incoherent_combination():
    """Validation applies to overrides, not only to direct construction."""
    with pytest.raises(ValueError, match="psr_accept"), config.override_thresholds(psr_accept=1.0):
        pass  # pragma: no cover - construction raises before the body runs


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def test_lowering_the_accept_bar_stops_escalation(two_scale_pair):
    """Threshold tuning is the lever on runtime: a stopped ladder is cheaper."""
    reference, search, _ = two_scale_pair

    with config.override_thresholds(psr_accept=0.0, psr_ambiguous=0.0):
        stopped = localize(search, reference)

    assert stopped.diagnostics.mode_used == "fast"


def test_raising_the_accept_bar_forces_escalation(two_scale_pair):
    """The opposite lever, so a sweep can bracket the real operating point."""
    reference, search, _ = two_scale_pair

    with config.override_thresholds(psr_accept=1e9, psr_ambiguous=1e9):
        escalated = localize(search, reference)

    assert escalated.diagnostics.mode_used == "ambiguous"


def test_thresholds_do_not_break_the_never_raises_contract():
    """Even an extreme tuning must still return an answer."""
    with config.override_thresholds(psr_accept=1e9, psr_ambiguous=1e9):
        result = localize(np.zeros((40, 40)), np.zeros((10, 10)))

    assert np.isfinite(result.x)
    assert 0.0 <= result.confidence <= 1.0
