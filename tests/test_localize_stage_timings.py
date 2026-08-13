"""Per-stage timing instrumentation.

Ad-hoc profiling is how the uniqueness map was identified as 68% of runtime, but
a cProfile run is not something a tuning sweep or a regression check can consume.
This makes the same breakdown available from an ordinary ``localize`` call,
without costing the shipped path anything when it is off.
"""

import numpy as np
import pytest

from src import config
from src.localize import clear_uniqueness_cache, localize


@pytest.fixture(autouse=True)
def _cold_cache():
    """Timings are meaningless if a previous test already warmed the cache."""
    clear_uniqueness_cache()
    yield
    clear_uniqueness_cache()


@pytest.fixture
def _timed(monkeypatch):
    """Enable collection for one test."""
    monkeypatch.setattr(config, "COLLECT_STAGE_TIMINGS", True)


# ---------------------------------------------------------------------------
# Off by default
# ---------------------------------------------------------------------------


def test_collection_is_off_by_default():
    """A tool making thousands of moves a day should not be timing itself."""
    assert config.COLLECT_STAGE_TIMINGS is False


def test_no_timings_are_recorded_when_disabled(two_scale_pair):
    """The shipped path pays a boolean test and nothing else."""
    reference, search, _ = two_scale_pair

    assert localize(search, reference).diagnostics.stage_ms == {}


def test_enabling_collection_does_not_change_the_answer(two_scale_pair, monkeypatch):
    """Instrumentation must be observation, not participation."""
    reference, search, _ = two_scale_pair

    plain = localize(search, reference)
    monkeypatch.setattr(config, "COLLECT_STAGE_TIMINGS", True)
    timed = localize(search, reference)

    assert (plain.x, plain.y) == (timed.x, timed.y)
    assert plain.confidence == timed.confidence
    assert plain.diagnostics.psr == pytest.approx(timed.diagnostics.psr, nan_ok=True)
    assert plain.low_confidence_flag == timed.low_confidence_flag


# ---------------------------------------------------------------------------
# What it records
# ---------------------------------------------------------------------------


def test_the_expensive_stages_are_covered(two_scale_pair, _timed):
    """The breakdown must name the stages an optimisation would target."""
    reference, search, _ = two_scale_pair

    stages = localize(search, reference).diagnostics.stage_ms

    assert {"correlate", "select_candidate", "refine_subpixel"} <= stages.keys()


def test_every_timing_is_a_finite_non_negative_float(two_scale_pair, _timed):
    """These get averaged and tabulated; a NaN would poison a whole column."""
    stages = localize(search=two_scale_pair[1], reference=two_scale_pair[0]).diagnostics.stage_ms

    assert stages
    for name, value in stages.items():
        assert isinstance(value, float), name
        assert np.isfinite(value), name
        assert value >= 0.0, name


def test_stage_time_does_not_exceed_the_call(two_scale_pair, _timed):
    """Stages are timed inside the call, so no stage can outlast it.

    Guards against the accumulator being shared across calls, which would show
    up as a stage total that grows without bound.
    """
    result = localize(search=two_scale_pair[1], reference=two_scale_pair[0])

    assert max(result.diagnostics.stage_ms.values()) <= result.diagnostics.elapsed_ms


def test_timings_do_not_leak_between_calls(two_scale_pair, _timed):
    """Each call reports its own cost, not a running total for the process."""
    reference, search, _ = two_scale_pair

    first = localize(search, reference).diagnostics.stage_ms["correlate"]
    second = localize(search, reference).diagnostics.stage_ms["correlate"]

    assert second < first * 10.0


def test_the_breakdown_accumulates_across_tiers(two_scale_pair, _timed):
    """A tier that escalates must not discard the cheaper tier's cost.

    The record is rebuilt per tier, so the accumulator is deliberately shared;
    without that, the breakdown would report only whichever tier answered.
    """
    reference, search, _ = two_scale_pair

    with config.override_thresholds(psr_accept=1e9, psr_ambiguous=1e9):
        escalated = localize(search, reference)

    assert escalated.diagnostics.mode_used == "ambiguous"
    assert escalated.diagnostics.stage_ms["pose"] > 0.0
    assert set(escalated.diagnostics.stage_ms) >= {"pose", "psf_sigma", "correlate"}


def test_a_failed_call_still_reports_what_it_spent(_timed):
    """Failure analysis needs the breakdown most, so it must survive the fallback."""
    result = localize(np.zeros((5, 5)), np.zeros((10, 10)))

    assert result.diagnostics.failure_mode != "none"
    assert np.isfinite(result.diagnostics.elapsed_ms)
