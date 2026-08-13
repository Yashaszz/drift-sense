"""Process-wide reuse of R3's uniqueness map.

Scoring a reference is 68% of a ``localize`` call at production shapes and
depends on the reference alone. These tests pin the two properties that make
caching it safe: equal references must hit, and different references must miss.
A false hit would silently weight one site's correlation by another site's
anchors, which no downstream assertion would catch.
"""

import numpy as np
import pytest

from src import config
from src import localize as localize_module
from src.localize import clear_uniqueness_cache, localize


@pytest.fixture(autouse=True)
def _cold_cache():
    """Every test starts and finishes with an empty cache."""
    clear_uniqueness_cache()
    yield
    clear_uniqueness_cache()


def _count_uniqueness_calls(monkeypatch):
    """Wrap the uniqueness map so tests can count real computations."""
    calls = []
    original = localize_module.disambiguate.uniqueness_map

    def counting(reference, *args, **kwargs):
        calls.append(reference.shape)
        return original(reference, *args, **kwargs)

    monkeypatch.setattr(localize_module.disambiguate, "uniqueness_map", counting)
    return calls


# ---------------------------------------------------------------------------
# Reuse
# ---------------------------------------------------------------------------


def test_repeat_call_reuses_the_map(two_scale_pair, monkeypatch):
    """The same reference twice computes the map once."""
    reference, search, _ = two_scale_pair
    calls = _count_uniqueness_calls(monkeypatch)

    localize(search, reference)
    after_first = len(calls)
    localize(search, reference)

    assert after_first >= 1
    assert len(calls) == after_first


def test_an_equal_but_distinct_array_hits(two_scale_pair, monkeypatch):
    """Keying is by content, not identity.

    A caller reading the same site from disk twice gets two distinct arrays.
    Keying on ``id`` would miss every time and the cache would never pay off.
    """
    reference, search, _ = two_scale_pair
    calls = _count_uniqueness_calls(monkeypatch)

    localize(search, reference)
    after_first = len(calls)
    localize(search, reference.copy())

    assert len(calls) == after_first


def test_a_different_reference_misses(two_scale_pair, monkeypatch):
    """Different content must not collide onto one entry."""
    reference, search, _ = two_scale_pair
    other = np.ascontiguousarray(reference[::-1, :])
    calls = _count_uniqueness_calls(monkeypatch)

    localize(search, reference)
    after_first = len(calls)
    localize(search, other)

    assert len(calls) > after_first


def test_a_substituted_implementation_is_not_served_the_real_map(two_scale_pair, monkeypatch):
    """The reference does not determine the map on its own; the scorer does.

    ``benchmarks/verify_uniqueness_integration.py`` swaps in a stand-in map to
    measure what R3's stage is worth. Keying on content alone would hand that
    run the real map it had already cached and report the two as identical —
    silently invalidating the comparison rather than failing it.
    """
    reference, search, _ = two_scale_pair

    localize(search, reference)  # populate with the real implementation

    stand_in = np.linspace(0.0, 1.0, reference.size, dtype=np.float32).reshape(reference.shape)
    monkeypatch.setattr(
        localize_module.disambiguate,
        "uniqueness_map",
        lambda ref, *a, **k: stand_in,
    )

    cache = localize_module._StageCache(
        np.ascontiguousarray(search, dtype=np.float32),
        np.ascontiguousarray(reference, dtype=np.float32),
    )
    assert cache.uniqueness() is stand_in


def test_the_tile_size_is_part_of_the_key(two_scale_pair, monkeypatch):
    """A sweep over tile size must not be served one tile size's answer."""
    reference, search, _ = two_scale_pair
    calls = _count_uniqueness_calls(monkeypatch)

    localize(search, reference)
    after_first = len(calls)
    monkeypatch.setattr(config, "DEFAULT_UNIQUENESS_TILE_PX", 32)
    localize(search, reference)

    assert len(calls) > after_first


def test_reuse_does_not_change_the_answer(two_scale_pair):
    """A cache hit must be indistinguishable from a cold computation."""
    reference, search, _ = two_scale_pair

    cold = localize(search, reference)
    warm = localize(search, reference)

    assert (cold.x, cold.y) == (warm.x, warm.y)
    assert cold.confidence == warm.confidence
    assert cold.diagnostics.psr == pytest.approx(warm.diagnostics.psr, nan_ok=True)
    assert cold.diagnostics.uniqueness_score == pytest.approx(
        warm.diagnostics.uniqueness_score, nan_ok=True
    )


# ---------------------------------------------------------------------------
# Bounds and the disable switch
# ---------------------------------------------------------------------------


def test_cache_is_bounded(two_scale_pair, monkeypatch):
    """The map is reference-sized, so an unbounded cache would leak memory."""
    reference, search, _ = two_scale_pair
    monkeypatch.setattr(config, "UNIQUENESS_CACHE_ENTRIES", 2)

    for shift in range(4):
        localize(search, np.ascontiguousarray(np.roll(reference, shift, axis=0)))

    assert len(localize_module._UNIQUENESS_CACHE) <= 2


def test_eviction_is_least_recently_used(two_scale_pair, monkeypatch):
    """A reference kept warm survives; the idle one is evicted."""
    reference, search, _ = two_scale_pair
    second = np.ascontiguousarray(np.roll(reference, 3, axis=0))
    third = np.ascontiguousarray(np.roll(reference, 7, axis=0))
    monkeypatch.setattr(config, "UNIQUENESS_CACHE_ENTRIES", 2)
    # Installed before the warming calls: the implementation is part of the
    # cache key, so swapping it partway would itself invalidate every entry.
    calls = _count_uniqueness_calls(monkeypatch)

    localize(search, reference)
    localize(search, second)
    localize(search, reference)  # refresh the first
    localize(search, third)  # evicts the idle second

    settled = len(calls)
    localize(search, reference)
    assert len(calls) == settled, "the refreshed reference should have survived"

    localize(search, second)
    assert len(calls) > settled, "the idle reference should have been evicted"


def test_zero_entries_disables_the_cache(two_scale_pair, monkeypatch):
    """Benchmarks measuring a cold pipeline need the cache out of the way."""
    reference, search, _ = two_scale_pair
    monkeypatch.setattr(config, "UNIQUENESS_CACHE_ENTRIES", 0)
    calls = _count_uniqueness_calls(monkeypatch)

    localize(search, reference)
    after_first = len(calls)
    localize(search, reference)

    assert len(calls) > after_first
    assert not localize_module._UNIQUENESS_CACHE


def test_clear_forces_recomputation(two_scale_pair, monkeypatch):
    """The documented way for a benchmark to get a cold measurement."""
    reference, search, _ = two_scale_pair
    calls = _count_uniqueness_calls(monkeypatch)

    localize(search, reference)
    after_first = len(calls)
    clear_uniqueness_cache()
    localize(search, reference)

    assert len(calls) > after_first
