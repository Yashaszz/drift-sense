"""Stage 3b (T3) — behaviour tests for peak extraction and suppression.

Stage 3b is what turns a correlation surface into the shortlist Stage 4
disambiguates, so its contract is recall: the true position must survive into
the candidate list. These tests assert that, plus the separation guarantee that
makes the shortlist one-candidate-per-lattice-cell rather than fifty clustered
on a single peak.
"""

import numpy as np
import pytest

from src import matcher
from src.config import DEFAULT_NMS_RADIUS_PX
from src.types import Peak
from tests.conftest import TEMPLATE_SIZE, crop, make_periodic_field


def impulse_surface(
    shape: tuple[int, int],
    impulses: dict[tuple[int, int], float],
    background: float = 0.0,
) -> np.ndarray:
    """Build a surface with known impulses at known ``(col, row)`` positions."""
    surface = np.full(shape, background, dtype=np.float32)
    for (col, row), value in impulses.items():
        surface[row, col] = value
    return surface


def chebyshev(a: Peak, b: Peak) -> int:
    """Return the Chebyshev distance between two peaks."""
    return max(abs(a.col - b.col), abs(a.row - b.row))


# ---------------------------------------------------------------------------
# Basic extraction
# ---------------------------------------------------------------------------


def test_finds_a_single_impulse():
    surface = impulse_surface((100, 100), {(40, 70): 1.0})
    peaks = matcher.top_k_peaks(surface, k=10, nms_radius=4)
    assert len(peaks) == 1
    assert (peaks[0].col, peaks[0].row) == (40, 70)
    assert peaks[0].score == pytest.approx(1.0)


def test_finds_several_well_separated_impulses():
    positions = {(10, 10): 0.9, (50, 50): 0.7, (80, 20): 0.5}
    surface = impulse_surface((100, 100), positions)
    peaks = matcher.top_k_peaks(surface, k=10, nms_radius=4)

    assert len(peaks) == 3
    assert {(p.col, p.row) for p in peaks} == set(positions)


def test_peaks_are_sorted_by_descending_score():
    surface = impulse_surface((100, 100), {(10, 10): 0.3, (50, 50): 0.9, (80, 20): 0.6})
    peaks = matcher.top_k_peaks(surface, k=10, nms_radius=4)

    scores = [p.score for p in peaks]
    assert scores == sorted(scores, reverse=True)
    assert (peaks[0].col, peaks[0].row) == (50, 50)


def test_reported_score_matches_the_surface():
    surface = impulse_surface((60, 60), {(20, 30): 0.42, (45, 12): 0.17})
    for peak in matcher.top_k_peaks(surface, k=5, nms_radius=3):
        assert peak.score == pytest.approx(float(surface[peak.row, peak.col]))


def test_negative_scores_are_still_found():
    """ZNCC surfaces are signed; a weak match is not the same as no match."""
    surface = impulse_surface((80, 80), {(30, 40): -0.2}, background=-0.9)
    peaks = matcher.top_k_peaks(surface, k=3, nms_radius=4)
    assert (peaks[0].col, peaks[0].row) == (30, 40)
    assert peaks[0].score == pytest.approx(-0.2)


# ---------------------------------------------------------------------------
# Non-maximum suppression
# ---------------------------------------------------------------------------


def test_suppresses_neighbours_of_a_stronger_peak():
    surface = impulse_surface((100, 100), {(50, 50): 1.0, (52, 51): 0.8})
    peaks = matcher.top_k_peaks(surface, k=10, nms_radius=5)

    assert len(peaks) == 1
    assert (peaks[0].col, peaks[0].row) == (50, 50)


def test_keeps_peaks_just_outside_the_radius():
    surface = impulse_surface((100, 100), {(50, 50): 1.0, (56, 50): 0.8})
    peaks = matcher.top_k_peaks(surface, k=10, nms_radius=5)
    assert len(peaks) == 2


@pytest.mark.parametrize("radius", [1, 3, 5, 9, 15])
def test_returned_peaks_always_respect_the_radius(radius):
    rng = np.random.default_rng(4)
    surface = rng.random((200, 200), dtype=np.float32)
    peaks = matcher.top_k_peaks(surface, k=25, nms_radius=radius)

    for i, first in enumerate(peaks):
        for second in peaks[i + 1 :]:
            assert chebyshev(first, second) > radius


def test_larger_radius_returns_fewer_peaks():
    rng = np.random.default_rng(11)
    surface = rng.random((200, 200), dtype=np.float32)
    counts = [len(matcher.top_k_peaks(surface, k=1000, nms_radius=r)) for r in (2, 8, 20, 40)]
    assert counts == sorted(counts, reverse=True)


def test_radius_zero_returns_the_k_strongest_positions():
    surface = impulse_surface((50, 50), {(10, 10): 0.9, (11, 10): 0.8, (12, 10): 0.7})
    peaks = matcher.top_k_peaks(surface, k=3, nms_radius=0)
    assert [(p.col, p.row) for p in peaks] == [(10, 10), (11, 10), (12, 10)]


def test_suppression_is_square_not_circular():
    """A Chebyshev radius must suppress the diagonal corner of its own cell.

    An orthogonal lattice has square cells, so a Euclidean radius would leave
    the corners unsuppressed and admit two candidates from one cell.
    """
    surface = impulse_surface((60, 60), {(30, 30): 1.0, (34, 34): 0.9})
    peaks = matcher.top_k_peaks(surface, k=10, nms_radius=5)
    assert len(peaks) == 1


# ---------------------------------------------------------------------------
# k handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [1, 2, 5, 17])
def test_never_returns_more_than_k(k):
    rng = np.random.default_rng(2)
    surface = rng.random((300, 300), dtype=np.float32)
    assert len(matcher.top_k_peaks(surface, k=k, nms_radius=3)) <= k


def test_returns_fewer_than_k_rather_than_padding():
    surface = impulse_surface((100, 100), {(20, 20): 1.0, (70, 70): 0.5})
    peaks = matcher.top_k_peaks(surface, k=50, nms_radius=10)
    assert len(peaks) == 2


def test_k_of_one_returns_the_global_maximum():
    rng = np.random.default_rng(3)
    surface = rng.random((150, 150), dtype=np.float32)
    peak = matcher.top_k_peaks(surface, k=1, nms_radius=4)[0]

    row, col = np.unravel_index(int(np.argmax(surface)), surface.shape)
    assert (peak.col, peak.row) == (int(col), int(row))


# ---------------------------------------------------------------------------
# Degenerate surfaces
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [0.0, 1.0, -1.0, 0.37])
def test_flat_surface_yields_no_candidates(value):
    """No position is distinguishable, so inventing one would be dishonest."""
    surface = np.full((200, 200), value, dtype=np.float32)
    assert matcher.top_k_peaks(surface, k=30, nms_radius=8) == []


def test_flat_surface_is_fast():
    """Every position is technically a maximum; the short-circuit must catch it."""
    import time

    surface = np.zeros((901, 901), dtype=np.float32)
    started = time.perf_counter()
    matcher.top_k_peaks(surface, k=30, nms_radius=8)
    assert (time.perf_counter() - started) < 0.05


def test_plateau_yields_one_representative_not_the_whole_plateau():
    surface = np.zeros((100, 100), dtype=np.float32)
    surface[40:48, 40:48] = 1.0
    peaks = matcher.top_k_peaks(surface, k=30, nms_radius=8)

    assert len(peaks) == 1
    assert 40 <= peaks[0].col < 48
    assert 40 <= peaks[0].row < 48


def test_single_pixel_surface():
    surface = np.array([[0.5]], dtype=np.float32)
    assert matcher.top_k_peaks(surface, k=5, nms_radius=3) == []


def test_peaks_at_the_surface_border_are_found():
    """Border positions are valid matches and must not be filtered out."""
    for col, row in ((0, 0), (99, 0), (0, 99), (99, 99)):
        surface = impulse_surface((100, 100), {(col, row): 1.0})
        peaks = matcher.top_k_peaks(surface, k=5, nms_radius=6)
        assert (peaks[0].col, peaks[0].row) == (col, row)


def test_is_deterministic():
    rng = np.random.default_rng(5)
    surface = rng.random((200, 200), dtype=np.float32)
    first = matcher.top_k_peaks(surface, k=20, nms_radius=6)
    second = matcher.top_k_peaks(surface, k=20, nms_radius=6)
    assert first == second


def test_does_not_mutate_the_surface():
    rng = np.random.default_rng(6)
    surface = rng.random((120, 120), dtype=np.float32)
    before = surface.copy()
    matcher.top_k_peaks(surface, k=10, nms_radius=5)
    np.testing.assert_array_equal(surface, before)


# ---------------------------------------------------------------------------
# _local_maxima
# ---------------------------------------------------------------------------


def test_local_maxima_marks_the_impulse():
    surface = impulse_surface((50, 50), {(20, 30): 1.0})
    mask = matcher._local_maxima(surface, radius=3)
    assert mask[30, 20]


def test_local_maxima_radius_zero_marks_everything():
    rng = np.random.default_rng(8)
    surface = rng.random((30, 30), dtype=np.float32)
    assert matcher._local_maxima(surface, radius=0).all()


def test_local_maxima_is_boolean_and_shape_preserving():
    surface = np.zeros((40, 60), dtype=np.float32)
    mask = matcher._local_maxima(surface, radius=2)
    assert mask.shape == surface.shape
    assert mask.dtype == np.bool_


# ---------------------------------------------------------------------------
# Integration with Stage 3 — the recall contract
# ---------------------------------------------------------------------------


def test_true_position_survives_into_the_shortlist(search_image, template_and_truth):
    """R4 is measured on recall@K: the truth must reach Stage 4's shortlist."""
    template, true_col, true_row = template_and_truth
    surface = matcher.zncc_surface(template, search_image)
    peaks = matcher.top_k_peaks(surface, k=30, nms_radius=DEFAULT_NMS_RADIUS_PX)

    assert (true_col, true_row) in {(p.col, p.row) for p in peaks}
    assert (peaks[0].col, peaks[0].row) == (true_col, true_row)


@pytest.mark.parametrize("noise", [0.1, 0.3, 0.5])
def test_recall_holds_under_noise(noise):
    template = crop(make_periodic_field(noise=0.0), 250, 120, TEMPLATE_SIZE)
    surface = matcher.zncc_surface(template, make_periodic_field(noise=noise, noise_seed=42))
    peaks = matcher.top_k_peaks(surface, k=30, nms_radius=DEFAULT_NMS_RADIUS_PX)

    assert (250, 120) in {(p.col, p.row) for p in peaks}


def test_unanchored_field_is_an_information_theoretic_failure():
    """With no anchor in frame, correlation cannot identify the position at all.

    Hundreds of positions score *exactly* the maximum, so which of them reaches
    a K-length shortlist is decided by iteration order, not by evidence. Recall
    is a lottery, and no change to peak extraction improves it — the data does
    not contain the answer.

    This is the distinction the failure analysis turns on: an
    information-theoretic failure, not an algorithmic one. Measured here rather
    than asserted in prose, so the writeup can cite a number.
    """
    bare = make_periodic_field(anchored=False)
    template = crop(bare, 250, 120, TEMPLATE_SIZE)
    surface = matcher.zncc_surface(template, bare)

    # The true position scores exactly as well as the best candidate...
    assert surface[120, 250] == pytest.approx(float(surface.max()), abs=1e-6)

    # ...and so do hundreds of others, so the maximum identifies nothing.
    exact_ties = int(np.sum(surface >= surface.max() - 1e-6))
    assert exact_ties > 500

    # Every shortlisted candidate is indistinguishable from every other.
    peaks = matcher.top_k_peaks(surface, k=50, nms_radius=DEFAULT_NMS_RADIUS_PX)
    assert len(peaks) == 50
    assert all(p.score == pytest.approx(float(surface.max()), abs=1e-6) for p in peaks)


def test_correlation_ties_can_be_closer_together_than_the_layout_pitch():
    """The NMS radius must track the *correlation* period, not the layout pitch.

    A separable ``sin * sin`` lattice of pitch p is unchanged by a half-pitch
    shift in both axes at once, because both factors flip sign and the product
    does not. Its correlation ties therefore sit p/2 apart on the diagonal, not
    p. Deriving the suppression radius from the layout pitch alone would
    suppress genuine lattice-distinct candidates.

    Recorded as a test because it is the kind of half-pitch reasoning error that
    is invisible until accuracy is mysteriously capped.
    """
    bare = make_periodic_field(anchored=False, pitch=16)
    template = crop(bare, 250, 120, TEMPLATE_SIZE)
    surface = matcher.zncc_surface(template, bare)

    tied = np.argwhere(surface >= surface.max() - 1e-6)
    distances = np.max(np.abs(tied - np.array([120, 250])), axis=1)
    nearest = int(np.min(distances[distances > 0]))

    assert nearest == 8


def test_lattice_pitch_radius_returns_one_candidate_per_cell():
    """The point of tying the radius to the pitch: no duplicates from one cell."""
    bare = make_periodic_field(anchored=False, pitch=16)
    template = crop(bare, 250, 120, TEMPLATE_SIZE)
    surface = matcher.zncc_surface(template, bare)
    peaks = matcher.top_k_peaks(surface, k=40, nms_radius=8)

    for i, first in enumerate(peaks):
        for second in peaks[i + 1 :]:
            assert chebyshev(first, second) > 8
