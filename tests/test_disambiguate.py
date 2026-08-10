"""Tests for R3 disambiguation: PSR and the mandated centre tie-break."""

import numpy as np

from src.disambiguate import peak_to_sidelobe, select_candidate
from src.types import Peak


def _noisy_surface(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0, 0.1, (50, 50)).astype(np.float32)


def test_sharp_peak_gives_high_psr() -> None:
    surface = _noisy_surface()
    surface[25, 25] = 1.0
    psr = peak_to_sidelobe(surface, Peak(col=25, row=25, score=1.0), 8)
    assert psr > 8.0


def test_flat_surface_returns_nan() -> None:
    surface = np.zeros((50, 50), dtype=np.float32)
    psr = peak_to_sidelobe(surface, Peak(col=25, row=25, score=0.0), 8)
    assert np.isnan(psr)


def test_edge_peak_does_not_crash() -> None:
    surface = _noisy_surface(seed=1)
    surface[0, 0] = 1.0
    psr = peak_to_sidelobe(surface, Peak(col=0, row=0, score=1.0), 8)
    assert np.isfinite(psr)


def test_clear_winner_no_tie_break() -> None:
    peaks = [
        Peak(col=10, row=10, score=0.9),
        Peak(col=80, row=80, score=0.3),
    ]
    chosen, tie_break_used = select_candidate(peaks, (50.0, 50.0), (10, 10), 0.05)
    assert chosen.col == 10
    assert tie_break_used is False


def test_tied_peaks_go_to_centre() -> None:
    # The 0.79 peak is nearest the centre and must win despite scoring
    # lower, because all three are within tolerance of the best.
    peaks = [
        Peak(col=90, row=90, score=0.80),
        Peak(col=52, row=52, score=0.79),
        Peak(col=10, row=10, score=0.80),
    ]
    chosen, tie_break_used = select_candidate(peaks, (50.0, 50.0), (10, 10), 0.05)
    assert (chosen.col, chosen.row) == (52, 52)
    assert tie_break_used is True
