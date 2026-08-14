"""Diffraction-limited optics for the RGB bonus.

The claim these tests defend is unusual: that the optic removes structure
entirely rather than attenuating it. A blur that merely attenuates would let a
matcher lock onto a lattice at 0.1% contrast that a real objective would never
deliver, and the bonus would report an accuracy number for something physically
unobservable. So the hard zero above cutoff is the property under test, not an
implementation detail.
"""

import numpy as np
import pytest

from src.optical import (
    CHANNEL_WAVELENGTHS_NM,
    DEFAULT_NA,
    cutoff_period_nm,
    diffraction_mtf,
    render_rgb,
)

FIELD_PX = 1024
"""Field width. Gratings are placed at integer cycle counts across it."""


def _grating(cycles: int, size: int = FIELD_PX) -> np.ndarray:
    """Return a sinusoidal grating with exactly ``cycles`` periods across the field.

    Integer cycles matter: a non-integer count is not periodic in the window, and
    the resulting spectral leakage puts energy at frequencies the optic passes,
    which reads as contrast surviving below cutoff when it has not.
    """
    x = np.arange(size)
    row = 0.5 + 0.5 * np.sin(2.0 * np.pi * cycles * x / size)
    return np.tile(row.astype(np.float32), (size, 1))


def _contrast(channel: np.ndarray) -> float:
    """Michelson contrast of one row."""
    hi, lo = float(channel.max()), float(channel.min())
    return (hi - lo) / (hi + lo + 1e-12)


def test_cutoff_matches_the_abbe_limit():
    """The finest surviving period must be lambda / (2 NA)."""
    assert cutoff_period_nm(550.0, 0.95) == pytest.approx(289.47, abs=0.01)
    assert cutoff_period_nm(450.0, 1.35) == pytest.approx(166.67, abs=0.01)


def test_mtf_is_one_at_dc_and_zero_above_cutoff():
    """The transfer function must be normalised, and must actually reach zero."""
    cutoff = 2.0 * DEFAULT_NA / 550.0

    assert float(diffraction_mtf(np.array([0.0]), 550.0)[0]) == pytest.approx(1.0)
    assert float(diffraction_mtf(np.array([cutoff * 1.01]), 550.0)[0]) == 0.0
    assert float(diffraction_mtf(np.array([cutoff * 10.0]), 550.0)[0]) == 0.0


def test_mtf_matches_the_closed_form():
    """The numerical filter must agree with the analytic circular-aperture OTF.

    A Gaussian would pass every other test in this file approximately. This one
    it fails, which is the point.
    """
    period = FIELD_PX / 2.0  # 512 nm at 1 nm/px, comfortably above cutoff
    rendered = render_rgb(_grating(2), px_nm=1.0)
    analytic = float(diffraction_mtf(np.array([1.0 / period]), CHANNEL_WAVELENGTHS_NM["g"])[0])

    assert _contrast(rendered[FIELD_PX // 2, :, 1]) == pytest.approx(analytic, abs=1e-3)


@pytest.mark.parametrize("cycles", [14, 10, 8, 6, 5])
def test_structure_below_cutoff_is_removed_not_attenuated(cycles):
    """Every period finer than the blue cutoff must vanish in all three channels.

    At 1 nm/px over a 1024 nm field these are 73 to 205 nm periods, which spans
    the FinFET fin pitch (72-108 nm) and the whole DRAM pitch range (144-216 nm).
    An optical capture of those fields is uniform grey, and that is the finding
    the bonus reports rather than a defect to tune around.
    """
    period = FIELD_PX / cycles
    assert period < cutoff_period_nm(CHANNEL_WAVELENGTHS_NM["b"])

    rendered = render_rgb(_grating(cycles), px_nm=1.0)

    for channel in range(3):
        assert _contrast(rendered[FIELD_PX // 2, :, channel]) == pytest.approx(0.0, abs=1e-9)


def test_blue_resolves_strictly_more_than_red():
    """Cutoff scales as 1 / lambda, so the channels must order b > g > r.

    This ordering is the measurable content of the RGB bonus: three channels
    that behaved identically would make the dataset a greyscale one in disguise.
    """
    rendered = render_rgb(_grating(3), px_nm=1.0)  # 341 nm period
    red, green, blue = (_contrast(rendered[FIELD_PX // 2, :, c]) for c in range(3))

    assert blue > green > red


def test_geometry_is_untouched():
    """The optic changes intensities only, so ground truth stays valid.

    Same contract as ``apply_sem_chain``: no resampling, translation or crop,
    which is what lets one ground_truth.jsonl score both modalities.
    """
    rng = np.random.default_rng(0)
    image = rng.random((64, 64)).astype(np.float32)

    rendered = render_rgb(image, px_nm=1.0)

    assert rendered.shape == (64, 64, 3)
    assert rendered.dtype == np.float32
    assert np.all(np.isfinite(rendered))
    assert float(rendered.min()) >= 0.0
    assert float(rendered.max()) <= 1.0
    # A uniform field must survive unchanged: the OTF is normalised at DC, so
    # mean intensity is preserved even when every spatial frequency is removed.
    flat = render_rgb(np.full((64, 64), 0.42, dtype=np.float32), px_nm=1.0)
    assert flat == pytest.approx(0.42, abs=1e-5)
