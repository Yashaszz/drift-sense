"""Diffraction-limited optical imaging of a layout, for the RGB bonus.

Owner: R1. This is a *generator* module, not an inference one: it renders the
same layouts through a visible-light optic instead of the SEM chain, so the
optical-bonus pairs can be scored by the existing harness. ``src.evaluate``
already collapses three channels to luminance, so nothing downstream changes.

What this module is for
-----------------------
The honest headline is that **visible light cannot resolve this layout**, and
that is the finding rather than a problem to engineer around.

Incoherent imaging through a circular aperture passes no spatial frequency above
``f_c = 2 NA / lambda``, so the finest period that survives at all is
``lambda / (2 NA)`` -- the Abbe limit. At NA 0.95 that is 237 nm in blue and
337 nm in red. Against those numbers:

===========================  =================  ==============================
Feature                      Period             Survives?
===========================  =================  ==============================
FinFET fin pitch             72-108 nm          No, at any wavelength or NA
DRAM pitch                   144-216 nm         No at NA 0.95; marginal at 1.35
FinFET gate pitch            336-504 nm         Blue and green yes, red marginal
===========================  =================  ==============================

So an optical capture of a DRAM field is a uniform grey, and a FinFET field
retains only its coarse gate lattice. That is not a modelling shortcut; it is
why the problem is posed with an SEM in the first place, and it is measurable:
the blue channel carries strictly more structure than the red one, because
``f_c`` scales as ``1 / lambda``.

Deliberately not a Gaussian
---------------------------
A Gaussian blur has infinite support in frequency, so it *attenuates* fine
structure but never removes it -- a matcher could still lock onto a lattice at
0.1% contrast that a real optic would not deliver at all. The circular-aperture
OTF is exactly zero above cutoff, which is the property that makes the "cannot
resolve" claim true rather than rhetorical. It costs one extra ``arccos``.

Coordinate contract
-------------------
Like :func:`src.sem_physics.apply_sem_chain`, this only changes intensities. It
never resamples, translates, rotates or crops, so R1's pixel-centre ground truth
is unchanged and the same ``ground_truth.jsonl`` scores both modalities.
"""

from typing import Final, cast

import numpy as np

from src.types import FloatArray

__all__ = [
    "CHANNEL_WAVELENGTHS_NM",
    "DEFAULT_NA",
    "cutoff_period_nm",
    "diffraction_mtf",
    "render_rgb",
]

CHANNEL_WAVELENGTHS_NM: Final[dict[str, float]] = {"r": 640.0, "g": 550.0, "b": 450.0}
"""Representative centre wavelengths for the three channels, in nanometres.

Band centres rather than a measured filter response: the point of the RGB bonus
is the *ordering* -- blue resolves finer than red because cutoff goes as
``1 / lambda`` -- and that ordering is insensitive to the exact centres. Anyone
modelling a specific camera should replace these with its filter curves.
"""

DEFAULT_NA: Final[float] = 0.95
"""Numerical aperture of the modelled objective.

0.95 is about the practical ceiling for a dry objective; oil immersion reaches
roughly 1.4. Stated rather than derived -- it is the one free parameter here,
and the conclusion is not sensitive to it: even at NA 1.35 the Abbe limit is
167 nm in blue, still coarser than the FinFET fin pitch this dataset uses.
"""


def cutoff_period_nm(wavelength_nm: float, na: float = DEFAULT_NA) -> float:
    """Return the finest period that survives the optic, in nanometres.

    Parameters
    ----------
    wavelength_nm
        Illumination wavelength.
    na
        Numerical aperture.

    Returns
    -------
    float
        ``lambda / (2 NA)``. Periods finer than this are transmitted with
        exactly zero contrast, not merely attenuated.
    """
    if wavelength_nm <= 0 or na <= 0:
        msg = f"wavelength and NA must be positive, got {wavelength_nm!r} and {na!r}"
        raise ValueError(msg)
    return float(wavelength_nm / (2.0 * na))


def diffraction_mtf(
    frequencies_per_nm: FloatArray,
    wavelength_nm: float,
    na: float = DEFAULT_NA,
) -> FloatArray:
    """Return the incoherent OTF of a circular aperture at each frequency.

    Parameters
    ----------
    frequencies_per_nm
        Radial spatial frequency, in cycles per nanometre.
    wavelength_nm
        Illumination wavelength.
    na
        Numerical aperture.

    Returns
    -------
    FloatArray
        Modulation transfer in ``[0, 1]``, exactly zero above cutoff.

    Notes
    -----
    For incoherent illumination the OTF is the autocorrelation of the pupil,
    which for a circular pupil has the closed form

    ``H(x) = (2/pi) [ arccos(x) - x sqrt(1 - x^2) ]``, ``x = f / f_c``

    with ``f_c = 2 NA / lambda``. The hard zero above cutoff is the whole point:
    see the module docstring.
    """
    cutoff = 2.0 * na / wavelength_nm
    x = np.clip(np.asarray(frequencies_per_nm, dtype=np.float64) / cutoff, 0.0, 1.0)
    mtf = (2.0 / np.pi) * (np.arccos(x) - x * np.sqrt(np.maximum(0.0, 1.0 - x * x)))
    return np.asarray(mtf, dtype=np.float32)


def _apply_channel(image: FloatArray, px_nm: float, wavelength_nm: float, na: float) -> FloatArray:
    """Filter one channel through the diffraction-limited optic."""
    rows, cols = image.shape
    fy = np.fft.fftfreq(rows, d=px_nm)
    fx = np.fft.fftfreq(cols, d=px_nm)
    radial = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)

    filtered = np.fft.ifft2(np.fft.fft2(image) * diffraction_mtf(radial, wavelength_nm, na)).real
    return cast(FloatArray, np.clip(filtered, 0.0, 1.0).astype(np.float32))


def render_rgb(
    image: FloatArray,
    px_nm: float,
    *,
    na: float = DEFAULT_NA,
    wavelengths_nm: dict[str, float] | None = None,
) -> FloatArray:
    """Render a geometry-only image as it would appear through a visible optic.

    Parameters
    ----------
    image
        Geometry-only greyscale image in ``[0, 1]``, as produced by
        :func:`src.render.rasterize`.
    px_nm
        Sampling pitch of this capture, in nanometres.
    na
        Numerical aperture of the objective.
    wavelengths_nm
        Channel centres; defaults to :data:`CHANNEL_WAVELENGTHS_NM`. Order of
        the output channels follows ``("r", "g", "b")``.

    Returns
    -------
    FloatArray
        ``(rows, cols, 3)`` float32 in ``[0, 1]``.

    Notes
    -----
    Each channel is filtered independently, so the channels differ only through
    cutoff. Blue keeps the most structure and red the least, which is the
    measurable claim this module exists to support. Nothing here is stochastic:
    the optic is deterministic, and any noise belongs to the capture chain
    rather than to diffraction.
    """
    if image.ndim != 2:
        msg = f"image must be 2-D greyscale, got shape {image.shape!r}"
        raise ValueError(msg)
    if px_nm <= 0 or not np.isfinite(px_nm):
        msg = f"px_nm must be finite and positive, got {px_nm!r}"
        raise ValueError(msg)

    bands = CHANNEL_WAVELENGTHS_NM if wavelengths_nm is None else wavelengths_nm
    channels = [_apply_channel(image, px_nm, bands[name], na) for name in ("r", "g", "b")]
    return np.stack(channels, axis=-1).astype(np.float32)
