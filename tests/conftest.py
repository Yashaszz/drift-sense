"""Shared synthetic fixtures.

Deliberately self-contained: these depend on nothing from R1's generator, so R4
can build and test the full matching path without waiting on anyone. They are
crude compared with the real dataset — no SEM physics, no edge brightening, no
Poisson noise — but they have *exact* ground truth, which is what correctness
tests need. Realism is R1's job and it feeds the accuracy numbers, not these.

Anchored and unanchored fields
------------------------------
``anchored=True`` adds smooth aperiodic texture on top of the lattice, so every
window is distinguishable and a correct answer exists. This is the right fixture
for testing that the matcher finds the right place.

``anchored=False`` is a bare lattice. A window taken from it matches *perfectly*
at every lattice-aligned position, so there is no unique answer and argmax is
arbitrary among the ties. That is not a defect to be fixed — it is the core
difficulty of the problem and the reason Stage 4 exists. Use it only to
demonstrate ambiguity, never to test localization accuracy.

Structure versus noise
----------------------
``structure_seed`` fixes the scene; ``noise_seed`` fixes one capture of it. The
reference and the search image are two independent captures of the *same* wafer,
so tests that model that pair must hold the structure seed constant and vary
only the noise seed. Varying both would be testing something the problem never
asks of us.
"""

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

SEARCH_SHAPE = (400, 400)
TEMPLATE_SIZE = 64
TRUE_COL = 250
TRUE_ROW = 120
STRUCTURE_SEED = 1234


def make_periodic_field(
    shape: tuple[int, int] = SEARCH_SHAPE,
    pitch: int = 16,
    *,
    noise: float = 0.0,
    noise_seed: int = 0,
    anchored: bool = True,
    structure_seed: int = STRUCTURE_SEED,
) -> np.ndarray:
    """Build a DRAM-like periodic field, optionally with aperiodic structure.

    Parameters
    ----------
    shape
        Image shape as ``(rows, cols)``.
    pitch
        Lattice period in pixels.
    noise
        Standard deviation of additive Gaussian noise, modelling one capture's
        independent sensor noise.
    noise_seed
        Seed for the noise draw. Vary this between the two captures of a pair.
    anchored
        When true, add smooth aperiodic texture so every window is unique. When
        false, return a bare lattice with genuinely ambiguous windows.
    structure_seed
        Seed for the aperiodic texture. Hold this constant across the two
        captures of a pair — it represents the wafer, not the capture.

    Returns
    -------
    numpy.ndarray
        ``float32`` image, roughly zero-mean and unit-scale.
    """
    rows, cols = shape
    grid_row, grid_col = np.mgrid[0:rows, 0:cols]
    field = np.sin(2 * np.pi * grid_col / pitch) * np.sin(2 * np.pi * grid_row / pitch)

    if anchored:
        # Smooth broadband texture, well below the lattice frequency. Stands in
        # for the array boundaries, contact pads and dummy-fill edges that break
        # periodicity in a real die.
        texture = gaussian_filter(
            np.random.default_rng(structure_seed).normal(0.0, 1.0, size=shape),
            sigma=4.0,
        )
        field = field + 0.9 * (texture / texture.std())

    if noise > 0.0:
        field = field + np.random.default_rng(noise_seed).normal(0.0, noise, size=shape)

    return field.astype(np.float32)


def crop(image: np.ndarray, col: int, row: int, size: int) -> np.ndarray:
    """Extract a square window by its top-left corner.

    Parameters
    ----------
    image
        Source image.
    col
        Left edge column.
    row
        Top edge row.
    size
        Window edge length in pixels.

    Returns
    -------
    numpy.ndarray
        Contiguous ``float32`` copy of the window.
    """
    return np.ascontiguousarray(image[row : row + size, col : col + size], dtype=np.float32)


def rescale_to_dtype(image: np.ndarray, dtype: np.dtype | type) -> np.ndarray:
    """Rescale a float image into the usable range of a target dtype.

    Parameters
    ----------
    image
        Source image.
    dtype
        Target dtype. Integer types are scaled into ``[0, 200]`` to stay clear of
        overflow at the top of an 8-bit range.

    Returns
    -------
    numpy.ndarray
        Rescaled array of the requested dtype.
    """
    span = float(image.max() - image.min())
    normalized = (image - image.min()) / (span if span > 0 else 1.0)
    if np.issubdtype(np.dtype(dtype), np.integer):
        return (normalized * 200.0).astype(dtype)
    return normalized.astype(dtype)


@pytest.fixture
def search_image() -> np.ndarray:
    """Return a clean anchored synthetic search image."""
    return make_periodic_field()


@pytest.fixture
def template_and_truth(search_image: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Return a template cropped from the search image, with its true position.

    Returns
    -------
    tuple
        ``(template, true_col, true_row)`` where the template was taken from
        ``search_image`` at exactly that top-left corner.
    """
    return (crop(search_image, TRUE_COL, TRUE_ROW, TEMPLATE_SIZE), TRUE_COL, TRUE_ROW)
