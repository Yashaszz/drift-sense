"""Stage 1 — rotation and scale estimation from the Fourier domain.

Owned by R2. This file is a stub placed by R4 so that the pipeline imports and
runs end-to-end; R2 replaces the body without changing the signature.

The idea
--------
By the Fourier shift theorem, translation changes only the *phase* of the
spectrum — the magnitude is unchanged. So ``|FFT|`` does not depend on where the
match is. Rotating the image rotates its spectrum by the same angle, and scaling
by ``s`` scales the spectrum by ``1/s``. Under a log-polar remap both become
pure translations, which phase correlation recovers directly.

Rotation and scale are therefore solved **without knowing position**, collapsing
a four-dimensional search into two cheap two-dimensional ones. A sharper lattice
produces sharper spectral peaks, so the cases that are hardest to localize are
the easiest to pose: the periodicity that causes the core difficulty is turned
into the thing that resolves pose.

This is the Fourier-Mellin transform, a standard and citable registration
technique.

Notes for the implementer
-------------------------
Four caveats from the design discussion, all to be handled explicitly:

- **180 degree ambiguity.** The magnitude spectrum of a real image is
  centrosymmetric, so ``theta`` and ``theta + 180`` are indistinguishable here.
  Both hypotheses go forward and Stage 3 keeps the better correlation.
- **90 degree ambiguity.** A square DRAM grid is symmetric under quarter turns.
- **FinFET anisotropy.** A fin field gives strong spectral peaks along one axis
  and weak ones along the other; the estimate stays valid but noisier in the
  weak direction, so the search band should widen accordingly.
- **Fallback.** When spectral peaks are too weak, drop to a bounded grid search
  over ``(theta, scale)``. We estimate a *residual* around the nominal ratio,
  never a scale from scratch.

Status
------
Stub. Returns the nominal pose with zero quality, which is exactly what ``fast``
mode assumes, so callers behave sensibly until the real estimator lands.
"""

from src import config
from src.types import FloatArray, PoseEstimate

__all__ = ["estimate_pose"]


def estimate_pose(
    reference: FloatArray,
    search: FloatArray,
    nominal_scale: float = config.NOMINAL_SCALE,
) -> PoseEstimate:
    """Estimate the rotation and scale relating the reference to the search image.

    Parameters
    ----------
    reference
        Reference image at 1 nm/px, normalized, as ``(rows, cols)``.
    search
        Search image at 10 nm/px, normalized, as ``(rows, cols)``.
    nominal_scale
        Expected decimation ratio. The estimator refines a residual around this
        value rather than searching scale from scratch.

    Returns
    -------
    PoseEstimate
        Rotation in degrees, total scale (not the residual), and a quality
        metric in ``[0, 1]``. A low quality means the spectral peaks were too
        weak to trust and the caller should fall back to nominal pose or to a
        bounded grid search.

    Notes
    -----
    Never raises. Pose estimation failing is an expected outcome on a highly
    aperiodic reference; it is reported through ``quality``, not an exception,
    because :func:`src.localize.localize` must not propagate errors.
    """
    del reference, search  # stub: Fourier/log-polar estimation is R2's work
    return PoseEstimate(theta_deg=0.0, scale=nominal_scale, quality=0.0)
