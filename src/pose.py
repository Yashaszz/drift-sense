"""
Stage 1 — Local Fourier-Mellin pose estimation.

Owned by R2.

Estimates:
    - rotation
    - residual scale

The returned scale is:

    total_scale = nominal_scale * residual_scale

Pipeline:
    1. preprocess images
    2. estimate rotation/scale in Fourier-log-polar space
    3. accept a bounded spectral rotation only when its peak is credible
    4. otherwise use the nominal identity residual pose
    5. keep residual scale fixed at 1.0 because nominal_scale is authoritative

Spectral only, by design
------------------------
Rotation is read straight off the log-polar correlation peak, and the estimate
is therefore quantised to one row of that surface: 360/512 = **0.703125 deg**.
Every recorded estimate across the tracked runs sits exactly on that grid, and
a zero-rotation pair typically reports -0.703 rather than 0.000. It costs
little because the template is rebuilt at the estimated angle either way, but
it is a real floor on rotation accuracy and it is quoted as a limitation rather
than hidden.

There is deliberately **no spatial refinement or spatial fallback**. Both were
tried and both scored candidate angles by NCC between the reference and the
search resized to a common square -- a 1 um field against a 10 um field, whose
lattices sit a decade apart in frequency, so the score carries no content
correspondence and its argmax is noise. Closing the 0.703 deg gap needs
sub-bin interpolation of the log-polar peak itself, or a spatial score computed
against the *decimated* template rather than the raw reference. See
docs/failure_analysis.md.

Public API is frozen:
    estimate_pose(reference, search, nominal_scale)

Never raises.

Known limitation — rotation is quantised to the log-polar bin
------------------------------------------------------------
The estimate is read off a 512-row log-polar correlation surface, so it can only
take multiples of 360/512 = 0.703125 deg. On the 324-pair set that is exactly
what it does: every ``theta_est`` in ``results/full_324.csv`` is an exact bin
multiple, spread over 24 distinct values, and the rotation error carries a
systematic negative bias of -0.50 deg mean / -0.62 deg median.

A spatial-NCC refiner that would resolve sub-bin angles was written and never
wired in. It is deleted rather than left dormant: it was unreachable, so it was
never executed or tested, and the accuracy it claims is unmeasured. Enabling it
would move every published figure, which is a change to make with a
re-measurement rather than late.

The next step is that refinement, or a parabolic interpolation across the three
rows around the peak, measured against the same 324 pairs before anything
downstream quotes a new number.
"""

from __future__ import annotations

import logging
from typing import cast

import numpy as np
from scipy import ndimage
from skimage.transform import resize, warp_polar

from src import config
from src.types import FloatArray, PoseEstimate

__all__ = ["estimate_pose"]

logger = logging.getLogger(__name__)


# The real-data generator uses rotations in [-8, +8] degrees.
# Wider Fourier-Mellin hypotheses on periodic, translated crops are aliases.
_MAX_DATASET_ROTATION_DEG = 10.0
_MIN_FM_QUALITY = 0.20


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------


def _normalize_angle(angle: float) -> float:
    """Normalize angle to [-180, 180)."""
    return float((angle + 180.0) % 360.0 - 180.0)


def _prepare(image: FloatArray, size: int) -> np.ndarray:
    """
    Prepare an image for Fourier-Mellin/NCC processing.

    Steps:
        - convert to float32
        - replace NaNs/Infs
        - normalize intensity
        - resize to common square size
        - remove slow illumination variation
        - apply Hann window
    """
    x = np.asarray(image, dtype=np.float32)

    if x.ndim != 2:
        raise ValueError("image must be 2-D")

    if min(x.shape) < 8:
        raise ValueError("image too small")

    finite = np.isfinite(x)

    if not finite.any():
        raise ValueError("image contains no finite values")

    median = float(np.nanmedian(x[finite]))

    x = np.where(finite, x, median)

    std = float(x.std())

    if std < 1e-7:
        raise ValueError("constant image")

    x = (x - float(x.mean())) / (std + 1e-7)

    if x.shape != (size, size):
        # scikit-image ships only partial annotations, so this call is untyped
        # as far as mypy is concerned. The cast pins what it actually returns.
        resized = cast(
            np.ndarray,
            resize(  # type: ignore[no-untyped-call]
                x,
                (size, size),
                order=1,
                mode="reflect",
                anti_aliasing=True,
                preserve_range=True,
            ),
        )
        x = resized.astype(np.float32)

    # Remove low-frequency illumination variation.
    low = ndimage.gaussian_filter(
        x,
        sigma=2.0,
    )

    x = x - low

    # Reduce FFT boundary leakage.
    hann_1d = np.hanning(size).astype(np.float32)

    window = np.outer(
        hann_1d,
        hann_1d,
    )

    # `x` is a bare `np.ndarray`, so `.astype` on the product is typed `Any`.
    # `np.asarray` with an explicit dtype carries the annotation through; the
    # product is already float32, so this is the same array, not a copy.
    return np.asarray(
        x * window,
        dtype=np.float32,
    )


def _fft_log_magnitude(image: np.ndarray) -> np.ndarray:
    """Return centered logarithmic FFT magnitude."""
    spectrum = np.fft.fftshift(np.fft.fft2(image))

    magnitude = np.abs(spectrum)

    return np.asarray(
        np.log1p(magnitude),
        dtype=np.float32,
    )


def _log_polar(magnitude: np.ndarray) -> np.ndarray:
    """Convert centered FFT magnitude into log-polar coordinates."""
    h, w = magnitude.shape

    center = (
        (h - 1) / 2.0,
        (w - 1) / 2.0,
    )

    radius = min(h, w) / 2.0

    result = cast(
        np.ndarray,
        warp_polar(
            magnitude,
            center=center,
            radius=radius,
            output_shape=(h, w),
            scaling="log",
        ),
    )

    return np.asarray(
        result,
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Fourier-Mellin
# ---------------------------------------------------------------------------


def _estimate_fourier_mellin(
    reference: np.ndarray,
    search: np.ndarray,
) -> tuple[float, float, float]:
    """
    Estimate rotation and residual scale.

    Returns
    -------
        angle_deg
        residual_scale
        quality
    """
    side = max(
        reference.shape[0],
        reference.shape[1],
        search.shape[0],
        search.shape[1],
        64,
    )

    side = min(int(side), 512)

    if side % 2:
        side += 1

    ref = _prepare(
        reference,
        side,
    )

    sea = _prepare(
        search,
        side,
    )

    ref_mag = _fft_log_magnitude(ref)
    sea_mag = _fft_log_magnitude(sea)

    ref_lp = _log_polar(ref_mag)
    sea_lp = _log_polar(sea_mag)

    ref_lp -= ref_lp.mean()
    sea_lp -= sea_lp.mean()

    ref_std = float(ref_lp.std())
    sea_std = float(sea_lp.std())

    if ref_std < 1e-7 or sea_std < 1e-7:
        raise ValueError("weak Fourier spectrum")

    ref_lp /= ref_std
    sea_lp /= sea_std

    window = np.outer(
        np.hanning(side),
        np.hanning(side),
    ).astype(np.float32)

    ref_lp *= window
    sea_lp *= window

    # Use several local maxima from the unnormalised
    # spectral-correlation surface.
    corr = np.fft.ifft2(np.fft.fft2(ref_lp) * np.conj(np.fft.fft2(sea_lp)))

    corr_abs = np.abs(corr)

    row_shifts = np.arange(
        side,
        dtype=np.float32,
    )

    row_shifts[row_shifts > side // 2] -= side

    allowed_rows = np.abs(row_shifts / float(side) * 360.0) <= _MAX_DATASET_ROTATION_DEG

    local_peaks = corr_abs == ndimage.maximum_filter(
        corr_abs,
        size=(3, 3),
        mode="wrap",
    )

    candidate_mask = local_peaks & allowed_rows[:, None]

    candidate_indices = np.argwhere(candidate_mask)

    if candidate_indices.size == 0:
        candidate_indices = np.argwhere(allowed_rows[:, None])

    candidate_values = corr_abs[
        candidate_indices[:, 0],
        candidate_indices[:, 1],
    ]

    strongest = candidate_indices[np.argsort(candidate_values)[::-1][:8]]

    best_y, best_x = strongest[0]

    dy = float(row_shifts[best_y])

    dx = float(best_x if best_x <= side // 2 else best_x - side)

    # Log-polar vertical displacement -> rotation.
    angle = dy / float(side) * 360.0

    angle = _normalize_angle(angle)

    # Log-polar horizontal displacement -> scale.
    radius = max(
        float(side) / 2.0,
        2.0,
    )

    log_radius = np.log(radius)

    if log_radius <= 1e-8:
        raise ValueError("invalid log-polar radius")

    residual_scale = np.exp(-(dx / float(side)) * log_radius)

    if not np.isfinite(residual_scale) or residual_scale <= 0.0:
        raise ValueError("invalid residual scale")

    # The dataset expects nominal_scale to carry the scale.
    # Therefore this spectral value is only diagnostic.
    residual_scale = float(
        np.clip(
            residual_scale,
            0.80,
            1.25,
        )
    )

    # Confidence is angular discrimination.
    winner_angle = _normalize_angle(dy / float(side) * 360.0)

    winner_value = float(corr_abs[best_y, best_x])

    rival_values = []

    for candidate_y, candidate_x in strongest[1:]:
        candidate_angle = _normalize_angle(float(row_shifts[candidate_y]) / float(side) * 360.0)

        angular_gap = abs(_normalize_angle(candidate_angle - winner_angle))

        if angular_gap >= 2.0:
            rival_values.append(
                float(
                    corr_abs[
                        candidate_y,
                        candidate_x,
                    ]
                )
            )

    if rival_values and winner_value > 1e-8:
        quality = float(
            np.clip(
                1.0 - max(rival_values) / winner_value,
                0.0,
                1.0,
            )
        )
    else:
        quality = 1.0

    return (
        float(angle),
        float(residual_scale),
        quality,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def estimate_pose(
    reference: FloatArray,
    search: FloatArray,
    nominal_scale: float = config.NOMINAL_SCALE,
) -> PoseEstimate:
    """
    Estimate rotation and total scale.

    Parameters
    ----------
    reference:
        Reference image.

    search:
        Local candidate/search image.

    nominal_scale:
        Expected reference-to-search scale ratio.

    Returns
    -------
    PoseEstimate
        theta_deg:
            Estimated rotation.

        scale:
            TOTAL scale.

        quality:
            Confidence-like value in [0, 1].

    The function never raises.
    """
    try:
        reference = np.asarray(
            reference,
            dtype=np.float32,
        )

        search = np.asarray(
            search,
            dtype=np.float32,
        )

        if (
            reference.ndim != 2
            or search.ndim != 2
            or min(reference.shape) < 8
            or min(search.shape) < 8
        ):
            raise ValueError("invalid image dimensions")

        nominal_scale = float(nominal_scale)

        if not np.isfinite(nominal_scale) or nominal_scale <= 0.0:
            raise ValueError("invalid nominal scale")

        (
            fm_angle,
            fm_scale,
            fm_quality,
        ) = _estimate_fourier_mellin(
            reference,
            search,
        )

        logger.debug(
            "FM: theta=%.2f residual_scale=%.4f quality=%.3f",
            fm_angle,
            fm_scale,
            fm_quality,
        )

        if fm_quality >= _MIN_FM_QUALITY and abs(fm_angle) <= _MAX_DATASET_ROTATION_DEG:
            final_angle = fm_angle
            quality = fm_quality

            logger.debug("FM decision: accepted spectral rotation")
        else:
            final_angle = 0.0
            quality = 0.0

            logger.debug("FM decision: nominal-angle fallback")

        return PoseEstimate(
            theta_deg=float(final_angle),
            scale=float(nominal_scale),
            quality=quality,
        )

    except Exception:
        # Nominal pose at zero quality, which localize reads as "no estimate"
        # and replaces with nominal anyway. Deliberately not a spatial rotation
        # search: the previous revision fell back to sweeping angles and scoring
        # them by NCC between reference and search resized to a common square.
        # Those two images are a 1 um field and a 10 um field, so their lattices
        # sit at frequencies a decade apart and the correlation carries no
        # content correspondence -- the argmax is noise. It capped quality at
        # 0.25, above localize's _MIN_POSE_QUALITY of 0.20, so that noise was
        # accepted and rotated the template by a meaningless angle. It never
        # fired on any tracked run: all 1830 theta_est values across
        # results/*.csv sit exactly on the Fourier-Mellin bin grid, deviation
        # 0.000e+00. So removing it moves no measured number; it was a live
        # hazard only on inputs that make the spectral path raise.
        logger.debug("pose estimation failed; using nominal pose", exc_info=True)

        try:
            safe_nominal_scale = float(nominal_scale)
        except (TypeError, ValueError):
            safe_nominal_scale = float(config.NOMINAL_SCALE)

        if not np.isfinite(safe_nominal_scale) or safe_nominal_scale <= 0.0:
            safe_nominal_scale = float(config.NOMINAL_SCALE)

        return PoseEstimate(
            theta_deg=0.0,
            scale=safe_nominal_scale,
            quality=0.0,
        )
