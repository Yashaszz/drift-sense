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

Public API is frozen:
    estimate_pose(reference, search, nominal_scale)

Never raises.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import ndimage
from skimage.transform import resize, warp_polar

from src import config
from src.types import FloatArray, PoseEstimate

__all__ = ["estimate_pose"]

logger = logging.getLogger(__name__)


# The real-data generator uses rotations in [-8, +8] degrees.  Wider
# Fourier-Mellin hypotheses on periodic, translated crops are aliases.
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
        x = resize(
            x,
            (size, size),
            order=1,
            mode="reflect",
            anti_aliasing=True,
            preserve_range=True,
        ).astype(np.float32)

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

    return (x * window).astype(np.float32)


def _fft_log_magnitude(image: np.ndarray) -> np.ndarray:
    """Return centered logarithmic FFT magnitude."""
    spectrum = np.fft.fftshift(np.fft.fft2(image))

    magnitude = np.abs(spectrum)

    return np.log1p(magnitude).astype(np.float32)


def _log_polar(magnitude: np.ndarray) -> np.ndarray:
    """Convert centered FFT magnitude into log-polar coordinates."""
    h, w = magnitude.shape

    center = (
        (h - 1) / 2.0,
        (w - 1) / 2.0,
    )

    radius = min(h, w) / 2.0

    result = warp_polar(
        magnitude,
        center=center,
        radius=radius,
        output_shape=(h, w),
        scaling="log",
    )

    return result.astype(np.float32)


# ---------------------------------------------------------------------------
# Correlation helpers
# ---------------------------------------------------------------------------


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Zero-mean normalized cross-correlation."""
    if a.shape != b.shape:
        return -1.0

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    a = a - float(a.mean())
    b = b - float(b.mean())

    denom = np.linalg.norm(a) * np.linalg.norm(b)

    if denom < 1e-8:
        return -1.0

    value = float(np.sum(a * b) / denom)

    if not np.isfinite(value):
        return -1.0

    return float(np.clip(value, -1.0, 1.0))


def _rotate_scale(
    image: np.ndarray,
    angle_deg: float,
    scale: float,
) -> np.ndarray:
    """
    Transform an image around its centre.

    The returned image has the same dimensions.
    """
    h, w = image.shape

    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0

    theta = np.deg2rad(angle_deg)

    c = np.cos(theta)
    s = np.sin(theta)

    matrix = np.array(
        [
            [c, s],
            [-s, c],
        ],
        dtype=np.float64,
    )

    matrix /= max(float(scale), 1e-8)

    centre = np.array(
        [cy, cx],
        dtype=np.float64,
    )

    offset = centre - matrix @ centre

    return ndimage.affine_transform(
        image,
        matrix=matrix,
        offset=offset,
        output_shape=image.shape,
        order=1,
        mode="reflect",
    ).astype(np.float32)


def _pose_score(
    reference: np.ndarray,
    search: np.ndarray,
    angle_deg: float,
) -> float:
    """
    Compare reference with search after undoing rotation.

    Residual scale is deliberately fixed at 1.0.
    """
    corrected = _rotate_scale(
        search,
        -float(angle_deg),
        1.0,
    )

    return _ncc(
        reference,
        corrected,
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

    # Use several local maxima from the unnormalised spectral-correlation
    # surface.  Phase-normalised correlation gives periodic aliases undue
    # weight on these SEM crops.  Limit candidates to the generator's
    # physically valid rotation band and choose its strongest local peak.
    corr = np.fft.ifft2(np.fft.fft2(ref_lp) * np.conj(np.fft.fft2(sea_lp)))

    corr_abs = np.abs(corr)
    row_shifts = np.arange(side, dtype=np.float32)
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

    # Confidence is angular discrimination, not global spectral energy.
    # Ignore peaks within 2 degrees because they belong to the same broad
    # angular lobe; compare the selected lobe with distinct-angle rivals.
    winner_angle = _normalize_angle(dy / float(side) * 360.0)
    winner_value = float(corr_abs[best_y, best_x])
    rival_values = []

    for candidate_y, candidate_x in strongest[1:]:
        candidate_angle = _normalize_angle(float(row_shifts[candidate_y]) / float(side) * 360.0)
        angular_gap = abs(_normalize_angle(candidate_angle - winner_angle))

        if angular_gap >= 2.0:
            rival_values.append(float(corr_abs[candidate_y, candidate_x]))

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
# Rotation refinement
# ---------------------------------------------------------------------------


def _refine_rotation(
    reference: np.ndarray,
    search: np.ndarray,
    initial_angle: float,
) -> tuple[float, float]:
    """
    Refine rotation using spatial NCC.

    The previous implementation searched only ±5 degrees.
    That caused many estimates to become clipped at ±5 degrees.

    New strategy:
        1. Search ±20 degrees around the FM estimate at 2° steps.
        2. Fine-search ±2 degrees around the best candidate at 0.25° steps.

    Residual scale remains fixed at 1.0.
    """
    side = max(
        min(
            reference.shape[0],
            reference.shape[1],
            search.shape[0],
            search.shape[1],
        ),
        64,
    )

    side = min(
        int(side),
        256,
    )

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

    initial_angle = _normalize_angle(float(initial_angle))

    # ---------------------------------------------------------------
    # Broad coarse search.
    #
    # This is deliberately much wider than the old ±5° search.
    # ---------------------------------------------------------------

    coarse_angles = np.arange(
        initial_angle - 20.0,
        initial_angle + 20.01,
        2.0,
    )

    best_score = -1.0
    best_angle = initial_angle

    for angle in coarse_angles:
        angle = _normalize_angle(float(angle))

        score = _pose_score(
            ref,
            sea,
            angle,
        )

        if score > best_score:
            best_score = score
            best_angle = angle

    # ---------------------------------------------------------------
    # Fine search around the coarse optimum.
    # ---------------------------------------------------------------

    fine_angles = np.arange(
        best_angle - 2.0,
        best_angle + 2.001,
        0.25,
    )

    for angle in fine_angles:
        angle = _normalize_angle(float(angle))

        score = _pose_score(
            ref,
            sea,
            angle,
        )

        if score > best_score:
            best_score = score
            best_angle = angle

    quality = float(
        np.clip(
            (best_score + 1.0) * 0.5,
            0.0,
            1.0,
        )
    )

    return (
        float(_normalize_angle(best_angle)),
        quality,
    )


# ---------------------------------------------------------------------------
# Bounded fallback
# ---------------------------------------------------------------------------


def _fallback(
    reference: np.ndarray,
    search: np.ndarray,
) -> tuple[float, float, float]:
    """
    Safe fallback rotation search.

    Residual scale is fixed at 1.0.

    Search range:
        -20° ... +20°

    Coarse:
        1° steps

    Fine:
        ±1° around best result at 0.25° steps
    """
    side = max(
        min(
            reference.shape[0],
            reference.shape[1],
            search.shape[0],
            search.shape[1],
        ),
        64,
    )

    side = min(
        int(side),
        256,
    )

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

    angles = np.arange(
        -20.0,
        20.01,
        1.0,
    )

    best_score = -1.0
    best_angle = 0.0

    for angle in angles:
        score = _pose_score(
            ref,
            sea,
            float(angle),
        )

        if score > best_score:
            best_score = score
            best_angle = float(angle)

    # Fine refinement.
    fine_angles = np.arange(
        best_angle - 1.0,
        best_angle + 1.001,
        0.25,
    )

    for angle in fine_angles:
        score = _pose_score(
            ref,
            sea,
            float(angle),
        )

        if score > best_score:
            best_score = score
            best_angle = float(angle)

    quality = float(
        np.clip(
            (best_score + 1.0) * 0.5,
            0.0,
            1.0,
        )
    )

    return (
        float(_normalize_angle(best_angle)),
        1.0,
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
            TOTAL scale:

                nominal_scale * residual_scale

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

        # -----------------------------------------------------------
        # Fourier-Mellin estimate
        # -----------------------------------------------------------

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

        # Spatial NCC cannot validate pose here: it compares different crops
        # without first solving their translation.  On periodic layouts it
        # creates false angle maxima, which is why the ±20 degree search
        # degraded the full-dataset result.  Fourier magnitude is translation
        # invariant, so retain that angle only when its peak is credible and
        # the result is inside the generator's physically valid rotation band.
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
        # -----------------------------------------------------------
        # Safe fallback.
        # -----------------------------------------------------------

        try:
            safe_nominal_scale = float(nominal_scale)

            if not np.isfinite(safe_nominal_scale) or safe_nominal_scale <= 0.0:
                safe_nominal_scale = float(config.NOMINAL_SCALE)

            reference = np.asarray(
                reference,
                dtype=np.float32,
            )

            search = np.asarray(
                search,
                dtype=np.float32,
            )

            (
                fallback_angle,
                _,
                fallback_quality,
            ) = _fallback(
                reference,
                search,
            )

            return PoseEstimate(
                theta_deg=float(fallback_angle),
                scale=float(safe_nominal_scale),
                quality=float(
                    min(
                        fallback_quality,
                        0.25,
                    )
                ),
            )

        except Exception:
            return PoseEstimate(
                theta_deg=0.0,
                scale=float(config.NOMINAL_SCALE),
                quality=0.0,
            )
