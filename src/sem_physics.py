"""Initial, deterministic SEM-like image degradation for Drift-Sense.

This module is deliberately a *simulation baseline*, not a calibrated SEM
instrument model. The numeric presets below are initial engineering
assumptions selected to create independently degraded reference and search
captures. They must be tuned against the evaluation strata and replaced (or
supported) by literature citations before final submission.

Coordinate contract
-------------------
``apply_sem_chain`` only changes intensities. It never resamples, translates,
rotates, crops, or otherwise changes image geometry, so R1's pixel-centre
ground truth remains valid.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

import numpy as np

try:
    # SciPy is preferred; the NumPy fallback keeps the project runnable.
    from scipy import ndimage
except ImportError:  # pragma: no cover - exercised only in minimal runtimes
    ndimage = None

Array = np.ndarray


# ---------------------------------------------------------------------------
# Noise-level interface
# ---------------------------------------------------------------------------
#
# Noise strata modify ONLY shot noise and read noise.
#
# "none" is the baseline preset.
# "medium" intentionally uses the same noise parameters as the baseline.
#
# Low:
#     more electrons / higher white_counts -> less shot noise
#     lower read-noise sigma
#
# Medium:
#     baseline preset values
#
# High:
#     fewer electrons / lower white_counts -> more shot noise
#     higher read-noise sigma
#
# These are provisional simulation assumptions agreed with R1.
# They are NOT experimentally calibrated SEM settings.

NOISE_LEVELS = (
    "none",
    "low",
    "medium",
    "high",
)

NOISE_SCALING: dict[str, dict[str, float]] = {
    "none": {
        "white_counts": 1.0,
        "read_noise_sigma": 1.0,
    },
    "low": {
        "white_counts": 2.0,
        "read_noise_sigma": 0.5,
    },
    "medium": {
        "white_counts": 1.0,
        "read_noise_sigma": 1.0,
    },
    "high": {
        "white_counts": 0.35,
        "read_noise_sigma": 2.0,
    },
}


def _gaussian_kernel(sigma: float) -> Array:
    radius = max(1, int(np.ceil(3.0 * sigma)))
    axis = np.arange(
        -radius,
        radius + 1,
        dtype=np.float32,
    )

    kernel = np.exp(
        -0.5 * (axis / sigma) ** 2
    )

    return kernel / kernel.sum()


def _gaussian_filter1d(
    values: Array,
    sigma: float,
    axis: int = 0,
) -> Array:
    """Gaussian smoothing via SciPy, or a compact NumPy fallback."""

    if sigma <= 0:
        return values.astype(
            np.float32,
            copy=True,
        )

    if ndimage is not None:
        return ndimage.gaussian_filter1d(
            values,
            sigma,
            axis=axis,
            mode="reflect",
        )

    kernel = _gaussian_kernel(sigma)
    pad = len(kernel) // 2

    padded = np.pad(
        values,
        [
            (pad, pad) if i == axis else (0, 0)
            for i in range(values.ndim)
        ],
        mode="reflect",
    )

    return np.apply_along_axis(
        lambda line: np.convolve(
            line,
            kernel,
            mode="valid",
        ),
        axis,
        padded,
    )


def _gaussian_filter(
    values: Array,
    sigma: float,
) -> Array:
    if sigma <= 0:
        return values.astype(
            np.float32,
            copy=True,
        )

    if ndimage is not None:
        return ndimage.gaussian_filter(
            values,
            sigma=sigma,
            mode="reflect",
        )

    return _gaussian_filter1d(
        _gaussian_filter1d(
            values,
            sigma,
            axis=0,
        ),
        sigma,
        axis=1,
    )


def _sobel_magnitude(
    values: Array,
) -> Array:
    if ndimage is not None:
        return np.hypot(
            ndimage.sobel(
                values,
                axis=0,
            ),
            ndimage.sobel(
                values,
                axis=1,
            ),
        )

    padded = np.pad(
        values,
        1,
        mode="reflect",
    )

    gx = (
        (
            padded[:-2, 2:]
            + 2 * padded[1:-1, 2:]
            + padded[2:, 2:]
        )
        - (
            padded[:-2, :-2]
            + 2 * padded[1:-1, :-2]
            + padded[2:, :-2]
        )
    )

    gy = (
        (
            padded[2:, :-2]
            + 2 * padded[2:, 1:-1]
            + padded[2:, 2:]
        )
        - (
            padded[:-2, :-2]
            + 2 * padded[:-2, 1:-1]
            + padded[:-2, 2:]
        )
    )

    return np.hypot(
        gx,
        gy,
    )


# ---------------------------------------------------------------------------
# Initial simulation assumptions
# ---------------------------------------------------------------------------
# NOT measured SEM settings.
#
# Length-like values are in nm and converted using the supplied px_nm.
# Count-like values are effective detector counts at white, used solely to
# parameterise Poisson variance.
#
# The search preset is intentionally harsher because it represents a
# separate, lower-resolution capture (per the project brief).

REFERENCE_PRESET: dict[str, Any] = {
    "edge": {
        "strength": 0.055,
        "sigma_nm": 1.5,
        "percentile": 99.0,
    },
    "psf": {
        "sigma_nm": 1.2,
    },
    "shot_noise": {
        "white_counts": 12_000.0,
    },
    "read_noise": {
        "sigma": 0.004,
    },
    "scan": {
        "row_gain_std": 0.006,
        "row_offset_std": 0.003,
        "correlation_rows": 3.0,
        "stripe_amplitude": 0.002,
    },
}


SEARCH_PRESET: dict[str, Any] = {
    "edge": {
        "strength": 0.070,
        "sigma_nm": 15.0,
        "percentile": 99.0,
    },
    "psf": {
        "sigma_nm": 12.0,
    },
    "shot_noise": {
        "white_counts": 3_000.0,
    },
    "read_noise": {
        "sigma": 0.010,
    },
    "scan": {
        "row_gain_std": 0.018,
        "row_offset_std": 0.009,
        "correlation_rows": 4.0,
        "stripe_amplitude": 0.006,
    },
}


PRESETS: dict[str, dict[str, Any]] = {
    "reference": REFERENCE_PRESET,
    "search": SEARCH_PRESET,
}


def _validate_noise_level(
    noise_level: str,
) -> str:
    """Validate the requested noise severity level."""

    if noise_level not in NOISE_LEVELS:
        raise ValueError(
            f"unknown noise level {noise_level!r}; "
            f"supported levels are {NOISE_LEVELS}"
        )

    return noise_level


def _apply_noise_level(
    config: dict[str, Any],
    noise_level: str,
) -> dict[str, Any]:
    """Apply shot/read-noise severity scaling to a copied preset.

    The input config is modified in place and also returned.

    ``white_counts`` is multiplied by the severity factor. Because Poisson
    noise has variance proportional to the signal count, lower counts produce
    stronger relative shot noise.

    ``read_noise.sigma`` is multiplied directly by its severity factor.
    """

    if noise_level == "none":
        return config

    scaling = NOISE_SCALING[noise_level]

    config["shot_noise"]["white_counts"] *= (
        scaling["white_counts"]
    )

    config["read_noise"]["sigma"] *= (
        scaling["read_noise_sigma"]
    )

    return config


def _as_unit_float32(
    img: Array,
) -> Array:
    """Return a clipped 2-D grayscale image as float32 in [0, 1]."""

    image = np.asarray(img)

    if image.ndim != 2:
        raise ValueError(
            f"img must be a 2-D grayscale array; "
            f"got {image.shape!r}"
        )

    if np.issubdtype(
        image.dtype,
        np.integer,
    ):
        info = np.iinfo(image.dtype)

        if info.min < 0:
            raise ValueError(
                "integer img must have an unsigned intensity dtype"
            )

        image = (
            image.astype(np.float32)
            / np.float32(info.max)
        )

    else:
        image = image.astype(
            np.float32,
            copy=False,
        )

    if not np.all(
        np.isfinite(image)
    ):
        raise ValueError(
            "img contains NaN or infinity"
        )

    return np.clip(
        image,
        0.0,
        1.0,
    ).astype(
        np.float32,
        copy=False,
    )


def _nm_to_px(
    length_nm: float,
    px_nm: float,
) -> float:
    if px_nm <= 0 or not np.isfinite(px_nm):
        raise ValueError(
            "px_nm must be a finite positive number"
        )

    if length_nm < 0 or not np.isfinite(length_nm):
        raise ValueError(
            "length parameters must be finite and non-negative"
        )

    return float(
        length_nm / px_nm
    )


def edge_brightening(
    img: Array,
    *,
    strength: float,
    sigma_nm: float,
    px_nm: float,
    percentile: float = 99.0,
) -> Array:
    """Add a local gradient-derived edge signal.

    This is an intentionally simple proxy for SEM edge brightening: Sobel
    gradient magnitude is robustly normalised and softly spread near edges.
    """

    if (
        strength < 0
        or not 0 < percentile <= 100
    ):
        raise ValueError(
            "edge strength must be >= 0 and "
            "percentile in (0, 100]"
        )

    gradient = _sobel_magnitude(
        img
    )

    scale = float(
        np.percentile(
            gradient,
            percentile,
        )
    )

    if (
        scale
        <= np.finfo(
            np.float32
        ).eps
        or strength == 0
    ):
        return img.astype(
            np.float32,
            copy=True,
        )

    edge = np.minimum(
        gradient / scale,
        1.0,
    )

    sigma_px = _nm_to_px(
        sigma_nm,
        px_nm,
    )

    if sigma_px > 0:
        edge = _gaussian_filter(
            edge,
            sigma=sigma_px,
        )

    return np.clip(
        img + strength * edge,
        0.0,
        1.0,
    ).astype(
        np.float32
    )


def psf_blur(
    img: Array,
    *,
    sigma_nm: float,
    px_nm: float,
) -> Array:
    """Apply an isotropic Gaussian PSF approximation."""

    sigma_px = _nm_to_px(
        sigma_nm,
        px_nm,
    )

    if sigma_px == 0:
        return img.astype(
            np.float32,
            copy=True,
        )

    return _gaussian_filter(
        img,
        sigma=sigma_px,
    ).astype(
        np.float32
    )


def poisson_shot_noise(
    img: Array,
    *,
    white_counts: float,
    rng: np.random.Generator,
) -> Array:
    """Sample signal-dependent shot noise using an effective-count scale."""

    if (
        white_counts <= 0
        or not np.isfinite(white_counts)
    ):
        raise ValueError(
            "white_counts must be finite and positive"
        )

    counts = rng.poisson(
        np.clip(
            img,
            0.0,
            1.0,
        ) * white_counts
    )

    return (
        counts / white_counts
    ).astype(
        np.float32
    )


def read_noise(
    img: Array,
    *,
    sigma: float,
    rng: np.random.Generator,
) -> Array:
    """Add independent zero-mean Gaussian detector/read noise."""

    if (
        sigma < 0
        or not np.isfinite(sigma)
    ):
        raise ValueError(
            "read-noise sigma must be finite and non-negative"
        )

    if sigma == 0:
        return img.astype(
            np.float32,
            copy=True,
        )

    return (
        img
        + rng.normal(
            0.0,
            sigma,
            size=img.shape,
        )
    ).astype(
        np.float32
    )


def scan_artifacts(
    img: Array,
    *,
    row_gain_std: float,
    row_offset_std: float,
    correlation_rows: float,
    stripe_amplitude: float,
    rng: np.random.Generator,
) -> Array:
    """Apply correlated line gain/offset noise plus weak banding.

    This models intensity variation between scan lines only. It deliberately
    does not apply row displacement, which would alter image geometry.
    """

    values = (
        row_gain_std,
        row_offset_std,
        correlation_rows,
        stripe_amplitude,
    )

    if any(
        value < 0
        or not np.isfinite(value)
        for value in values
    ):
        raise ValueError(
            "scan parameters must be finite and non-negative"
        )

    rows = img.shape[0]

    gain = rng.normal(
        0.0,
        row_gain_std,
        size=rows,
    )

    offset = rng.normal(
        0.0,
        row_offset_std,
        size=rows,
    )

    if correlation_rows > 0:
        gain = _gaussian_filter1d(
            gain,
            correlation_rows,
        )

        offset = _gaussian_filter1d(
            offset,
            correlation_rows,
        )

    # Random phase avoids a fixed artifact pattern while keeping the passed
    # RNG as the sole source of stochasticity.
    phase = rng.uniform(
        0.0,
        2.0 * np.pi,
    )

    period_rows = max(
        8.0,
        rows / 12.0,
    )

    banding = (
        stripe_amplitude
        * np.sin(
            2.0
            * np.pi
            * np.arange(rows)
            / period_rows
            + phase
        )
    )

    return (
        img * (1.0 + gain[:, None])
        + offset[:, None]
        + banding[:, None]
    ).astype(
        np.float32
    )


def _merge_params(
    base: Mapping[str, Any],
    overrides: Mapping[str, Any],
) -> dict[str, Any]:
    merged = deepcopy(
        dict(base)
    )

    for name, value in overrides.items():
        if (
            isinstance(value, Mapping)
            and isinstance(
                merged.get(name),
                Mapping,
            )
        ):
            merged[name].update(
                value
            )
        else:
            merged[name] = value

    return merged


def apply_sem_chain(
    img: Array,
    px_nm: float,
    params: Mapping[str, Any] | str | None,
    rng: np.random.Generator,
) -> Array:
    """Apply the frozen R2 SEM simulation chain and return float32 [0, 1].

    Noise strata modify only Poisson shot noise and read noise.
    Geometry-affecting parameters remain unchanged across strata.

    Args:
        img:
            Grayscale source image, uint8-like or float in [0, 1].

        px_nm:
            Nanometres per pixel for this capture.

        params:
            ``"reference"``/``"search"`` preset, or a mapping containing
            ``preset``, ``noise_level`` and optional nested parameter
            overrides.

        rng:
            Caller-owned numpy.random.Generator.

    Returns:
        Same-shape float32 image clipped to [0, 1].
    """

    if not isinstance(rng, np.random.Generator):
        raise TypeError(
            "rng must be numpy.random.Generator"
        )

    # ---------------------------------------------------------------
    # Preset selection
    # ---------------------------------------------------------------

    if isinstance(params, str):

        if params not in PRESETS:
            raise ValueError(
                f"unknown preset {params!r}; "
                f"choose from {sorted(PRESETS)}"
            )

        preset = params
        noise_level = "none"

        config = deepcopy(
            PRESETS[preset]
        )

    else:

        supplied = (
            {}
            if params is None
            else dict(params)
        )

        preset = supplied.pop(
            "preset",
            "reference",
        )

        if preset not in PRESETS:
            raise ValueError(
                f"unknown preset {preset!r}; "
                f"choose from {sorted(PRESETS)}"
            )

        noise_level = supplied.pop(
            "noise_level",
            "none",
        )

        _validate_noise_level(
            noise_level
        )

        config = _merge_params(
            PRESETS[preset],
            supplied,
        )

    # ---------------------------------------------------------------
    # Apply noise-stratum scaling
    # ---------------------------------------------------------------
    #
    # IMPORTANT:
    # Only shot noise and read noise are modified.
    #
    # Edge brightening, PSF blur and scan artifacts remain fixed
    # within each capture preset.

    scaling = NOISE_SCALING[
        noise_level
    ]

    config["shot_noise"]["white_counts"] *= (
        scaling["white_counts"]
    )

    config["read_noise"]["sigma"] *= (
        scaling["read_noise_sigma"]
    )

    # ---------------------------------------------------------------
    # Frozen SEM physics chain
    # ---------------------------------------------------------------

    out = _as_unit_float32(img)

    out = edge_brightening(
        out,
        px_nm=px_nm,
        **config["edge"],
    )

    out = psf_blur(
        out,
        px_nm=px_nm,
        **config["psf"],
    )

    out = poisson_shot_noise(
        out,
        rng=rng,
        **config["shot_noise"],
    )

    out = read_noise(
        out,
        rng=rng,
        **config["read_noise"],
    )

    out = scan_artifacts(
        out,
        rng=rng,
        **config["scan"],
    )

    return np.clip(
        out,
        0.0,
        1.0,
    ).astype(np.float32)