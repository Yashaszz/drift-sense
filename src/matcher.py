"""Stages 2, 3, 3b and 5 of the localization pipeline.

Stage 2 — template construction
    Render the reference patch as it *should appear* in the search image:
    rotate, match the search image's point-spread function, then decimate by
    area-averaging.

Stage 3 — matched filtering
    Score every candidate position with zero-mean normalized cross-correlation.

Stage 3b — peak extraction
    Reduce the correlation surface to a short candidate list.

Stage 5 — subpixel refinement
    Interpolate the winning peak to a fraction of a pixel.

Status
------
Stage 3 unmasked ZNCC is implemented (T2). Everything else is still a
placeholder that returns a correctly typed and correctly *shaped* result, so the
end-to-end pipeline runs and downstream modules can be written against it.
Remaining work: T3 peak extraction, T4 template construction, T5 subpixel
refinement, T8 masked ZNCC.

Placeholder outputs are deliberately shape-correct rather than pass-through.
Shape is the part of the contract that downstream code depends on — a
pass-through stub would let shape bugs hide until integration.
"""

import warnings
from collections.abc import Sequence

import cv2
import numpy as np

from src import config
from src.types import AnyArray, BoolArray, FloatArray, Peak, Shape2D

_MIN_TEMPLATE_STD: float = 1e-6
"""Standard deviation below which a template is treated as carrying no signal.

Absolute rather than relative, and safe for both raw 8-bit data (sigma of order
tens) and Stage 0 normalized data (sigma of order one): only a genuinely
constant patch falls below it.

This guard is not optional. ``cv2.matchTemplate`` with ``TM_CCOEFF_NORMED``
returns **1.0 at every position** for a zero-variance template rather than NaN,
because its normalization divides by a zero denominator. Left unguarded, a
degenerate template reads as a perfect match everywhere — the worst possible
failure, since it is both wrong and maximally confident.
"""

__all__ = [
    "area_average_downsample",
    "build_template",
    "estimate_psf_sigma",
    "match_psf",
    "refine_subpixel",
    "refine_subpixel_crop",
    "rotate_image",
    "top_k_peaks",
    "zncc_surface",
]


# ===========================================================================
# Stage 2 — template construction
# ===========================================================================


def estimate_psf_sigma(
    search: FloatArray,
    candidates: Sequence[float] | None = None,
) -> float:
    """Estimate the search image's effective point-spread function width.

    The reference and the search image are two independent physical captures and
    went through different filter chains. Decimating the reference without
    matching the search image's blur injects high-frequency content that does
    not exist in the search image, and the correlation peak drops. In receiver
    terms: match the filters before you correlate.

    On our own generated data the value is known. On unseen data it is recovered
    from the radial power-spectrum rolloff, or by scanning a small set of
    candidate widths and keeping the one that yields the strongest peak.

    Parameters
    ----------
    search
        Search image, normalized, as ``(rows, cols)``.
    candidates
        Optional explicit sigma values to scan, in search pixels. When ``None``,
        the width is estimated from the power spectrum directly.

    Returns
    -------
    float
        Estimated Gaussian sigma in search pixels. Falls back to
        ``config.DEFAULT_PSF_SIGMA_PX`` when the spectrum is too flat to fit.

    Notes
    -----
    Never raises. An unreliable estimate degrades match quality; an exception
    would break the never-raises contract of :func:`src.localize.localize`.
    """
    del search, candidates  # placeholder: estimation lands in T4
    return config.DEFAULT_PSF_SIGMA_PX


def match_psf(image: FloatArray, sigma_px: float) -> FloatArray:
    """Low-pass an image to match a target point-spread function.

    Parameters
    ----------
    image
        Input image as ``(rows, cols)``.
    sigma_px
        Gaussian sigma in pixels of the *same grid as* ``image``. Values at or
        below zero leave the image unchanged.

    Returns
    -------
    FloatArray
        Blurred image, same shape as the input, ``float32``.
    """
    del sigma_px  # placeholder: Gaussian convolution lands in T4
    return np.zeros_like(image, dtype=np.float32)


def area_average_downsample(image: FloatArray, factor: float) -> FloatArray:
    """Decimate an image by averaging over pixel areas.

    Area-averaging, never bicubic or Lanczos. Two reasons, and both belong in
    the presentation. Physically, a detector *integrates* over the pixel area,
    so a box average is what the instrument actually does. Numerically,
    interpolating kernels with negative lobes ring at edges — a measured
    undershoot to 121 against a 128 background — and that ringing is structured
    noise correlated with feature edges, which is exactly where the matcher
    draws its signal.

    Parameters
    ----------
    image
        Input image as ``(rows, cols)``.
    factor
        Decimation ratio, greater than 1 to shrink. Non-integer values are
        supported and expected: the true scale is near 10x but not exactly
        10.000x.

    Returns
    -------
    FloatArray
        Decimated image, ``float32``, with shape
        ``(round(rows / factor), round(cols / factor))``, at least 1x1.

    Raises
    ------
    ValueError
        If ``factor`` is not strictly positive.
    """
    if factor <= 0.0:
        msg = f"factor must be strictly positive, got {factor!r}"
        raise ValueError(msg)
    rows, cols = image.shape
    out_shape = (max(1, round(rows / factor)), max(1, round(cols / factor)))
    return np.zeros(out_shape, dtype=np.float32)


def rotate_image(image: FloatArray, theta_deg: float) -> FloatArray:
    """Rotate an image about its centre, preserving shape.

    Applied at full reference resolution, *before* any decimation. The reference
    is heavily oversampled at 1 nm/px, so interpolation error there is
    negligible; rotating after decimation would interpolate a signal that has
    already lost the detail the rotation needs.

    Parameters
    ----------
    image
        Input image as ``(rows, cols)``.
    theta_deg
        Rotation in degrees, positive counter-clockwise.

    Returns
    -------
    FloatArray
        Rotated image, same shape as the input, ``float32``. Regions rotated in
        from outside the original frame are filled with the image mean, not
        zero, so the fill does not act as a synthetic edge feature.
    """
    del theta_deg  # placeholder: affine warp lands in T4
    return np.zeros_like(image, dtype=np.float32)


def build_template(
    reference: FloatArray,
    theta: float,
    scale: float,
    psf_sigma_px: float,
) -> FloatArray:
    """Build the template as the reference patch should appear in the search image.

    Composes the Stage 2 operations in the order that matters:
    rotate at full resolution, match the PSF, then decimate by area-averaging.

    Parameters
    ----------
    reference
        Reference image at 1 nm/px, as ``(rows, cols)``.
    theta
        Rotation in degrees, from :class:`src.types.PoseEstimate`.
    scale
        Decimation ratio, from :class:`src.types.PoseEstimate`. Near
        ``config.NOMINAL_SCALE`` but not exactly equal to it.
    psf_sigma_px
        Target PSF width in *search* pixels, from :func:`estimate_psf_sigma`.

    Returns
    -------
    FloatArray
        Template, ``float32``, approximately
        ``config.TEMPLATE_NOMINAL_PX`` square. Callers must read the actual
        shape rather than assuming it — the scale residual changes it.

    Raises
    ------
    ValueError
        If ``scale`` is not strictly positive.
    """
    del theta, psf_sigma_px  # placeholder: composition lands in T4
    return area_average_downsample(reference, scale)


# ===========================================================================
# Stage 3 — matched filtering
# ===========================================================================


def zncc_surface(
    template: FloatArray,
    search: FloatArray,
    weight: FloatArray | None = None,
) -> FloatArray:
    """Score every candidate position by zero-mean normalized cross-correlation.

    ZNCC rather than sum-of-squared-differences because the two images are
    independent captures with possibly different detector gain and offset. ZNCC
    is invariant to any affine intensity change; SSD is not, and fails outright.

    Parameters
    ----------
    template
        Template from :func:`build_template`, as ``(t_rows, t_cols)``.
    search
        Search image, as ``(rows, cols)``.
    weight
        Optional per-pixel weight mask over the template, same shape as
        ``template``. When given, a masked correlation is computed instead,
        concentrating the match on the informative parts of the reference. When
        ``None``, the standard unmasked correlation is used.

    Returns
    -------
    FloatArray
        Correlation surface of shape ``(rows - t_rows + 1, cols - t_cols + 1)``,
        ``float32``, with values in ``[-1, 1]``. Index ``(r, c)`` scores the
        template placed with its *top-left corner* at that position — use
        :func:`src.config.window_topleft_to_centre` to convert.

    Raises
    ------
    ValueError
        If the template is larger than the search image in either axis, or if
        ``weight`` is given with a shape differing from ``template``.

    Notes
    -----
    Dispatches to an OpenCV path when unmasked and a masked-Fourier path
    otherwise. OpenCV's ``TM_CCOEFF_NORMED`` is exactly ZNCC and is already
    DFT-accelerated with running sums, but it cannot take a mask — which is why
    the masked variant is implemented separately.
    """
    template_f = _as_working_array(template, "template")
    search_f = _as_working_array(search, "search")

    t_rows, t_cols = template_f.shape
    rows, cols = search_f.shape
    if t_rows > rows or t_cols > cols:
        msg = f"template {template_f.shape} exceeds search image {search_f.shape}"
        raise ValueError(msg)
    if weight is not None and weight.shape != template_f.shape:
        msg = f"weight shape {weight.shape} does not match template {template_f.shape}"
        raise ValueError(msg)
    if weight is None:
        return _zncc_opencv(template_f, search_f)
    return _zncc_masked_fft(template_f, search_f, _as_working_array(weight, "weight"))


def _as_working_array(array: AnyArray, name: str) -> FloatArray:
    """Coerce an input to the contiguous ``float32`` form OpenCV requires.

    Parameters
    ----------
    array
        Input array of any real dtype.
    name
        Parameter name, used in error messages.

    Returns
    -------
    FloatArray
        Two-dimensional contiguous ``float32`` view or copy.

    Raises
    ------
    ValueError
        If the array is not two-dimensional, or holds non-finite values.

    Notes
    -----
    ``cv2.matchTemplate`` accepts only ``uint8`` and ``float32``; passing
    ``float64`` raises ``cv2.error``. Coercing here rather than demanding the
    right dtype from callers keeps the public interface permissive, which
    matters because the deliverable must run on whatever the harness hands it.

    Non-finite values are rejected rather than silently repaired. NaN propagates
    through the correlation and would produce a plausible-looking surface with
    an arbitrary peak; a clear error, caught and recorded by
    :func:`src.localize.localize`, is far easier to diagnose.
    """
    as_array = np.asarray(array)
    if as_array.ndim != 2:
        msg = f"{name} must be two-dimensional, got shape {as_array.shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(as_array)):
        msg = f"{name} contains non-finite values (NaN or inf)"
        raise ValueError(msg)
    return np.ascontiguousarray(as_array, dtype=np.float32)


def _zncc_opencv(template: FloatArray, search: FloatArray) -> FloatArray:
    """Compute unmasked ZNCC via OpenCV's ``TM_CCOEFF_NORMED``.

    ``TM_CCOEFF_NORMED`` *is* zero-mean normalized cross-correlation: OpenCV
    subtracts the template mean and the per-window search mean, then divides by
    the product of the two standard deviations. It is already DFT-accelerated
    with running sums for the local statistics, which is the standard fast
    normalized cross-correlation formulation.

    Parameters
    ----------
    template
        Template as ``(t_rows, t_cols)``, contiguous ``float32``.
    search
        Search image as ``(rows, cols)``, contiguous ``float32``.

    Returns
    -------
    FloatArray
        Correlation surface, ``float32``, values in ``[-1, 1]``.

    Notes
    -----
    Two post-conditions are enforced rather than assumed.

    A zero-variance template short-circuits to an all-zero surface. OpenCV would
    otherwise return 1.0 everywhere, which reads as a perfect match at every
    position. Zero is the honest answer: a constant patch contains no
    information about where it sits.

    The surface is sanitised and clipped. Windows of the search image with zero
    variance can yield non-finite intermediates, and ordinary floating-point
    error can push a perfect match marginally past 1.0, which would violate the
    range this function documents.
    """
    t_rows, t_cols = template.shape
    rows, cols = search.shape

    if float(np.std(template, dtype=np.float64)) < _MIN_TEMPLATE_STD:
        return np.zeros((rows - t_rows + 1, cols - t_cols + 1), dtype=np.float32)

    # matchTemplate already yields float32, so asarray is a no-copy view; it is
    # here to pin the static type, which OpenCV's stubs leave broader.
    surface: FloatArray = np.asarray(
        cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED),
        dtype=np.float32,
    )
    np.nan_to_num(surface, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
    np.clip(surface, -1.0, 1.0, out=surface)
    return surface


def _zncc_masked_fft(
    template: FloatArray,
    search: FloatArray,
    weight: FloatArray,
) -> FloatArray:
    """Masked normalized cross-correlation in the Fourier domain.

    Implements the masked-correlation formulation, which extends standard
    FFT-based normalized cross-correlation to a per-pixel weight mask by
    accumulating the masked sums as additional correlations.

    Parameters
    ----------
    template
        Template as ``(t_rows, t_cols)``.
    search
        Search image as ``(rows, cols)``.
    weight
        Per-pixel weights over the template, same shape as ``template``.

    Returns
    -------
    FloatArray
        Correlation surface, ``float32``.

    Warns
    -----
    UserWarning
        Always, until T8. The masked formulation is not implemented yet and this
        falls back to unmasked correlation, which ignores the weight entirely.
        Warning rather than raising keeps the never-raises contract intact while
        making sure a caller cannot mistake the fallback for a working masked
        correlation.

    Notes
    -----
    With an all-ones weight this must agree with :func:`_zncc_opencv` to within
    floating-point tolerance. That equivalence is the primary correctness test
    once the real implementation lands.
    """
    del weight  # placeholder: masked formulation lands in T8
    warnings.warn(
        "masked ZNCC is not implemented yet (T8); ignoring the weight mask and "
        "returning unmasked correlation",
        UserWarning,
        stacklevel=3,
    )
    return _zncc_opencv(template, search)


# ===========================================================================
# Stage 3b — peak extraction
# ===========================================================================


def _local_maxima(surface: FloatArray, radius: int) -> BoolArray:
    """Mark strict local maxima within a square neighbourhood.

    Parameters
    ----------
    surface
        Correlation surface.
    radius
        Neighbourhood half-width in pixels.

    Returns
    -------
    BoolArray
        Mask, same shape as ``surface``, true at local maxima.

    Raises
    ------
    ValueError
        If ``radius`` is negative.
    """
    if radius < 0:
        msg = f"radius must be non-negative, got {radius!r}"
        raise ValueError(msg)
    return np.zeros(surface.shape, dtype=np.bool_)


def top_k_peaks(surface: FloatArray, k: int, nms_radius: int) -> list[Peak]:
    """Reduce a correlation surface to its strongest well-separated candidates.

    Non-maximum suppression is applied at a radius tied to the estimated lattice
    pitch, so the shortlist holds one candidate *per lattice cell* rather than
    fifty clustered on a single peak. In a periodic field the true match may be
    any one of them, and the shortlist is what Stage 4 disambiguates.

    Parameters
    ----------
    surface
        Correlation surface from :func:`zncc_surface`.
    k
        Maximum number of candidates to return.
    nms_radius
        Suppression radius in surface pixels. Derived at runtime from the
        lattice pitch; falls back to ``config.DEFAULT_NMS_RADIUS_PX``.

    Returns
    -------
    list of Peak
        At most ``k`` peaks, sorted by descending score. Returns fewer when the
        surface holds fewer separated maxima; the list is never padded. Returns
        an empty list when ``k`` is non-positive or the surface is empty.

    Raises
    ------
    ValueError
        If ``nms_radius`` is negative.
    """
    if nms_radius < 0:
        msg = f"nms_radius must be non-negative, got {nms_radius!r}"
        raise ValueError(msg)
    if k <= 0 or surface.size == 0:
        return []
    # Placeholder: a single candidate at the surface centre, so the downstream
    # chain is exercised end-to-end. Real extraction lands in T3.
    row = surface.shape[0] // 2
    col = surface.shape[1] // 2
    return [Peak(col=col, row=row, score=float(surface[row, col]))]


# ===========================================================================
# Stage 5 — subpixel refinement
# ===========================================================================


def refine_subpixel(
    surface: FloatArray,
    peak: Peak,
    upsample: int = config.DEFAULT_UPSAMPLE,
) -> tuple[float, float]:
    """Refine a peak position by upsampled-DFT interpolation of the surface.

    Zero-padding in the frequency domain is exact sinc interpolation of a
    band-limited surface. A parabola fitted to three samples is a cruder
    interpolator with a known bias toward the sample centre — at a 0.5 px
    tolerance that bias is not academic.

    Parameters
    ----------
    surface
        Correlation surface from :func:`zncc_surface`.
    peak
        Winning integer-valued peak.
    upsample
        Interpolation factor; 100 gives 1/100 px resolution.

    Returns
    -------
    tuple of float
        Residual offset ``(dx, dy)`` in surface pixels, to be *added* to the
        integer peak position. Not an absolute coordinate.

    Raises
    ------
    ValueError
        If ``upsample`` is less than 1.
    """
    del surface, peak  # placeholder: upsampled DFT lands in T5
    if upsample < 1:
        msg = f"upsample must be at least 1, got {upsample!r}"
        raise ValueError(msg)
    return (0.0, 0.0)


def refine_subpixel_crop(
    template: FloatArray,
    search: FloatArray,
    peak: Peak,
    upsample: int = config.DEFAULT_UPSAMPLE,
) -> tuple[float, float]:
    """Refine a peak by phase-correlating the template against the winning crop.

    Alternative to :func:`refine_subpixel` with an identical return contract, so
    the two are interchangeable behind a single flag. This variant applies the
    upsampled-DFT registration method to the image data directly, rather than to
    a correlation surface whose shape is distorted by the local-variance
    denominator of the ZNCC normalization.

    Which is more accurate is an empirical question. Both are cheap; we measure
    rather than argue.

    Parameters
    ----------
    template
        Template from :func:`build_template`.
    search
        Search image.
    peak
        Winning integer-valued peak locating the crop.
    upsample
        Interpolation factor.

    Returns
    -------
    tuple of float
        Residual offset ``(dx, dy)`` in search pixels, to be *added* to the
        integer peak position.

    Raises
    ------
    ValueError
        If ``upsample`` is less than 1.
    """
    del template, search, peak  # placeholder: crop phase correlation lands in T5
    if upsample < 1:
        msg = f"upsample must be at least 1, got {upsample!r}"
        raise ValueError(msg)
    return (0.0, 0.0)


# ===========================================================================
# Shape helpers
# ===========================================================================


def surface_shape(template_shape: Shape2D, search_shape: Shape2D) -> Shape2D:
    """Return the shape of the correlation surface for a template/search pair.

    Parameters
    ----------
    template_shape
        Template shape as ``(rows, cols)``.
    search_shape
        Search-image shape as ``(rows, cols)``.

    Returns
    -------
    tuple of int
        Surface shape as ``(rows, cols)``.

    Raises
    ------
    ValueError
        If the template exceeds the search image in either axis.
    """
    t_rows, t_cols = template_shape
    rows, cols = search_shape
    if t_rows > rows or t_cols > cols:
        msg = f"template {template_shape} exceeds search image {search_shape}"
        raise ValueError(msg)
    return (rows - t_rows + 1, cols - t_cols + 1)
