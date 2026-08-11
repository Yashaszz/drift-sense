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
Stages 2 (T4), 3 (T2), 3b (T3) and 5 (T5) are implemented, including the
uniqueness-weighted correlation path (T8).

The weighted path is exercised but not yet *informative*: ``uniqueness_map``
still returns a constant, under which weighted correlation provably reduces to
the unweighted result. That equivalence is the contract between the two paths
and is asserted directly, so the weighting starts doing real work the moment R3
lands a non-constant map, with no change here.
"""

from collections.abc import Sequence
from functools import lru_cache

import cv2
import numpy as np
from skimage.registration import phase_cross_correlation

from src import config
from src.types import (
    AnyArray,
    BoolArray,
    Float64Array,
    FloatArray,
    Peak,
    Shape2D,
    SubpixelRefinement,
)

_MAX_SUBPIXEL_OFFSET_PX: float = 1.0
"""Largest residual a refinement may report before it is rejected.

Stage 3b already found the peak to the nearest whole pixel, so a legitimate
residual is sub-pixel by construction.
"""

_SURFACE_PATCH_RADIUS: int = 3
"""Half-width of the correlation-surface neighbourhood used by the fallback."""

_MIN_PSF_SIGMA_PX: float = 0.1
_MAX_PSF_SIGMA_PX: float = 5.0
"""Bounds on the estimated PSF width, in search pixels.

A width below ~0.1 px is indistinguishable from no blur at all, and one above
~5 px would smear a 100 px template beyond usefulness. The spectral fit is
confounded by the scene's own spectrum, so clamping hard is preferable to
trusting an outlier.
"""

_SPECTRUM_BINS: int = 64
_FIT_BAND: tuple[float, float] = (0.05, 0.35)
_MIN_FIT_POINTS: int = 8
_MIN_SPECTRUM_PIXELS: int = 64
_MIN_FIT_R_SQUARED: float = 0.90
_SPECTRUM_MAX_EDGE: int = 512
"""Parameters of the radial power-spectrum fit.

The band excludes the lowest frequencies, where the scene's own structure
dominates, and the highest, where the noise floor flattens the rolloff.
"""

_MIN_WEIGHTED_VARIANCE: float = 1e-12
"""Weighted variance below which a template or window carries no usable signal.

Smaller than the unweighted guard because both inputs are standardised before
the weighted correlation, so variances are of order one rather than of order
the data range.
"""

_FLAT_SURFACE_ATOL: float = 1e-12
"""Range below which a correlation surface is treated as carrying no peaks.

A surface with no variation distinguishes no position, so the honest answer is
an empty candidate list rather than an arbitrary one. Short-circuiting also
avoids a performance cliff: on a perfectly flat surface every one of the ~800000
positions is technically a local maximum.
"""

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
    "build_weight",
    "estimate_psf_sigma",
    "match_psf",
    "refine_subpixel",
    "refine_subpixel_crop",
    "refine_subpixel_detailed",
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

    This is a *coarse prior*, not a precise measurement. A Gaussian blur of
    width sigma multiplies the power spectrum by ``exp(-4 pi^2 sigma^2 f^2)``, so
    a straight-line fit of ``log P`` against ``f^2`` recovers sigma from the
    slope. The confound is that the scene's own spectrum is not flat, and on a
    strongly periodic layout it is dominated by lattice impulses rather than a
    smooth rolloff. The fit is therefore restricted to a mid-frequency band and
    the result is clamped hard.

    The precise value is better obtained by scanning a few candidate widths and
    keeping the one that maximises the correlation peak. That needs *both*
    images, so it belongs in :func:`src.localize.localize` rather than here.
    """
    image = np.asarray(search, dtype=np.float32)
    if image.ndim != 2 or image.size < _MIN_SPECTRUM_PIXELS or not np.all(np.isfinite(image)):
        return _snap_to_candidates(config.DEFAULT_PSF_SIGMA_PX, candidates)

    estimate = _sigma_from_spectrum(_centre_crop(image, _SPECTRUM_MAX_EDGE))
    if estimate is None:
        return _snap_to_candidates(config.DEFAULT_PSF_SIGMA_PX, candidates)

    clamped = float(np.clip(estimate, _MIN_PSF_SIGMA_PX, _MAX_PSF_SIGMA_PX))
    return _snap_to_candidates(clamped, candidates)


def _centre_crop(image: FloatArray, max_edge: int) -> FloatArray:
    """Take a centred window of at most ``max_edge`` on each side.

    Parameters
    ----------
    image
        Source image.
    max_edge
        Maximum edge length to retain.

    Returns
    -------
    FloatArray
        Centred crop, or the input unchanged when already small enough.

    Notes
    -----
    The PSF estimate reads the *shape* of the spectral rolloff, which a
    representative window captures as well as the whole frame. The full
    transform of a 1000x1000 image costs around 60 ms — more than half the
    entire Robust-mode budget — for no gain in an estimate that is deliberately
    clamped and gated anyway.
    """
    rows, cols = image.shape
    if rows <= max_edge and cols <= max_edge:
        return image
    keep_rows, keep_cols = min(rows, max_edge), min(cols, max_edge)
    top = (rows - keep_rows) // 2
    left = (cols - keep_cols) // 2
    return np.ascontiguousarray(image[top : top + keep_rows, left : left + keep_cols])


def _sigma_from_spectrum(image: FloatArray) -> float | None:
    """Fit a Gaussian width from the radially averaged power spectrum.

    Parameters
    ----------
    image
        Two-dimensional image, finite valued.

    Returns
    -------
    float or None
        Estimated sigma in pixels of ``image``'s own grid, or ``None`` when the
        spectrum does not support a fit.
    """
    rows, cols = image.shape
    window = np.outer(np.hanning(rows), np.hanning(cols))
    spectrum = np.fft.fftshift(np.abs(np.fft.fft2((image - image.mean()) * window)) ** 2)

    freq_row = np.fft.fftshift(np.fft.fftfreq(rows))[:, None]
    freq_col = np.fft.fftshift(np.fft.fftfreq(cols))[None, :]
    radius = np.sqrt(freq_row**2 + freq_col**2)

    bins = np.linspace(0.0, 0.5, _SPECTRUM_BINS + 1)
    index = np.digitize(radius.ravel(), bins) - 1
    valid = (index >= 0) & (index < _SPECTRUM_BINS)

    totals = np.bincount(index[valid], weights=spectrum.ravel()[valid], minlength=_SPECTRUM_BINS)
    counts = np.bincount(index[valid], minlength=_SPECTRUM_BINS)
    occupied = counts > 0
    if not np.any(occupied):
        return None

    centres = 0.5 * (bins[:-1] + bins[1:])
    profile = np.zeros(_SPECTRUM_BINS, dtype=np.float64)
    profile[occupied] = totals[occupied] / counts[occupied]

    band = occupied & (centres >= _FIT_BAND[0]) & (centres <= _FIT_BAND[1]) & (profile > 0.0)
    if int(np.count_nonzero(band)) < _MIN_FIT_POINTS:
        return None

    abscissa = centres[band] ** 2
    ordinate = np.log(profile[band])
    coefficients = np.polyfit(abscissa, ordinate, 1)
    slope = float(coefficients[0])
    if slope >= 0.0:
        # Power rising with frequency: the fit is meaningless, usually because a
        # lattice impulse dominates the band. Report no estimate rather than a
        # confident wrong one.
        return None

    # Refuse a poorly-fitting line. On a strongly periodic layout the radial
    # profile is a comb of lattice impulses rather than a smooth rolloff, and a
    # least-squares line through it yields a confident, meaningless number.
    # Measured separation is wide: a genuine Gaussian rolloff fits at R^2 above
    # 0.95, whereas a lattice-dominated spectrum sits near 0.7. The same gate
    # also rejects widths whose rolloff falls outside the fit band, where the
    # estimate is unreliable for a different reason.
    residuals = ordinate - np.polyval(coefficients, abscissa)
    total_variance = float(np.var(ordinate))
    if total_variance <= 0.0:
        return None
    if 1.0 - float(np.var(residuals)) / total_variance < _MIN_FIT_R_SQUARED:
        return None

    return float(np.sqrt(-slope / (4.0 * np.pi**2)))


def _snap_to_candidates(sigma: float, candidates: Sequence[float] | None) -> float:
    """Snap a continuous sigma estimate onto an allowed set.

    Parameters
    ----------
    sigma
        Continuous estimate in pixels.
    candidates
        Allowed values, or ``None`` to accept the estimate unchanged.

    Returns
    -------
    float
        The nearest candidate, or ``sigma`` when no candidates are supplied.
    """
    if not candidates:
        return sigma
    return min(candidates, key=lambda value: abs(value - sigma))


def _reference_grid_sigma(target_sigma_search_px: float, scale: float) -> float:
    """Convert a target PSF width in search pixels to the blur to apply before decimation.

    Two conversions, and both matter.

    The blur is applied on the *reference* grid but specified on the *search*
    grid, so it scales up by ``scale``: a 1 px blur in the search image is a
    10 px blur at 1 nm/px.

    Area-average decimation is itself a low-pass. Averaging over a box of width
    ``scale`` has variance ``scale^2 / 12``, so it contributes roughly
    ``1/sqrt(12)`` of a search pixel of blur for free. Applying the full target
    width on top of that would over-blur the template and cost correlation peak.
    Since variances of independent Gaussians add, the width to apply is the
    quadrature *difference*.

    Parameters
    ----------
    target_sigma_search_px
        Desired effective PSF width of the finished template, in search pixels.
    scale
        Decimation ratio.

    Returns
    -------
    float
        Sigma to apply on the reference grid, in reference pixels. Zero when the
        decimation alone already exceeds the target width.

    Notes
    -----
    The box-to-Gaussian equivalence is an approximation, as is treating
    ``cv2.INTER_AREA`` at a non-integer ratio as a plain box. Both are second
    order next to getting the ``scale`` factor right, which is the conversion
    that would otherwise be wrong by 10x.
    """
    target_reference_px = target_sigma_search_px * scale
    decimation_variance = (scale**2) / 12.0
    residual_variance = target_reference_px**2 - decimation_variance
    return float(np.sqrt(residual_variance)) if residual_variance > 0.0 else 0.0


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

    Notes
    -----
    Border handling replicates the edge rather than padding with zeros. A zero
    pad would darken the template's rim and create a synthetic edge exactly
    where the correlation is most sensitive to it.
    """
    working = np.ascontiguousarray(image, dtype=np.float32)
    if sigma_px <= 0.0:
        return working
    blurred: FloatArray = np.asarray(
        cv2.GaussianBlur(
            working,
            ksize=(0, 0),  # derived from sigma
            sigmaX=sigma_px,
            sigmaY=sigma_px,
            borderType=cv2.BORDER_REPLICATE,
        ),
        dtype=np.float32,
    )
    return blurred


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

    working = np.ascontiguousarray(image, dtype=np.float32)
    rows, cols = working.shape
    out_rows = max(1, round(rows / factor))
    out_cols = max(1, round(cols / factor))
    if (out_rows, out_cols) == (rows, cols):
        return working

    # INTER_AREA is the area-average, and is only meaningful when shrinking; it
    # degenerates to nearest-neighbour on enlargement. Enlargement is not a case
    # we expect - the reference is always the finer grid - but the guard keeps
    # the function honest if it is ever called that way.
    interpolation = cv2.INTER_AREA if (out_rows <= rows and out_cols <= cols) else cv2.INTER_LINEAR
    resized: FloatArray = np.asarray(
        cv2.resize(
            working,
            dsize=(out_cols, out_rows),  # cv2 takes (width, height)
            interpolation=interpolation,
        ),
        dtype=np.float32,
    )
    return resized


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

    Notes
    -----
    The rotation centre is ``((cols - 1) / 2, (rows - 1) / 2)``, which is the
    geometric centre *in OpenCV's own indexing*, where pixel ``(i, j)`` sits at
    coordinate ``(j, i)``. It is deliberately **not** taken from
    :func:`src.config.image_centre`: that function encodes how we *report*
    coordinates, which is still an open question with the organizers, whereas
    this is a fact about how ``warpAffine`` addresses pixels. Wiring the two
    together would turn a reporting-convention change into a half-pixel
    resampling shift, which is precisely the class of bug the config module
    exists to prevent.

    Bilinear interpolation, not bicubic. The reference is oversampled tenfold
    relative to the search grid, so everything bilinear attenuates lies above
    the frequency that survives decimation anyway, and bilinear has no negative
    lobes to ring with.

    Shape is preserved, so the corners of the frame necessarily contain fill
    after a rotation. :func:`build_template` crops that fill away rather than
    correlating against it; this function stays shape-preserving because callers
    and tests depend on it.
    """
    working = np.ascontiguousarray(image, dtype=np.float32)
    if theta_deg == 0.0:
        return working

    rows, cols = working.shape
    centre = ((cols - 1) / 2.0, (rows - 1) / 2.0)
    matrix = cv2.getRotationMatrix2D(centre, theta_deg, 1.0)
    rotated: FloatArray = np.asarray(
        cv2.warpAffine(
            working,
            matrix,
            dsize=(cols, rows),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=float(working.mean()),
        ),
        dtype=np.float32,
    )
    return rotated


def _crop_to_valid_rotation(image: FloatArray, theta_deg: float) -> FloatArray:
    """Trim the border that a rotation filled with synthetic values.

    After rotating a frame in place, the corners hold fill rather than data. At
    5 degrees that is on the order of a sixth of the frame. Correlating against
    it is strictly harmful: the fill is constant, so it contributes nothing to
    the zero-mean numerator while still inflating the template's variance in the
    denominator, which dilutes the correlation peak.

    Parameters
    ----------
    image
        Rotated image as ``(rows, cols)``.
    theta_deg
        The rotation that was applied, in degrees.

    Returns
    -------
    FloatArray
        Centred crop containing only real data. Returned unchanged when
        ``theta_deg`` is zero.

    Notes
    -----
    The same number of rows is removed from the top and the bottom, and the same
    number of columns from each side, so the crop's centre coincides exactly
    with the original's. That matters more than the size: an off-centre crop
    would shift the reported match centre by half a pixel per unit of asymmetry,
    which is a systematic error rather than a loss of information.

    The retained fraction is ``1 / (|cos t| + |sin t|)``, the largest
    axis-aligned square inscribed in a square rotated by ``t``. Applying the
    same fraction to both axes is conservative for non-square inputs.
    """
    if theta_deg == 0.0:
        return image

    radians = np.deg2rad(abs(theta_deg) % 90.0)
    retained = 1.0 / (abs(np.cos(radians)) + abs(np.sin(radians)))

    rows, cols = image.shape
    margin_rows = int(np.ceil(rows * (1.0 - retained) / 2.0))
    margin_cols = int(np.ceil(cols * (1.0 - retained) / 2.0))

    # Never crop away everything, however extreme the angle.
    margin_rows = min(margin_rows, (rows - 1) // 2)
    margin_cols = min(margin_cols, (cols - 1) // 2)
    if margin_rows == 0 and margin_cols == 0:
        return image

    return np.ascontiguousarray(
        image[margin_rows : rows - margin_rows, margin_cols : cols - margin_cols]
    )


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
        shape rather than assuming it — both the scale residual and the
        rotation crop change it.

    Raises
    ------
    ValueError
        If ``scale`` is not strictly positive, or ``reference`` is not a finite
        two-dimensional array.

    Notes
    -----
    The order of the four steps is load-bearing.

    1. **Rotate at full reference resolution.** The reference is oversampled
       tenfold relative to the search grid, so resampling error here is far
       below what survives decimation. Rotating afterwards would interpolate a
       signal that has already lost the detail the rotation needs.
    2. **Crop the rotation fill.** Only real data reaches the correlator.
    3. **Match the PSF**, converting the target width from the search grid to
       the reference grid and discounting the blur that decimation supplies for
       free. See :func:`_reference_grid_sigma`.
    4. **Decimate by area-averaging**, never by an interpolating kernel with
       negative lobes.

    Steps 3 and 4 cannot be swapped: blurring after decimation would apply the
    low-pass to a signal that has already been aliased, which does not undo the
    aliasing.

    The template's centre corresponds exactly to the reference's centre. The
    rotation is about the centre and the crop is symmetric, so no step
    introduces a translation — which is what allows the match position to be
    reported directly in search-image coordinates.
    """
    if scale <= 0.0:
        msg = f"scale must be strictly positive, got {scale!r}"
        raise ValueError(msg)

    working = _as_working_array(reference, "reference")
    rotated = rotate_image(working, theta)
    cropped = _crop_to_valid_rotation(rotated, theta)
    blurred = match_psf(cropped, _reference_grid_sigma(psf_sigma_px, scale))
    return area_average_downsample(blurred, scale)


def build_weight(
    weight_map: FloatArray,
    theta: float,
    scale: float,
) -> FloatArray:
    """Carry a reference-resolution weight map onto the template grid.

    ``uniqueness_map`` scores the reference at 1 nm/px, but the correlation
    consumes a weight shaped like the template, roughly ten times smaller. The
    weight must land on the same grid as the template and stay aligned with it
    pixel for pixel, so it goes through the *same* rotation, the same valid-area
    crop and the same area-average decimation.

    Parameters
    ----------
    weight_map
        Uniqueness weights at reference resolution, same shape as the reference.
    theta
        Rotation in degrees. Must match the value given to
        :func:`build_template`.
    scale
        Decimation ratio. Must match the value given to :func:`build_template`.

    Returns
    -------
    FloatArray
        Weights on the template grid, ``float32``, the same shape
        :func:`build_template` produces for the same ``theta`` and ``scale``.

    Raises
    ------
    ValueError
        If ``scale`` is not strictly positive, or the map is not a finite
        two-dimensional array.

    Notes
    -----
    Two steps of the image pipeline are deliberately **not** applied.

    No PSF blur. The point-spread function models how the instrument smeared the
    *signal*; the weight map is not signal, it is a statement about which parts
    of the reference are informative. Blurring it would bleed an anchor's
    importance into neighbouring periodic regions and dilute exactly the
    discrimination the weighting exists to provide.

    No standardisation. The weights are a non-negative importance profile, not
    an intensity field. Centring them would make some negative, which is
    meaningless in a weighted variance, and rescaling them is pointless because
    :func:`_zncc_masked_fft` normalises by their sum regardless.

    Area-averaging *is* applied, and is the right decimation here for the same
    reason as for the image: the coarse-grid weight of a cell should be the mean
    importance of the fine pixels that fall inside it.
    """
    if scale <= 0.0:
        msg = f"scale must be strictly positive, got {scale!r}"
        raise ValueError(msg)

    working = _as_working_array(weight_map, "weight_map")
    rotated = rotate_image(working, theta)
    cropped = _crop_to_valid_rotation(rotated, theta)
    return area_average_downsample(cropped, scale)


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


def _standardise(image: FloatArray) -> FloatArray:
    """Centre and scale an image to unit variance.

    Parameters
    ----------
    image
        Input array.

    Returns
    -------
    FloatArray
        ``(image - mean) / std``, or the centred image when the standard
        deviation is degenerate.

    Notes
    -----
    Purely a conditioning step, and free of consequence: weighted ZNCC is
    invariant to any positive affine rescaling of either input, so this cannot
    change the result. It matters because the weighted formulation accumulates
    ``sum(w * S**2)`` and then subtracts ``(sum(w * S))**2``. On raw 8-bit data
    those terms are of order 1e4 and nearly equal, and float32 carries about
    seven significant digits, so the difference loses most of its precision to
    cancellation. Standardising first keeps both terms of order one.
    """
    centred = image - float(np.mean(image, dtype=np.float64))
    spread = float(np.std(centred, dtype=np.float64))
    if spread < _MIN_TEMPLATE_STD:
        return np.ascontiguousarray(centred, dtype=np.float32)
    return np.ascontiguousarray(centred / spread, dtype=np.float32)


def _zncc_masked_fft(
    template: FloatArray,
    search: FloatArray,
    weight: FloatArray,
) -> FloatArray:
    """Weighted zero-mean normalized cross-correlation.

    Extends ZNCC with a per-pixel weight over the template, so that informative
    regions of the reference count for more than periodic ones. Every mean and
    variance below is taken *under the weights* rather than uniformly.

    With ``w`` normalised to sum to one, writing ``mu_T = sum(w * T)`` and
    ``S_uv`` for the search window at ``(u, v)``:

    - numerator   ``sum(w * (T - mu_T) * (S_uv - mu_S))``
    - denominator ``sqrt(sum(w * (T - mu_T)**2) * sum(w * (S_uv - mu_S)**2))``

    Expanding removes every per-position sum over the template, leaving three
    plain cross-correlations that OpenCV evaluates directly::

        C1 = xcorr(S,    w * T)      -> sum(w * T * S_uv)
        C2 = xcorr(S,    w)          -> sum(w * S_uv)      = mu_S
        C3 = xcorr(S**2, w)          -> sum(w * S_uv**2)

        numerator   = C1 - mu_T * C2
        var_T       = sum(w * T**2) - mu_T**2        (a scalar)
        var_S       = C3 - C2**2

    Parameters
    ----------
    template
        Template as ``(t_rows, t_cols)``, contiguous ``float32``.
    search
        Search image as ``(rows, cols)``, contiguous ``float32``.
    weight
        Per-pixel weights over the template, same shape as ``template``.
        Non-negative; scale is irrelevant because the weights are normalised
        internally by their sum.

    Returns
    -------
    FloatArray
        Correlation surface, ``float32``, values in ``[-1, 1]``.

    Notes
    -----
    Normalising by the weight sum is what makes a *constant* weight map
    reproduce plain ZNCC exactly: every weighted mean collapses to the ordinary
    mean and the common factor cancels between numerator and denominator. That
    equivalence is the contract with the unweighted path and is asserted
    directly by the tests, which is what makes this implementation checkable at
    all while ``uniqueness_map`` still returns a constant.

    The weights are clipped at zero. A negative weight would make the
    denominator's variance term meaningless, and the interface promises
    non-negative values, so a negative one is a caller error rather than a case
    to model.
    """
    t_rows, t_cols = template.shape
    rows, cols = search.shape
    surface_shape_ = (rows - t_rows + 1, cols - t_cols + 1)

    weights = np.clip(weight, 0.0, None).astype(np.float32)
    total = float(np.sum(weights, dtype=np.float64))
    if total <= 0.0:
        return np.zeros(surface_shape_, dtype=np.float32)
    weights = np.ascontiguousarray(weights / total, dtype=np.float32)

    template_n = _standardise(template)
    search_n = _standardise(search)

    mean_t = float(np.sum(weights * template_n, dtype=np.float64))
    var_t = float(np.sum(weights * template_n.astype(np.float64) ** 2)) - mean_t**2
    if var_t < _MIN_WEIGHTED_VARIANCE:
        # Under these weights the template is effectively flat, so it localises
        # nothing. Zero is the honest surface, matching the unweighted guard.
        return np.zeros(surface_shape_, dtype=np.float32)

    kernel_wt = np.ascontiguousarray(weights * template_n, dtype=np.float32)
    corr_wt = cv2.matchTemplate(search_n, kernel_wt, cv2.TM_CCORR)
    corr_w = cv2.matchTemplate(search_n, weights, cv2.TM_CCORR)
    corr_w_sq = cv2.matchTemplate(
        np.ascontiguousarray(search_n * search_n, dtype=np.float32),
        weights,
        cv2.TM_CCORR,
    )

    numerator = corr_wt - mean_t * corr_w
    var_s = np.clip(corr_w_sq - corr_w * corr_w, 0.0, None)

    denominator = np.sqrt(var_t * var_s, dtype=np.float32)
    surface: FloatArray = np.zeros(surface_shape_, dtype=np.float32)
    np.divide(numerator, denominator, out=surface, where=denominator > _MIN_WEIGHTED_VARIANCE)

    np.nan_to_num(surface, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
    np.clip(surface, -1.0, 1.0, out=surface)
    return surface


# ===========================================================================
# Stage 3b — peak extraction
# ===========================================================================


def _local_maxima(surface: FloatArray, radius: int) -> BoolArray:
    """Mark local maxima within a square neighbourhood.

    Parameters
    ----------
    surface
        Correlation surface.
    radius
        Neighbourhood half-width in pixels. A radius of zero marks every
        position, since each is trivially the maximum of a neighbourhood
        containing only itself.

    Returns
    -------
    BoolArray
        Mask, same shape as ``surface``, true where a position equals the
        maximum over its neighbourhood.

    Raises
    ------
    ValueError
        If ``radius`` is negative.

    Notes
    -----
    The neighbourhood is square, not circular, and the suppression in
    :func:`top_k_peaks` uses the same square metric. That is deliberate: an
    orthogonal DRAM lattice has square cells, so a Chebyshev radius of half the
    pitch covers exactly one cell. A Euclidean radius of the same size would
    leave the cell corners unsuppressed and admit duplicate candidates from a
    single cell.

    A position must also have *variation* in its neighbourhood. Equalling the
    neighbourhood maximum is not sufficient, because every point of a constant
    region trivially satisfies it: a surface with a flat background would report
    hundreds of thousands of "maxima" carrying no information. Requiring the
    neighbourhood range to be non-zero excludes flat regions while still marking
    the rim of a genuine plateau, which the greedy suppression in
    :func:`top_k_peaks` then thins to one representative.

    A radius of zero is exempt from the variation test and marks everything,
    which is the documented meaning of "no suppression".
    """
    if radius < 0:
        msg = f"radius must be non-negative, got {radius!r}"
        raise ValueError(msg)
    if radius == 0:
        return np.ones(surface.shape, dtype=np.bool_)

    # cv2.dilate and cv2.erode over a rectangular kernel are exactly the sliding
    # maximum and minimum. They are used in preference to scipy's rank filters
    # because OpenCV's implementation is O(1) per pixel in the kernel size,
    # whereas scipy's cost grows with it: at radius 8 on a 901x901 surface that
    # is the difference between roughly 2 ms and roughly 28 ms, which matters
    # when the whole Robust-mode budget is 100 ms.
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    neighbourhood_max = cv2.dilate(surface, kernel, borderType=cv2.BORDER_REPLICATE)
    neighbourhood_min = cv2.erode(surface, kernel, borderType=cv2.BORDER_REPLICATE)

    is_maximum = surface >= neighbourhood_max
    has_variation = (neighbourhood_max - neighbourhood_min) > _FLAT_SURFACE_ATOL
    return np.asarray(is_maximum & has_variation, dtype=np.bool_)


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
        At most ``k`` peaks, sorted by descending score, each separated from
        every other by more than ``nms_radius`` in Chebyshev distance. Returns
        fewer when the surface holds fewer separated maxima; the list is never
        padded. Returns an empty list when ``k`` is non-positive, the surface is
        empty, or the surface is flat.

    Raises
    ------
    ValueError
        If ``nms_radius`` is negative.

    Notes
    -----
    A flat surface returns no candidates at all. That is the honest answer — if
    every position scores identically then none is distinguishable — and it lets
    the caller report an SNR collapse rather than invent a peak. It also avoids a
    pathological case: on a perfectly flat surface every position is technically
    a local maximum, so the candidate list would run to hundreds of thousands of
    entries carrying no information.
    """
    if nms_radius < 0:
        msg = f"nms_radius must be non-negative, got {nms_radius!r}"
        raise ValueError(msg)
    if k <= 0 or surface.size == 0:
        return []

    if float(surface.max()) - float(surface.min()) <= _FLAT_SURFACE_ATOL:
        return []

    candidates = np.flatnonzero(_local_maxima(surface, nms_radius))
    scores = surface.ravel()[candidates]
    # Stable ordering so that equal scores resolve in raster order, which keeps
    # the output reproducible across runs and platforms.
    order = np.argsort(-scores, kind="stable")

    rows, cols = surface.shape
    suppressed = np.zeros(surface.shape, dtype=np.bool_)
    peaks: list[Peak] = []

    for position in order:
        row, col = divmod(int(candidates[position]), cols)
        if suppressed[row, col]:
            continue

        peaks.append(Peak(col=col, row=row, score=float(scores[position])))
        if len(peaks) >= k:
            break

        suppressed[
            max(0, row - nms_radius) : min(rows, row + nms_radius + 1),
            max(0, col - nms_radius) : min(cols, col + nms_radius + 1),
        ] = True

    return peaks


# ===========================================================================
# Stage 5 — subpixel refinement
# ===========================================================================


def _check_upsample(upsample: int) -> None:
    """Validate the interpolation factor.

    Parameters
    ----------
    upsample
        Requested factor.

    Raises
    ------
    ValueError
        If ``upsample`` is less than 1.
    """
    if upsample < 1:
        msg = f"upsample must be at least 1, got {upsample!r}"
        raise ValueError(msg)


def _bounded(refinement: SubpixelRefinement) -> SubpixelRefinement:
    """Reject a refinement that has wandered further than a refinement should.

    Parameters
    ----------
    refinement
        Candidate offset.

    Returns
    -------
    SubpixelRefinement
        The input when the offset is plausible, otherwise a zero offset marked
        ``"rejected"``.

    Notes
    -----
    Stage 3b already located the peak to the nearest whole pixel, so a genuine
    residual is under a pixel. Anything larger means the refinement latched onto
    noise rather than the true maximum. Measured at high noise, unbounded phase
    correlation produced offsets of ten pixels and more; applying one would turn
    a correct integer answer into a badly wrong sub-pixel one. Declining to
    refine is strictly better than that.
    """
    if not (np.isfinite(refinement.dx) and np.isfinite(refinement.dy)):
        return SubpixelRefinement.none("rejected")
    if max(abs(refinement.dx), abs(refinement.dy)) > _MAX_SUBPIXEL_OFFSET_PX:
        return SubpixelRefinement.none("rejected")
    return refinement


@lru_cache(maxsize=8)
def _hann_window(shape: Shape2D) -> FloatArray:
    """Return a separable Hann window of the given shape.

    Parameters
    ----------
    shape
        Window shape as ``(rows, cols)``.

    Returns
    -------
    FloatArray
        Two-dimensional Hann window, ``float32``.

    Notes
    -----
    Cached because the template shape is constant across a whole run, and
    rebuilding the window per call would be pure waste.

    The window matters more than it looks. A discrete Fourier transform treats
    its input as periodic, so a patch whose opposite edges do not match presents
    a step discontinuity, and that step leaks broadband energy into the spectrum.
    In registration the effect is a systematic *underestimate* of the shift,
    because the mismatched edges pull the correlation peak toward zero. Measured
    on fractional shifts of a band-limited field, an unwindowed patch recovered
    0.39 px of a true 0.50 px offset; windowed, it recovers 0.49 px.
    """
    rows, cols = shape
    return np.asarray(np.outer(np.hanning(rows), np.hanning(cols)), dtype=np.float32)


def _refine_by_phase_correlation(
    template: FloatArray,
    search: FloatArray,
    peak: Peak,
    upsample: int,
) -> SubpixelRefinement:
    """Register the template against the winning crop by upsampled-DFT correlation.

    Parameters
    ----------
    template
        Template from :func:`build_template`.
    search
        Search image.
    peak
        Winning integer-valued peak, locating the crop.
    upsample
        Interpolation factor.

    Returns
    -------
    SubpixelRefinement
        Residual offset and registration error.

    Notes
    -----
    ``normalization=None`` is passed deliberately. scikit-image defaults to
    ``"phase"``, which whitens the spectrum before correlating; that is the
    classic phase-correlation formulation and it is markedly less robust to
    noise, because whitening amplifies exactly the high frequencies that noise
    dominates. Our search image is by design the noisier of the two captures, so
    this is the wrong default for us. Measured on fractional shifts of a
    band-limited field:

    ======  ==================  ==================
    noise   ``"phase"`` error   ``None`` error
    ======  ==================  ==================
    0.0     0.010 px            0.003 px
    0.2     0.574 px            0.168 px
    0.5     10.671 px           0.622 px
    ======  ==================  ==================

    ``normalization=None`` is also the formulation of the published method this
    stage cites, and it is the only setting under which the returned error means
    anything: under ``"phase"`` it is a constant 1.0.
    """
    rows, cols = template.shape
    crop = search[peak.row : peak.row + rows, peak.col : peak.col + cols]
    if crop.shape != template.shape:
        return SubpixelRefinement.none("rejected")

    # A patch with no variance carries no registrable structure. Declining here
    # rather than letting the registration run keeps the never-raises contract
    # clean and avoids a library warning about an undefined error metric.
    if min(float(np.std(template)), float(np.std(crop))) < _MIN_TEMPLATE_STD:
        return SubpixelRefinement.none("rejected")

    window = _hann_window(template.shape)
    # scikit-image ships only partial annotations, so this call is untyped as
    # far as mypy is concerned. The return shape is pinned by the unpacking and
    # the float conversions below.
    shift, error, _ = phase_cross_correlation(  # type: ignore[no-untyped-call]
        (crop - crop.mean()) * window,
        (template - template.mean()) * window,
        upsample_factor=upsample,
        normalization=None,
    )
    return _bounded(
        SubpixelRefinement(
            dx=float(shift[1]),
            dy=float(shift[0]),
            error=float(error),
            method="phase_cross_correlation",
        )
    )


def _upsampled_patch(
    patch: FloatArray,
    upsample: int,
    span: float,
) -> tuple[Float64Array, Float64Array]:
    """Evaluate the band-limited interpolation of a patch on a fine grid.

    Zero-padding a spectrum is exact sinc interpolation of a band-limited
    signal. Rather than materialising a large padded transform, the inverse
    DFT is evaluated directly at the fine sample positions by matrix
    multiplication, which costs a few hundred operations instead of a few
    hundred thousand.

    Parameters
    ----------
    patch
        Small square neighbourhood of the correlation surface.
    upsample
        Samples per original pixel.
    span
        Half-width of the evaluated region, in original pixels, measured from
        the patch centre.

    Returns
    -------
    tuple
        ``(values, offsets)`` where ``values`` is the interpolated surface and
        ``offsets`` are the sample positions relative to the patch centre.
    """
    n_rows, n_cols = patch.shape
    spectrum = np.fft.fft2(patch)

    steps = int(round(2.0 * span * upsample)) + 1
    offsets = np.linspace(-span, span, steps)
    centre_row = (n_rows - 1) / 2.0
    centre_col = (n_cols - 1) / 2.0

    row_kernel = np.exp(2j * np.pi * np.outer(centre_row + offsets, np.fft.fftfreq(n_rows)))
    col_kernel = np.exp(2j * np.pi * np.outer(np.fft.fftfreq(n_cols), centre_col + offsets))

    values = np.real(row_kernel @ spectrum @ col_kernel) / (n_rows * n_cols)
    return values, offsets


def _refine_by_surface_upsampling(
    surface: FloatArray,
    peak: Peak,
    upsample: int,
) -> SubpixelRefinement:
    """Refine a peak by interpolating the correlation surface around it.

    Parameters
    ----------
    surface
        Correlation surface from :func:`zncc_surface`.
    peak
        Winning integer-valued peak.
    upsample
        Interpolation factor.

    Returns
    -------
    SubpixelRefinement
        Residual offset. The error is reported as ``1 - peak`` so that, like the
        registration error, smaller means better.

    Notes
    -----
    Interpolates the ZNCC surface itself, whose shape near the maximum is
    slightly distorted by the local-variance denominator of the normalization.
    That makes it the less accurate of the two routines, which is why it is the
    fallback rather than the primary.
    """
    radius = _SURFACE_PATCH_RADIUS
    rows, cols = surface.shape
    if peak.row - radius < 0 or peak.row + radius >= rows:
        return SubpixelRefinement.none("rejected")
    if peak.col - radius < 0 or peak.col + radius >= cols:
        return SubpixelRefinement.none("rejected")

    patch = np.ascontiguousarray(
        surface[
            peak.row - radius : peak.row + radius + 1,
            peak.col - radius : peak.col + radius + 1,
        ],
        dtype=np.float32,
    )
    values, offsets = _upsampled_patch(patch, upsample, span=1.0)
    row_index, col_index = np.unravel_index(int(np.argmax(values)), values.shape)

    return _bounded(
        SubpixelRefinement(
            dx=float(offsets[col_index]),
            dy=float(offsets[row_index]),
            error=float(np.clip(1.0 - values[row_index, col_index], 0.0, 1.0)),
            method="surface_upsampling",
        )
    )


def refine_subpixel_detailed(
    template: FloatArray,
    search: FloatArray,
    peak: Peak,
    surface: FloatArray | None = None,
    upsample: int = config.DEFAULT_UPSAMPLE,
) -> SubpixelRefinement:
    """Refine a peak to sub-pixel precision, reporting the error alongside.

    Runs the primary routine — upsampled-DFT registration of the template
    against the winning crop — and falls back to interpolating the correlation
    surface when that is rejected or unavailable.

    Parameters
    ----------
    template
        Template from :func:`build_template`.
    search
        Search image.
    peak
        Winning integer-valued peak.
    surface
        Correlation surface, used only for the fallback. When ``None`` there is
        no fallback and a rejected primary yields a zero offset.
    upsample
        Interpolation factor; 100 gives 1/100 px resolution.

    Returns
    -------
    SubpixelRefinement
        Residual offset ``(dx, dy)`` in search pixels, the registration error,
        and which routine produced it.

    Raises
    ------
    ValueError
        If ``upsample`` is less than 1.
    """
    _check_upsample(upsample)

    primary = _refine_by_phase_correlation(template, search, peak, upsample)
    if primary.method != "rejected":
        return primary
    if surface is None:
        return primary
    return _refine_by_surface_upsampling(surface, peak, upsample)


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

    See Also
    --------
    refine_subpixel_detailed : same refinement, reporting the registration error.
    """
    _check_upsample(upsample)
    refinement = _refine_by_surface_upsampling(surface, peak, upsample)
    return (refinement.dx, refinement.dy)


def refine_subpixel_crop(
    template: FloatArray,
    search: FloatArray,
    peak: Peak,
    upsample: int = config.DEFAULT_UPSAMPLE,
) -> tuple[float, float]:
    """Refine a peak by registering the template against the winning crop.

    The primary Stage 5 routine. Applies upsampled-DFT registration to the image
    data directly, rather than to a correlation surface whose shape is distorted
    by the local-variance denominator of the ZNCC normalization.

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

    See Also
    --------
    refine_subpixel_detailed : same refinement, reporting the registration error.
    """
    _check_upsample(upsample)
    refinement = _refine_by_phase_correlation(template, search, peak, upsample)
    return (refinement.dx, refinement.dy)


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
