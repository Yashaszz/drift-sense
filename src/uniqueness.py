"""Stage 4a — uniqueness weighting.

Owned by R3. Lives in its own module so that :mod:`src.disambiguate` stays a
thin Stage 4 surface; ``disambiguate.uniqueness_map`` re-exports from here, so
the frozen interface R4 codes against is unchanged.
"""

import numpy as np
from scipy.ndimage import gaussian_filter

from src import config
from src.types import FloatArray

__all__ = ["uniqueness_map", "uniqueness_score"]


def uniqueness_map(
    reference: FloatArray,
    tile: int = config.DEFAULT_UNIQUENESS_TILE_PX,
    *,
    prefilter_sigma_px: float = config.UNIQUENESS_PREFILTER_SIGMA_PX,
    floor: float = config.UNIQUENESS_FLOOR,
) -> FloatArray:
    """Score how uniquely each region of the reference identifies a position.

    Tiles the reference and scores each tile by the peak-to-sidelobe ratio of
    its own autocorrelation. A tile's autocorrelation *is* its ambiguity
    function as a matched filter: a periodic tile correlates against itself at
    every lattice offset and so carries sidelobes close to the zero-lag peak;
    an aperiodic tile does not. Uniqueness is ``1 - max_normalised_sidelobe``.

    Parameters
    ----------
    reference
        Reference image at 1 nm/px, as ``(rows, cols)``.
    tile
        Tile edge length in reference pixels. Tiles overlap at 50% stride.
    prefilter_sigma_px
        Gaussian low-pass width applied before scoring, in reference pixels.
        Set near the 10x decimation Nyquist so that sensor noise — maximally
        aperiodic, and therefore maximally misleading to this statistic — is
        removed before autocorrelation. Pass ``0.0`` to disable.
    floor
        Minimum weight. Keeps the map strictly positive so weighted
        correlation stays defined everywhere and no region is hard-excluded on
        the strength of one tile score.

    Returns
    -------
    FloatArray
        Weight map at **reference resolution**, same shape as ``reference``,
        ``float32``, values in ``[floor, 1.0]``. Never NaN, never all-zero.

    Notes
    -----
    **Absolute, not per-image normalised.** The map is deliberately not
    rescaled so its maximum is 1.0. A fully periodic reference with no anchor
    in frame correctly returns a near-flat map, and masked correlation then
    degrades gracefully to unmasked. Per-image renormalisation would
    manufacture an anchor out of the least-periodic tile and hand the matcher
    a confident wrong answer.

    **Resolution.** Produced at reference resolution;
    :func:`src.matcher.build_weight` carries it onto the template grid through
    the same rotation, valid-area crop and area-average decimation as
    :func:`src.matcher.build_template`.

    **Pitch constraint.** The autocorrelation exclusion radius derives from
    ``prefilter_sigma_px`` and must stay well below the smallest layout pitch
    in reference pixels (>= 40 px at 1 nm/px for a 40 nm word-line pitch). If
    it exceeded the pitch, the periodic sidelobe would be excluded from the
    statistic and periodic tiles would score as unique.
    """
    ref = np.asarray(reference, dtype=np.float32)
    if ref.ndim != 2:
        msg = f"reference must be 2-D, got shape {ref.shape}"
        raise ValueError(msg)

    rows, cols = ref.shape
    tile = int(min(int(tile), rows, cols))
    if tile < config.UNIQUENESS_MIN_TILE_PX:
        return np.ones(ref.shape, dtype=np.float32)

    work = ref - float(ref.mean())
    if prefilter_sigma_px > 0.0:
        work = gaussian_filter(work, sigma=float(prefilter_sigma_px), mode="nearest")

    stride = max(tile // 2, 1)
    row_starts = _tile_starts(rows, tile, stride)
    col_starts = _tile_starts(cols, tile, stride)

    taper = np.outer(np.hanning(tile), np.hanning(tile)).astype(np.float32)
    exclusion = max(int(round(3.0 * prefilter_sigma_px)), 2)

    coarse = np.empty((len(row_starts), len(col_starts)), dtype=np.float32)
    for i, r0 in enumerate(row_starts):
        for j, c0 in enumerate(col_starts):
            coarse[i, j] = _tile_uniqueness(work[r0 : r0 + tile, c0 : c0 + tile], taper, exclusion)

    full = _expand(coarse, row_starts, col_starts, tile, (rows, cols))
    return np.clip(full, float(floor), 1.0).astype(np.float32)


def uniqueness_score(weight_map: FloatArray) -> float:
    """Collapse a uniqueness map to the scalar diagnostic.

    Parameters
    ----------
    weight_map
        Output of :func:`uniqueness_map`.

    Returns
    -------
    float
        Spread between the most informative regions and the typical region, as
        ``p99 - median``. Near zero means the reference is uniformly periodic
        and carries no anchor: the position is genuinely underdetermined by
        correlation evidence. Returns ``float("nan")`` when the map is empty or
        wholly non-finite, matching the absent-measurement convention.

    Notes
    -----
    Validation target: this should separate R1's anchored stratum from the
    unanchored one. If it does not, the map is not working, whatever top-1 does.
    """
    values = np.asarray(weight_map, dtype=np.float64).ravel()
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.percentile(finite, 99.0) - np.median(finite))


def _tile_starts(extent: int, tile: int, stride: int) -> list[int]:
    """List tile origins covering ``extent``, with the last tile flush to the edge."""
    if extent <= tile:
        return [0]
    starts = list(range(0, extent - tile + 1, stride))
    if starts[-1] != extent - tile:
        starts.append(extent - tile)
    return starts


def _tile_uniqueness(patch: FloatArray, taper: FloatArray, exclusion: int) -> float:
    """Score one tile as ``1 - max normalised autocorrelation sidelobe``."""
    x = (patch - float(patch.mean())) * taper
    zero_lag = float(np.sum(x * x))
    if not np.isfinite(zero_lag) or zero_lag <= 0.0:
        return 0.0

    spectrum = np.fft.rfft2(x)
    acf = np.fft.fftshift(np.fft.irfft2(np.abs(spectrum) ** 2, s=x.shape)) / zero_lag

    n_rows, n_cols = acf.shape
    rr, cc = np.ogrid[:n_rows, :n_cols]
    chebyshev = np.maximum(np.abs(rr - n_rows // 2), np.abs(cc - n_cols // 2))
    sidelobe = acf[chebyshev > exclusion]

    if sidelobe.size == 0:
        return 0.0
    peak_sidelobe = float(np.nanmax(sidelobe))
    if not np.isfinite(peak_sidelobe):
        return 0.0
    return float(np.clip(1.0 - peak_sidelobe, 0.0, 1.0))


def _expand(
    coarse: FloatArray,
    row_starts: list[int],
    col_starts: list[int],
    tile: int,
    shape: tuple[int, int],
) -> FloatArray:
    """Bilinearly expand per-tile scores back to full reference resolution.

    Not a blocky repeat: a step-function weight map carries edges that are not
    in the image, on a grid whose period is the tile stride. Under correlation
    that is synthetic periodic structure added to the matched filter — the
    exact thing this stage exists to remove.
    """
    rows, cols = shape
    half = (tile - 1) / 2.0
    r_centres = np.asarray(row_starts, dtype=np.float64) + half
    c_centres = np.asarray(col_starts, dtype=np.float64) + half

    if coarse.shape[1] == 1:
        by_col = np.repeat(coarse, cols, axis=1)
    else:
        grid_c = np.arange(cols, dtype=np.float64)
        by_col = np.stack([np.interp(grid_c, c_centres, line) for line in coarse])

    if coarse.shape[0] == 1:
        return np.repeat(by_col, rows, axis=0).astype(np.float32)

    grid_r = np.arange(rows, dtype=np.float64)
    return np.stack([np.interp(grid_r, r_centres, line) for line in by_col.T], axis=1).astype(
        np.float32
    )
