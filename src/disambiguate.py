"""Stage 4 — turning "the answer is in the top-K" into "the answer is top-1".

Owned by R3. This file is a stub placed by R4 so that the pipeline imports and
runs end-to-end; R3 replaces the bodies without changing the signatures.

The problem
-----------
A periodic pattern correlated against a periodic field produces a *lattice* of
near-equal peaks, not one peak. For a 40 nm word-line pitch the search image
spans 250 lattice cells and the template 25, giving on the order of 50,000
placements that look essentially identical. Recall is easy; picking the right
one is the intellectual crux of the project.

Four layers, applied in order, each cheap
-----------------------------------------
(a) **Uniqueness weighting.** Not all parts of the reference are equally
    informative. Periodic regions match everywhere; aperiodic regions match in
    one place. Weight the matched filter toward the informative parts. When an
    anchor exists in frame this collapses the tied candidates to one.

(b) **Peak-ratio test.** The peak-to-sidelobe ratio is a detection statistic,
    structurally identical to CFAR thresholding in radar. Above threshold,
    accept and stop; below, keep escalating. The threshold must be validated
    across the whole noise range, not fitted at one level.

(c) **Learned re-ranker** *(optional)*. Only if (a) and (b) leave residual
    ambiguity. Ranking a shortlist, not searching 800,000 positions.

(d) **Centre tie-break** *(mandated)*. Among candidates statistically
    indistinguishable from the best, select the one nearest the search-image
    centre. Required by the problem statement, and physically correct: the stage
    aimed at the field-of-view centre, so the true site carries a centre prior.

Status
------
Stubs. ``uniqueness_map`` returns uniform weights, which makes masked
correlation reduce exactly to unmasked correlation — a useful equivalence test
for R4's Stage 4a work. ``select_candidate`` returns the strongest peak, which
is precisely the plain-NCC baseline behaviour, so the stub doubles as the
incumbent we are required to beat.
"""

import numpy as np

from src import config
from src.types import FloatArray, Peak

__all__ = ["peak_to_sidelobe", "select_candidate", "uniqueness_map"]


def uniqueness_map(
    reference: FloatArray,
    tile: int = config.DEFAULT_UNIQUENESS_TILE_PX,
) -> FloatArray:
    """Score how uniquely each region of the reference identifies a position.

    Tiles the reference and scores each tile by how sharply it autocorrelates —
    via the peak-to-sidelobe ratio of its own autocorrelation, or equivalently
    its spectral flatness. Periodic tiles have tall sidelobes and peaky spectra;
    aperiodic tiles do not.

    Parameters
    ----------
    reference
        Reference image at 1 nm/px, as ``(rows, cols)``.
    tile
        Tile edge length in reference pixels.

    Returns
    -------
    FloatArray
        Weight map at **reference resolution**, same shape as ``reference``,
        ``float32``, with values in ``[0, 1]``.

    Notes
    -----
    Resolution mismatch, worth stating plainly: this map is produced at
    reference resolution but :func:`src.matcher.zncc_surface` consumes a weight
    array shaped like the *template*, which is roughly ten times smaller. The
    decimation belongs on R4's side, inside the Stage 2 path, so that the weight
    goes through the same area-averaging as the image data and stays aligned
    with it. Returning the map at reference resolution keeps that decision in
    one place.
    """
    del tile  # stub: uniqueness scoring is R3's work
    return np.ones(reference.shape, dtype=np.float32)


def peak_to_sidelobe(
    surface: FloatArray,
    peak: Peak,
    exclusion_radius: int = config.DEFAULT_NMS_RADIUS_PX,
) -> float:
    """Measure how far the winning peak stands above the correlation background.

    ``PSR = (peak - mean_sidelobe) / std_sidelobe``, where the sidelobe region is
    the correlation surface outside the exclusion radius around the peak.

    Parameters
    ----------
    surface
        Correlation surface from :func:`src.matcher.zncc_surface`.
    peak
        The candidate being assessed.
    exclusion_radius
        Radius around the peak excluded from the sidelobe statistics, in surface
        pixels. Should match the non-maximum-suppression radius so that the
        peak's own shoulder is not counted as background.

    Returns
    -------
    float
        Peak-to-sidelobe ratio. Larger means less ambiguous. Returns
        ``float("nan")`` when the sidelobe region is empty or has zero
        variance, signalling an absent measurement rather than a poor
        one. R4's escalation logic treats NaN as unknown and escalates.

    Notes
    -----
    Ownership is still open between R3 and R4. R4 needs this for the Stage 6
    confidence features and for the fast-to-robust escalation trigger; R3 needs
    it for the peak-ratio test. It lives here, in R3's module, so there is one
    implementation rather than two that drift apart.
    """
    peak_value = float(surface[peak.row, peak.col])

    rows, cols = np.ogrid[: surface.shape[0], : surface.shape[1]]
    sq_dist = (rows - peak.row) ** 2 + (cols - peak.col) ** 2
    sidelobe = surface[sq_dist > exclusion_radius**2]

    if sidelobe.size == 0:
        return float("nan")

    sidelobe_mean = float(np.nanmean(sidelobe))
    sidelobe_std = float(np.nanstd(sidelobe))

    if not np.isfinite(sidelobe_std) or sidelobe_std == 0.0:
        return float("nan")

    return float((peak_value - sidelobe_mean) / sidelobe_std)


def select_candidate(
    peaks: list[Peak],
    image_centre: tuple[float, float],
    tolerance: float = config.TIE_SIGMA,
) -> tuple[Peak, bool]:
    """Choose one candidate from the shortlist, applying the mandated tie-break.

    Parameters
    ----------
    peaks
        Candidates from :func:`src.matcher.top_k_peaks`, sorted by descending
        score.
    image_centre
        Centre of the **search** image as ``(x, y)``, from
        :func:`src.config.image_centre`. The problem statement specifies the
        search image, not the reference.
    tolerance
        Tie width. Candidates scoring within this margin of the best are treated
        as statistically indistinguishable and go to the centre tie-break.
        Deliberately not "every local maximum", which would hand the tie-break
        far more cases than it should decide.

    Returns
    -------
    tuple of (Peak, bool)
        The selected candidate, and whether the centre tie-break decided it. The
        flag is reported in the diagnostics: a result decided by the tie-break
        rather than by correlation evidence is a qualitatively weaker answer and
        the caller should know.

    Raises
    ------
    ValueError
        If ``peaks`` is empty. Callers must handle an empty shortlist before
        reaching here; there is no sensible candidate to invent.
    """
    del image_centre, tolerance  # stub: tie-break logic is R3's work
    if not peaks:
        msg = "select_candidate requires at least one candidate peak"
        raise ValueError(msg)
    return (peaks[0], False)
