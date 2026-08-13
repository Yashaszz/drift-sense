"""Stage 4 — turning "the answer is in the top-K" into "the answer is top-1".

Owned by R3. Began as a stub placed by R4 so the pipeline imported and ran
end-to-end; the bodies are real since PR #13, and the signatures never changed.

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
Implemented, and measured on the 324-pair set (``results/ablation_324.csv``,
``results/full_324.csv``). Layers (a), (b) and (d) are live; the learned
re-ranker (c) was never needed and does not exist.

**(a) Uniqueness weighting is the entire disambiguation gain**: +0.080 on the
anchored stratum, 0.753 → 0.833. ``uniqueness_map`` and ``uniqueness_score``
are real — they live in :mod:`src.uniqueness` and are re-exported here so R4's
imports did not have to change — and the map is emphatically *not* uniform.
Cite the anchored stratum: across all 324 the unanchored half is pinned at
0.000 by construction and hides the effect.

**(b) PSR does not separate correct from incorrect on periodic layouts.** The
sidelobe region contains genuine lattice peaks rather than noise, so the
background the statistic normalises against is itself signal. With the
thresholds at 8.0/4.0, 320 of 324 cases escalate. That is reported as a finding,
not tuned away: lowering the thresholds would buy speed by making wrong answers
confident.

**(d) The tie-break is correct and never fires.** ``select_candidate`` applies
the mandated centre prior among candidates statistically indistinguishable from
the best, but ``TIE_SIGMA`` is 0.0 — exact ties only — and exact ties do not
occur on this dataset. ``n_tied`` is always 1, so it returns ``peaks[0]`` and
contributes exactly 0.000 to accuracy. Both ``n_tied`` and ``tie_break_used``
are consequently dead as confidence features. The tie-break stays because the
problem statement mandates it and because it is correct when ties exist, not
because it is currently earning anything.
"""

import numpy as np

from src import config
from src.types import FloatArray, Peak
from src.uniqueness import uniqueness_map, uniqueness_score

__all__ = [
    "peak_to_sidelobe",
    "uniqueness_score",
    "select_candidate",
    "sidelobe_stats",
    "tied_candidates",
    "uniqueness_map",
]


def sidelobe_stats(
    surface: FloatArray,
    peak: Peak,
    exclusion_radius: int = config.PSR_EXCLUSION_RADIUS_PX,
) -> tuple[float, float]:
    """Return ``(mean, std)`` of the surface outside the exclusion radius.

    Exposed separately because the tie tolerance is specified in sidelobe
    standard deviations (``config.TIE_SIGMA``) while :func:`select_candidate`
    takes a tolerance in score units and never receives the surface. The
    conversion happens at the call site; this is where the call site gets the
    number.

    Returns ``nan`` for either statistic when it is undefined, matching
    :func:`peak_to_sidelobe`'s absent-measurement convention.
    """
    rows, cols = np.ogrid[: surface.shape[0], : surface.shape[1]]
    chebyshev = np.maximum(np.abs(rows - peak.row), np.abs(cols - peak.col))
    sidelobe = surface[chebyshev > exclusion_radius]

    if sidelobe.size == 0:
        return (float("nan"), float("nan"))

    mean = float(np.nanmean(sidelobe))
    std = float(np.nanstd(sidelobe))

    if not np.isfinite(std) or std == 0.0:
        return (mean, float("nan"))

    return (mean, std)


def tied_candidates(peaks: list[Peak], tolerance: float) -> list[Peak]:
    """Return every candidate statistically indistinguishable from the best.

    A non-finite ``tolerance`` means the sidelobe statistics were unavailable,
    not that everything ties. It is coerced to zero — exact ties only — which
    keeps the returned list non-empty. Comparing against NaN would return
    ``False`` for every peak and hand :func:`select_candidate` an empty set.
    """
    if not peaks:
        return []
    if not np.isfinite(tolerance):
        tolerance = 0.0
    best_score = max(p.score for p in peaks)
    return [p for p in peaks if best_score - p.score <= tolerance]


def peak_to_sidelobe(
    surface: FloatArray,
    peak: Peak,
    exclusion_radius: int = config.PSR_EXCLUSION_RADIUS_PX,
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
    mean, std = sidelobe_stats(surface, peak, exclusion_radius)
    if not (np.isfinite(mean) and np.isfinite(std)):
        return float("nan")
    return float((float(surface[peak.row, peak.col]) - mean) / std)


def select_candidate(
    peaks: list[Peak],
    image_centre: tuple[float, float],
    template_shape: tuple[int, int],
    tolerance: float,
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
    template_shape
        Shape of the template as ``(rows, cols)``. Needed to convert each
        candidate's top-left corner to its centre before measuring distance.

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
    if not peaks:
        msg = "select_candidate requires at least one candidate peak"
        raise ValueError(msg)

    tied = tied_candidates(peaks, tolerance)

    if len(tied) == 1:
        return (tied[0], False)

    centre_x, centre_y = image_centre

    def _distance_key(p: Peak) -> tuple[float, int, int]:
        px, py = p.centre(template_shape)
        return ((px - centre_x) ** 2 + (py - centre_y) ** 2, p.row, p.col)

    nearest = min(tied, key=_distance_key)
    return (nearest, True)
