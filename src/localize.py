"""The submitted deliverable: locate a reference image inside a search image.

Orchestrates every stage and returns the centre of the match in search-image
pixels, together with the evidence behind it.

The contract
------------
:func:`localize` **must never raise on valid input.** A wafer-inspection tool
cannot have its alignment step throw; it needs an answer plus an honest quality
signal, so that a weak answer triggers a wider search rather than a crash. Every
stage is therefore wrapped, and any internal failure degrades to the best
available answer with ``low_confidence_flag`` set.

The last-resort answer is the centre of the search image. That is not an evasion:
the stage aimed at the field-of-view centre, so with no other information the
centre is the maximum-prior guess, and it is the same prior the mandated
tie-break rule encodes.

Operating modes
---------------
Cheap by default, expensive on demand. Runtime is scored, and on a tool making
thousands of moves a day the *average* is what matters — so compute escalates
only when the detection statistic says the easy path was not enough.

Status
------
Stages 2, 3, 3b, 4 and 5 are wired and produce real answers, with the escalation
ladder, the peak-to-sidelobe statistic and the uniqueness-weighted correlation
path (T8) in place.

A per-call memo (:class:`_StageCache`) collapses escalation tiers that resolve
to the same pose, PSF width and weighting. It is a caching layer only: a miss
computes exactly what the uncached path computed, and the outputs are verified
bit-identical across every diagnostic field on the generated dataset.

Weighting is skipped while ``uniqueness_map`` returns a constant, because a
constant map makes weighted correlation identical to unweighted. That is an
equivalence rather than an approximation, and it reverses itself automatically:
the moment R3 lands a non-constant map the weighted path engages with no change
here.
"""

import argparse
import hashlib
import time
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TypeAlias

import numpy as np

from src import config, disambiguate, matcher, pose
from src.confidence import ConfidenceModel, is_low_confidence
from src.types import (
    AnyArray,
    Diagnostics,
    FailureMode,
    FloatArray,
    LocalizationResult,
    Mode,
    Peak,
    PoseEstimate,
)

__all__ = ["localize", "main"]

_CacheKey: TypeAlias = tuple[float, float, float, bool]
"""Everything that can change a tier's outcome: pose, PSF width, weighting."""

_CONSTANT_WEIGHT_ATOL: float = 1e-6
"""Range below which a uniqueness map counts as constant, and weighting is skipped."""

_MIN_POSE_QUALITY: float = 0.2
"""Quality below which R2's pose estimate is discarded in favour of nominal.

Pose error produces a position error that grows with distance from the image
centre, so acting on an untrustworthy estimate is worse than ignoring it:
assuming nominal is at least unbiased. Tuned against the pose-quality
distribution once R2's estimator produces real values.
"""


_UniquenessKey: TypeAlias = tuple[object, int, bytes]
"""Implementation, tile size, and reference content. See :func:`_uniqueness_for`."""

_UNIQUENESS_CACHE: "OrderedDict[_UniquenessKey, FloatArray]" = OrderedDict()
"""Uniqueness maps from previous calls, newest last. See :func:`_uniqueness_for`."""


def _reference_digest(reference: FloatArray) -> bytes:
    """Fingerprint a reference image by content.

    Identity is the wrong key: callers routinely pass a fresh array read from
    disk for the same physical site, and two equal arrays must hit the same
    entry. Content hashing costs about 3 ms on a 1000x1000 float32 reference
    against the 400 ms it guards, and ``shape`` is folded in so that two arrays
    sharing a buffer layout but not a shape cannot collide.
    """
    digest = hashlib.blake2b(digest_size=16)
    digest.update(repr(reference.shape).encode())
    digest.update(np.ascontiguousarray(reference).view(np.uint8).reshape(-1).tobytes())
    return digest.digest()


def _uniqueness_for(reference: FloatArray) -> FloatArray:
    """Return R3's uniqueness map for a reference, reusing a recent computation.

    Measured at production shapes, ``uniqueness_map`` is 411 ms of a 647 ms
    ``localize`` call — 68% of runtime, and the single largest cost in the
    pipeline. It scores the *reference* alone, so it is invariant to everything
    the escalation ladder varies and to the search image entirely.

    :class:`_StageCache` already memoises it within one call. This is the tier
    above: a bounded, process-wide cache so that repeat visits to the same site
    pay for it once. That is the shape of the real workload — a wafer tool
    revisits a site, and every ablation sweep runs one pair through several
    configurations — where the per-call cache is by construction cold.

    A distinct reference every call, as in a 108-pair benchmark sweep, gets no
    benefit and pays only the digest. That is the intended trade.

    The key carries the *implementation* alongside the content, because the
    reference does not determine the map on its own — the function that scored
    it does. ``benchmarks/verify_uniqueness_integration.py`` substitutes a
    stand-in map to measure what R3's stage is worth, and an ablation sweep does
    the same; keying on content alone would serve the real map to a run that
    asked for the stand-in and quietly report the two as identical. Tile size is
    in the key for the same reason.

    Returns the cached array itself rather than a copy. Callers treat the map as
    read-only; :func:`src.matcher.build_weight` does not mutate it.
    """
    implementation = disambiguate.uniqueness_map
    tile = config.DEFAULT_UNIQUENESS_TILE_PX

    if config.UNIQUENESS_CACHE_ENTRIES <= 0:
        return implementation(reference, tile=tile)

    key: _UniquenessKey = (implementation, tile, _reference_digest(reference))
    hit = _UNIQUENESS_CACHE.get(key)
    if hit is not None:
        _UNIQUENESS_CACHE.move_to_end(key)
        return hit

    computed = implementation(reference, tile=tile)
    _UNIQUENESS_CACHE[key] = computed
    while len(_UNIQUENESS_CACHE) > config.UNIQUENESS_CACHE_ENTRIES:
        _UNIQUENESS_CACHE.popitem(last=False)
    return computed


def clear_uniqueness_cache() -> None:
    """Drop every cached uniqueness map.

    Exposed for benchmarks, which must measure a cold computation, and for tests
    that assert on call counts.
    """
    _UNIQUENESS_CACHE.clear()


@dataclass(frozen=True, slots=True)
class _TierOutcome:
    """Everything one escalation tier produces, apart from which tier ran.

    A tier's result is a pure function of the image pair and the three
    parameters that vary between tiers, so two tiers resolving to the same
    parameters produce identical outcomes. Capturing the whole outcome — rather
    than only the correlation surface — lets a repeat tier skip the sidelobe
    statistics, the candidate selection and the sub-pixel refinement as well.
    """

    centre: tuple[float, float]
    n_tied: int
    tie_break_used: bool
    ncc_peak: float
    theta_est: float
    scale_est: float
    psr: float
    subpixel_error: float
    subpixel_method: str

    def apply(self, diagnostics: Diagnostics) -> tuple[float, float]:
        """Write this outcome into a diagnostics record.

        Parameters
        ----------
        diagnostics
            Record to populate. ``mode_used`` is left untouched, because it is
            the one field that legitimately differs between tiers sharing an
            outcome.

        Returns
        -------
        tuple of float
            The matched centre as ``(x, y)``.
        """
        diagnostics.n_tied = self.n_tied
        diagnostics.tie_break_used = self.tie_break_used
        diagnostics.ncc_peak = self.ncc_peak
        diagnostics.theta_est = self.theta_est
        diagnostics.scale_est = self.scale_est
        diagnostics.psr = self.psr
        diagnostics.subpixel_error = self.subpixel_error
        diagnostics.subpixel_method = self.subpixel_method
        return self.centre


@lru_cache(maxsize=1)
def _confidence_model() -> ConfidenceModel:
    """Return the fitted Stage 6 calibrator, or an unfitted fallback.

    Returns
    -------
    ConfidenceModel
        A fitted model when ``config.CONFIDENCE_MODEL_PATH`` exists and parses,
        otherwise an unfitted one that scores by heuristic.

    Notes
    -----
    Cached because it is read from disk and never changes within a process. A
    missing file is the normal case, not an error: the deliverable must run
    from a clean unzip with no trained artefacts.
    """
    candidate = Path(config.CONFIDENCE_MODEL_PATH)
    return ConfidenceModel.load_or_default(candidate if candidate.is_file() else None)


class _StageCache:
    """Memoises the deterministic part of the pipeline across escalation tiers.

    The escalation ladder retries the same pair at progressively more expensive
    settings, but only two things vary between tiers: the pose hypothesis and
    the PSF width. Everything downstream of those — the template, the
    correlation surface, the candidate shortlist — is a pure function of
    ``(reference, search, theta, scale, psf_sigma)``. When a tier resolves to
    parameters already used, recomputing produces bit-identical results.

    Measured on the twelve generated pairs, the three tiers spent 67 ms in
    template construction and 69 ms in correlation *per pair*, of which two
    thirds was recomputation of unchanged values.

    Only the most recent key is retained. Tiers run in order and repeats are
    consecutive, so a one-entry cache captures every hit without unbounded
    growth on a 1000x1000 surface.

    This is a caching layer, not an algorithmic change: a miss computes exactly
    what the uncached path computed.
    """

    __slots__ = (
        "_key",
        "_outcomes",
        "_psf_sigma",
        "_reference",
        "_result",
        "_search",
        "_uniqueness",
        "_weights",
    )

    def __init__(self, search: FloatArray, reference: FloatArray) -> None:
        """Bind the cache to one image pair.

        Parameters
        ----------
        search
            Search image as ``float32``.
        reference
            Reference image as ``float32``.
        """
        self._search = search
        self._reference = reference
        self._psf_sigma: float | None = None
        self._key: _CacheKey | None = None
        self._result: tuple[FloatArray, FloatArray, list[Peak]] | None = None
        self._outcomes: dict[_CacheKey, _TierOutcome] = {}
        self._uniqueness: FloatArray | None = None
        self._weights: dict[tuple[float, float], FloatArray] = {}

    def outcome(self, key: _CacheKey) -> _TierOutcome | None:
        """Return a previously computed tier outcome, if one matches.

        Parameters
        ----------
        key
            ``(theta, scale, psf_sigma)`` for the tier.

        Returns
        -------
        _TierOutcome or None
            The stored outcome, or ``None`` on a miss.
        """
        return self._outcomes.get(key)

    def store(self, key: _CacheKey, outcome: _TierOutcome) -> None:
        """Record a tier outcome against its parameters.

        Parameters
        ----------
        key
            ``(theta, scale, psf_sigma)`` for the tier.
        outcome
            What that tier produced.
        """
        self._outcomes[key] = outcome

    def psf_sigma(self, tier: Mode) -> float:
        """Return the PSF width for a tier, estimating at most once per call.

        Parameters
        ----------
        tier
            Resolved operating mode.

        Returns
        -------
        float
            Target PSF width in search pixels.

        Notes
        -----
        ``fast`` uses the documented default rather than measuring: estimation
        costs around 20 ms, and on a periodic layout the estimator correctly
        declines and returns that same default anyway.

        The estimate depends only on the search image, which does not change
        between tiers, so it is computed once and reused.
        """
        if tier == "fast":
            return config.DEFAULT_PSF_SIGMA_PX
        if self._psf_sigma is None:
            self._psf_sigma = matcher.estimate_psf_sigma(self._search)
        return self._psf_sigma

    def uniqueness(self) -> FloatArray:
        """Return the reference-resolution uniqueness map, computed once.

        Returns
        -------
        FloatArray
            R3's weight map over the reference, same shape as the reference.

        Notes
        -----
        Scoring the reference depends on nothing that varies between tiers, so
        it is memoised alongside the PSF estimate. The process-wide cache behind
        :func:`_uniqueness_for` extends the same reuse across calls; this
        attribute stays because it avoids re-hashing the reference on every
        tier.
        """
        if self._uniqueness is None:
            self._uniqueness = _uniqueness_for(self._reference)
        return self._uniqueness

    def uniqueness_is_informative(self) -> bool:
        """Whether the uniqueness map actually distinguishes anything.

        Returns
        -------
        bool
            ``True`` when the map varies; ``False`` when it is constant.

        Notes
        -----
        A constant weight map is provably a no-op: normalising by the weight
        sum makes weighted correlation reduce exactly to the unweighted result,
        which is the contract asserted in the weighted-path tests. Running the
        weighted formulation anyway costs three cross-correlations instead of
        one - measured at 69 ms against 34 ms - for a surface that differs only
        at the 1e-6 level of float32 rounding.

        Skipping it is therefore an equivalence, not an approximation. It also
        means the weighting starts costing something exactly when it starts
        being worth something: the moment R3's map stops being constant, this
        returns ``True`` and the weighted path engages with no change here.
        """
        weights = self.uniqueness()
        return bool(float(weights.max()) - float(weights.min()) > _CONSTANT_WEIGHT_ATOL)

    def template_weight(self, theta: float, scale: float) -> FloatArray:
        """Return the uniqueness map carried onto the template grid.

        Parameters
        ----------
        theta
            Rotation in degrees, matching the template.
        scale
            Decimation ratio, matching the template.

        Returns
        -------
        FloatArray
            Weights on the template grid.
        """
        cached = self._weights.get((theta, scale))
        if cached is None:
            cached = matcher.build_weight(self.uniqueness(), theta, scale)
            self._weights[(theta, scale)] = cached
        return cached

    def correlate(
        self,
        theta: float,
        scale: float,
        psf_sigma: float,
        weighted: bool = False,
    ) -> tuple[FloatArray, FloatArray, list[Peak]]:
        """Build the template, correlate, and extract candidates.

        Parameters
        ----------
        theta
            Rotation in degrees.
        scale
            Decimation ratio.
        psf_sigma
            Target PSF width in search pixels.

        Returns
        -------
        tuple
            ``(template, surface, peaks)``. The arrays are shared with previous
            callers on a cache hit and must be treated as read-only.

        Notes
        -----
        ``weighted`` is part of the key because it changes the surface. The
        weight map itself is not: it is a deterministic function of the
        reference and of ``(theta, scale)``, all of which the key already
        covers.
        """
        key = (theta, scale, psf_sigma, weighted)
        if self._key != key or self._result is None:
            template = matcher.build_template(
                self._reference,
                theta=theta,
                scale=scale,
                psf_sigma_px=psf_sigma,
            )
            weight = self.template_weight(theta, scale) if weighted else None
            surface = matcher.zncc_surface(template, self._search, weight=weight)
            peaks = matcher.top_k_peaks(
                surface,
                k=config.DEFAULT_TOP_K,
                nms_radius=config.DEFAULT_NMS_RADIUS_PX,
            )
            self._key = key
            self._result = (template, surface, peaks)
        return self._result


def _psr_from_stats(
    surface: FloatArray,
    peak: Peak,
    sidelobe_mean: float,
    sidelobe_std: float,
) -> float:
    """Compute the peak-to-sidelobe ratio from already-measured statistics.

    Parameters
    ----------
    surface
        Correlation surface.
    peak
        The candidate being assessed.
    sidelobe_mean
        Mean of the sidelobe region around ``peak``.
    sidelobe_std
        Standard deviation of that region.

    Returns
    -------
    float
        The ratio, or NaN when either statistic is undefined.

    Notes
    -----
    Mirrors the final arithmetic of ``disambiguate.peak_to_sidelobe`` so the
    sidelobe does not have to be scanned twice. That function measures the
    statistics and then divides; the pipeline has already measured them for the
    tie tolerance, and re-measuring cost 24 ms per tier — 9.5% of total runtime
    — for a value already in hand.

    Duplicating the arithmetic risks drifting from R3's definition, so
    ``test_psr_from_stats_matches_r3`` asserts the two agree.
    """
    if not (np.isfinite(sidelobe_mean) and np.isfinite(sidelobe_std)):
        return float("nan")
    return float((float(surface[peak.row, peak.col]) - sidelobe_mean) / sidelobe_std)


class _NoCandidatesError(RuntimeError):
    """Peak extraction produced no candidates.

    Separated from ordinary failures because it is not a defect. A correlation
    surface with no distinguishable peak means the evidence genuinely does not
    single out a position — an SNR collapse, or a template carrying no signal.
    Reporting it as an internal error would hide a real, classifiable outcome
    behind a generic one and corrupt the failure taxonomy the writeup rests on.
    """


# ===========================================================================
# Input handling
# ===========================================================================


def _validate_inputs(search: AnyArray, reference: AnyArray) -> tuple[FloatArray, FloatArray]:
    """Coerce the two inputs to contiguous ``float32`` working arrays.

    Parameters
    ----------
    search
        Wide search image, expected two-dimensional.
    reference
        Zoomed-in reference image, expected two-dimensional.

    Returns
    -------
    tuple of FloatArray
        ``(search, reference)`` as contiguous ``float32``.

    Raises
    ------
    ValueError
        If either input is not two-dimensional or is empty. Callers inside
        :func:`localize` catch this and degrade; it is raised rather than
        silently repaired so that genuinely malformed input is visible in the
        diagnostics instead of producing a confident wrong answer.
    """
    arrays: list[FloatArray] = []
    for name, array in (("search", search), ("reference", reference)):
        as_array = np.asarray(array)
        if as_array.ndim != 2:
            msg = f"{name} must be two-dimensional, got shape {as_array.shape}"
            raise ValueError(msg)
        if as_array.size == 0:
            msg = f"{name} must be non-empty, got shape {as_array.shape}"
            raise ValueError(msg)
        arrays.append(np.ascontiguousarray(as_array, dtype=np.float32))
    return arrays[0], arrays[1]


def _fallback_result(
    search: AnyArray,
    diagnostics: Diagnostics,
    reason: str,
    failure_mode: FailureMode = "internal_error",
) -> LocalizationResult:
    """Build the degraded answer used when localization cannot complete.

    Parameters
    ----------
    search
        The search image as supplied, used only for its shape.
    diagnostics
        Evidence gathered before the failure. Annotated in place.
    reason
        Short description of what went wrong, recorded in the diagnostics.
    failure_mode
        Classification from the failure taxonomy. Defaults to ``internal_error``;
        callers that can name the cause more precisely should do so, because the
        taxonomy is what the failure analysis is written from.

    Returns
    -------
    LocalizationResult
        Centre-of-image answer, zero confidence, flagged.
    """
    diagnostics.failure_mode = failure_mode
    diagnostics.with_note(reason)

    shape = getattr(search, "shape", None)
    if isinstance(shape, tuple) and len(shape) == 2 and all(int(n) > 0 for n in shape):
        centre_x, centre_y = config.image_centre((int(shape[0]), int(shape[1])))
    else:
        centre_x, centre_y = 0.0, 0.0

    return LocalizationResult(
        x=centre_x,
        y=centre_y,
        confidence=0.0,
        low_confidence_flag=True,
        diagnostics=diagnostics,
    )


# ===========================================================================
# Mode implementations
# ===========================================================================


def _escalation_path(requested: Mode) -> tuple[Mode, ...]:
    """Return the tiers to attempt, in order, for a requested mode.

    Parameters
    ----------
    requested
        The mode the caller asked for.

    Returns
    -------
    tuple of Mode
        A single tier for an explicit request; the full ladder for ``"auto"``.

    Notes
    -----
    Cheap by default, expensive on demand. A tool making thousands of moves a
    day is judged on *average* time, so the fast path runs first and compute is
    escalated only when the detection statistic says it was not enough.
    """
    if requested == "auto":
        return ("fast", "robust", "ambiguous")
    return (requested,)


def _should_escalate(diagnostics: Diagnostics, tier: Mode) -> bool:
    """Decide whether the current tier's answer is too weak to accept.

    Parameters
    ----------
    diagnostics
        Evidence from the attempt that just finished.
    tier
        The tier that produced it.

    Returns
    -------
    bool
        ``True`` when a more expensive tier should be tried.

    Notes
    -----
    A non-finite peak-to-sidelobe ratio means *unknown*, not *good*. R3's
    implementation returns NaN when the sidelobe region is empty or has zero
    variance, and a naive ``psr < threshold`` comparison silently answers False
    for NaN — accepting an answer precisely when there is no evidence for it.
    Unknown escalates.

    The thresholds themselves are provisional. Measured on the first twelve
    generated pairs, PSR ranged 1.77-3.08 on correct answers and 1.44-3.41 on
    wrong ones: it does not yet separate the two, and the highest value in the
    set belonged to a wrong answer. That is an honest reading rather than a
    defect — an unweighted correlation surface over a periodic lattice really
    does have no dominant peak, which is what Stage 4a uniqueness weighting
    exists to fix. Until that lands, and until the dataset carries physics,
    every case escalates: slower, but never falsely confident.
    """
    if tier == "ambiguous":
        return False
    if not np.isfinite(diagnostics.psr):
        return True
    threshold = config.PSR_ACCEPT_THRESHOLD if tier == "fast" else config.PSR_AMBIGUOUS_THRESHOLD
    return bool(diagnostics.psr < threshold)


def _run_pipeline(
    cache: _StageCache,
    search: FloatArray,
    pose_estimate: PoseEstimate,
    psf_sigma: float,
    weighted: bool,
    diagnostics: Diagnostics,
) -> tuple[float, float]:
    """Run Stages 2, 3, 3b, 4 and 5 for one pose hypothesis.

    Parameters
    ----------
    cache
        Per-call memo for the template, surface and shortlist.
    search
        Search image as ``float32``, used for the sub-pixel crop and the
        image-centre tie-break.
    pose_estimate
        Rotation and scale hypothesis to build the template at.
    psf_sigma
        Target PSF width in search pixels.
    weighted
        Whether to weight the correlation by the reference's uniqueness map.
    diagnostics
        Evidence record, updated in place.

    Returns
    -------
    tuple of float
        Matched centre as ``(x, y)`` in search-image pixels.

    Raises
    ------
    _NoCandidatesError
        If peak extraction finds nothing, which means the correlation surface
        carries no distinguishable maximum.
    ValueError
        If the geometry is impossible, such as a template larger than the
        search image.
    """
    # A constant uniqueness map makes weighted correlation identical to
    # unweighted, so paying for it would buy nothing. Resolving the flag here
    # keeps the cache key honest: the two paths share an entry precisely when
    # they produce the same surface.
    effective_weighting = weighted and cache.uniqueness_is_informative()

    if weighted:
        # Set before the cache lookup, and outside the cached outcome, because
        # this is evidence about the *reference* rather than about the tier: it
        # answers "is there an anchor at all?". Two tiers can legitimately share
        # a correlation surface while only the later one has scored the
        # reference, so folding it into the outcome would report zero for a map
        # that was in fact computed.
        diagnostics.uniqueness_score = float(np.mean(cache.uniqueness(), dtype=np.float64))

    key = (pose_estimate.theta_deg, pose_estimate.scale, psf_sigma, effective_weighting)
    cached = cache.outcome(key)
    if cached is not None:
        # An earlier tier already resolved to these exact parameters. The whole
        # pipeline below is deterministic in them, so recomputing would produce
        # the same numbers at full cost.
        return cached.apply(diagnostics)

    template, surface, peaks = cache.correlate(*key)
    if not peaks:
        msg = "correlation surface has no distinguishable peak"
        raise _NoCandidatesError(msg)

    # PSR_EXCLUSION_RADIUS_PX, not DEFAULT_NMS_RADIUS_PX. The two are equal at 8
    # today, so passing the wrong one is currently invisible -- but they measure
    # different things (correlation main-lobe width versus lattice pitch) and
    # config.py splits them precisely so they can diverge. Passing the NMS
    # constant here re-coupled them at the call site, which meant any change to
    # the NMS radius would silently move every PSR value and every escalation
    # decision with it.
    sidelobe_mean, sidelobe_std = disambiguate.sidelobe_stats(
        surface,
        peaks[0],
        exclusion_radius=config.PSR_EXCLUSION_RADIUS_PX,
    )

    # TIE_SIGMA is a width in sidelobe standard deviations; select_candidate
    # takes score units and never sees the surface, so convert here.
    tolerance = config.TIE_SIGMA * sidelobe_std

    tied = disambiguate.tied_candidates(peaks, tolerance)
    best, tie_break_used = disambiguate.select_candidate(
        peaks,
        config.image_centre(search.shape),
        template.shape,
        tolerance=tolerance,
    )
    diagnostics.n_tied = len(tied)
    diagnostics.tie_break_used = tie_break_used
    diagnostics.ncc_peak = best.score
    diagnostics.theta_est = pose_estimate.theta_deg
    diagnostics.scale_est = pose_estimate.scale
    # The sidelobe statistics around peaks[0] are already in hand. When the
    # tie-break did not move the selection - the common case - the excluded
    # region is identical, so the ratio follows directly. Re-scanning cost 24 ms
    # per tier, 9.5% of total runtime, for a value already computed.
    if best is peaks[0]:
        diagnostics.psr = _psr_from_stats(surface, best, sidelobe_mean, sidelobe_std)
    else:
        diagnostics.psr = disambiguate.peak_to_sidelobe(
            surface,
            best,
            exclusion_radius=config.PSR_EXCLUSION_RADIUS_PX,
        )

    centre_x, centre_y = best.centre(template.shape)

    refinement = matcher.refine_subpixel_detailed(
        template,
        search,
        best,
        surface=surface,
        upsample=config.DEFAULT_UPSAMPLE,
    )
    outcome = _TierOutcome(
        centre=(centre_x + refinement.dx, centre_y + refinement.dy),
        n_tied=len(tied),
        tie_break_used=tie_break_used,
        ncc_peak=best.score,
        theta_est=pose_estimate.theta_deg,
        scale_est=pose_estimate.scale,
        psr=diagnostics.psr,
        subpixel_error=refinement.error,
        subpixel_method=refinement.method,
    )
    cache.store(key, outcome)
    return outcome.apply(diagnostics)


def _resolve_pose(search: FloatArray, reference: FloatArray, mode: Mode) -> PoseEstimate:
    """Choose the pose hypothesis appropriate to the requested mode.

    Parameters
    ----------
    search
        Search image as ``float32``.
    reference
        Reference image as ``float32``.
    mode
        Resolved operating mode; never ``"auto"``.

    Returns
    -------
    PoseEstimate
        Nominal pose in ``fast`` mode; R2's estimate otherwise, falling back to
        nominal when that estimate is not trustworthy.

    Notes
    -----
    This is the seam with R2. The call is already wired, so when
    :func:`src.pose.estimate_pose` gains a real body the pipeline picks it up
    with no change here — the stub returns a nominal pose with zero quality,
    which this function treats exactly as it would treat a genuine low-quality
    estimate.

    ``fast`` mode skips pose entirely and assumes nominal. That is the whole
    source of its speed advantage, and it is why escalation exists: when the
    detection statistic says the nominal assumption was wrong, the caller
    re-runs in ``robust`` and pays for pose estimation only then.

    A low-quality estimate is discarded rather than used. Pose error produces a
    position error that *grows with distance from the image centre*, so a bad
    estimate is worse than no estimate: assuming nominal is at least unbiased.
    """
    if mode == "fast":
        return PoseEstimate.nominal()

    estimate = pose.estimate_pose(reference, search, nominal_scale=config.NOMINAL_SCALE)
    if estimate.quality < _MIN_POSE_QUALITY:
        return PoseEstimate.nominal()
    return estimate


# ===========================================================================
# Public entry point
# ===========================================================================


def localize(
    search: AnyArray,
    reference: AnyArray,
    mode: Mode = "auto",
) -> LocalizationResult:
    """Locate the reference image inside the search image.

    Parameters
    ----------
    search
        Wide search image, nominally 1000x1000 at 10 nm/px. Any two-dimensional
        real array is accepted; it is converted to ``float32`` internally.
    reference
        Zoomed-in reference image, nominally 1000x1000 at 1 nm/px.
    mode
        Operating mode. ``"auto"`` starts cheap and escalates on a weak
        detection statistic. See :data:`src.types.Mode`.

    Returns
    -------
    LocalizationResult
        Centre of the match in search-image pixels, a calibrated confidence, the
        escalation flag, and the supporting diagnostics.

    Notes
    -----
    **This function never raises on valid input.** Malformed input, an internal
    failure, or a stage that cannot produce a candidate all degrade to a flagged
    centre-of-image answer with zero confidence. Callers should branch on
    ``low_confidence_flag``, not on exceptions.

    Examples
    --------
    >>> import numpy as np
    >>> result = localize(np.zeros((200, 200)), np.zeros((100, 100)))
    >>> 0.0 <= result.confidence <= 1.0
    True
    """
    started = time.perf_counter()
    diagnostics = Diagnostics(mode_used=mode)

    try:
        search_f, reference_f = _validate_inputs(search, reference)

        cache = _StageCache(search_f, reference_f)

        centre_x = centre_y = 0.0
        for tier in _escalation_path(mode):
            diagnostics = Diagnostics(mode_used=tier)
            pose_estimate = _resolve_pose(search_f, reference_f, tier)
            psf_sigma = cache.psf_sigma(tier)
            # Uniqueness weighting is the payload of escalation, not the default:
            # it costs three correlations instead of one, and it only helps when
            # the cheap path has already reported weak evidence.
            centre_x, centre_y = _run_pipeline(
                cache, search_f, pose_estimate, psf_sigma, tier != "fast", diagnostics
            )
            if not _should_escalate(diagnostics, tier):
                break

        model = _confidence_model()
        confidence = float(np.clip(model.predict(diagnostics), 0.0, 1.0))

        result = LocalizationResult(
            x=float(centre_x),
            y=float(centre_y),
            confidence=confidence,
            low_confidence_flag=is_low_confidence(confidence, diagnostics, model.threshold),
            diagnostics=diagnostics,
        )
    except _NoCandidatesError as exc:
        result = _fallback_result(search, diagnostics, str(exc), failure_mode="snr_collapse")
    except Exception as exc:  # noqa: BLE001 - the never-raises contract is the point
        result = _fallback_result(search, diagnostics, f"{type(exc).__name__}: {exc}")

    result.diagnostics.elapsed_ms = (time.perf_counter() - started) * 1000.0
    return result


# ===========================================================================
# Command-line interface
# ===========================================================================


def _load_image(path: Path) -> AnyArray:
    """Read a grayscale image from disk.

    Parameters
    ----------
    path
        Image file. ``.npy`` is loaded with NumPy; anything else is opened with
        Pillow and converted to 8-bit grayscale.

    Returns
    -------
    AnyArray
        Two-dimensional array.
    """
    if path.suffix.lower() == ".npy":
        loaded: AnyArray = np.load(path)
        return loaded

    from PIL import Image  # imported lazily: only the CLI needs Pillow

    with Image.open(path) as handle:
        return np.asarray(handle.convert("L"))


def main(argv: Sequence[str] | None = None) -> int:
    """Run localization on one image pair from the command line.

    Parameters
    ----------
    argv
        Argument list; defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit status: ``0`` on success, ``1`` if the inputs could not be
        read. A low-confidence answer is still a success — the flag carries that
        information, not the exit code.
    """
    parser = argparse.ArgumentParser(
        prog="drift-localize",
        description="Locate a reference SEM image inside a wider search image.",
    )
    parser.add_argument("search", type=Path, help="path to the search image")
    parser.add_argument("reference", type=Path, help="path to the reference image")
    parser.add_argument(
        "--mode",
        default="auto",
        choices=("auto", "fast", "robust", "ambiguous"),
        help="operating mode (default: auto)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the full result as JSON instead of plain coordinates",
    )
    args = parser.parse_args(argv)

    try:
        search = _load_image(args.search)
        reference = _load_image(args.reference)
    except (OSError, ValueError) as exc:
        parser.exit(status=1, message=f"error: could not read inputs: {exc}\n")

    result = localize(search, reference, mode=args.mode)

    if args.json:
        import json

        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"{result.x:.4f} {result.y:.4f} {result.confidence:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
