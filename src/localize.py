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
ladder and the peak-to-sidelobe statistic in place.

A per-call memo (:class:`_StageCache`) collapses escalation tiers that resolve
to the same pose and PSF width. It is a caching layer only: a miss computes
exactly what the uncached path computed, and the outputs were verified
bit-identical across every diagnostic field on the generated dataset. It took
the auto-mode median from 360 ms to 105 ms.

Masked correlation is the remaining gap. The integration point is
:meth:`_StageCache.correlate`, which is the single place a weight mask has to
enter both the correlation call and the cache key.
"""

import argparse
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

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

_MIN_POSE_QUALITY: float = 0.2
"""Quality below which R2's pose estimate is discarded in favour of nominal.

Pose error produces a position error that grows with distance from the image
centre, so acting on an untrustworthy estimate is worse than ignoring it:
assuming nominal is at least unbiased. Tuned against the pose-quality
distribution once R2's estimator produces real values.
"""


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

    __slots__ = ("_key", "_outcomes", "_psf_sigma", "_reference", "_result", "_search")

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
        self._key: tuple[float, float, float] | None = None
        self._result: tuple[FloatArray, FloatArray, list[Peak]] | None = None
        self._outcomes: dict[tuple[float, float, float], _TierOutcome] = {}

    def outcome(self, key: tuple[float, float, float]) -> _TierOutcome | None:
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

    def store(self, key: tuple[float, float, float], outcome: _TierOutcome) -> None:
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

    def correlate(
        self,
        theta: float,
        scale: float,
        psf_sigma: float,
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
        When the masked-correlation path lands, the weight mask becomes part of
        the key: it changes the surface, so it must invalidate the cache. The
        call to :func:`src.matcher.zncc_surface` already accepts ``weight`` and
        is the single place that needs to change.
        """
        key = (theta, scale, psf_sigma)
        if self._key != key or self._result is None:
            template = matcher.build_template(
                self._reference,
                theta=theta,
                scale=scale,
                psf_sigma_px=psf_sigma,
            )
            surface = matcher.zncc_surface(template, self._search)
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
    key = (pose_estimate.theta_deg, pose_estimate.scale, psf_sigma)
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

    sidelobe_mean, sidelobe_std = disambiguate.sidelobe_stats(
        surface,
        peaks[0],
        exclusion_radius=config.DEFAULT_NMS_RADIUS_PX,
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
            exclusion_radius=config.DEFAULT_NMS_RADIUS_PX,
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
            centre_x, centre_y = _run_pipeline(
                cache, search_f, pose_estimate, psf_sigma, diagnostics
            )
            if not _should_escalate(diagnostics, tier):
                break

        model = ConfidenceModel.load_or_default(None)
        confidence = float(np.clip(model.predict(diagnostics), 0.0, 1.0))

        result = LocalizationResult(
            x=float(centre_x),
            y=float(centre_y),
            confidence=confidence,
            low_confidence_flag=is_low_confidence(confidence, diagnostics),
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
