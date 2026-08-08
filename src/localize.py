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
T0 skeleton. The public interface and the never-raises contract are final and
exercised by tests. The stage bodies call placeholder implementations in
:mod:`src.matcher`, so coordinates returned now are not meaningful answers —
they are shape- and type-correct ones. Wiring lands in T6.
"""

import argparse
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from src import config, matcher, pose
from src.confidence import ConfidenceModel, is_low_confidence
from src.types import (
    AnyArray,
    Diagnostics,
    FailureMode,
    FloatArray,
    LocalizationResult,
    Mode,
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


def _resolve_psf_sigma(search: FloatArray, mode: Mode) -> float:
    """Choose the PSF width to build the template at.

    Parameters
    ----------
    search
        Search image as ``float32``.
    mode
        Resolved operating mode; never ``"auto"``.

    Returns
    -------
    float
        Target PSF width in search pixels.

    Notes
    -----
    ``fast`` mode uses the documented default rather than measuring. Estimation
    costs roughly 15 ms even on a bounded window, and on a periodic layout —
    which is most of this problem — the estimator correctly declines and returns
    that same default anyway. Paying for it on the cheap path buys nothing.
    """
    if mode == "fast":
        return config.DEFAULT_PSF_SIGMA_PX
    return matcher.estimate_psf_sigma(search)


def _run_pipeline(
    search: FloatArray,
    reference: FloatArray,
    pose_estimate: PoseEstimate,
    psf_sigma: float,
    diagnostics: Diagnostics,
) -> tuple[float, float]:
    """Run Stages 2, 3, 3b and 5 for one pose hypothesis.

    Parameters
    ----------
    search
        Search image as ``float32``.
    reference
        Reference image as ``float32``.
    pose_estimate
        Rotation and scale hypothesis to build the template at.
    psf_sigma
        Target PSF width in search pixels, from :func:`_resolve_psf_sigma`.
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
    template = matcher.build_template(
        reference,
        theta=pose_estimate.theta_deg,
        scale=pose_estimate.scale,
        psf_sigma_px=psf_sigma,
    )
    surface = matcher.zncc_surface(template, search)
    peaks = matcher.top_k_peaks(
        surface,
        k=config.DEFAULT_TOP_K,
        nms_radius=config.DEFAULT_NMS_RADIUS_PX,
    )
    if not peaks:
        msg = "correlation surface has no distinguishable peak"
        raise _NoCandidatesError(msg)

    # Stage 4 is R3's. Until disambiguate.select_candidate exists, take the
    # strongest peak — which is exactly the mandated plain-NCC baseline
    # behaviour, so this stub is a useful comparison point rather than a
    # throwaway.
    best = peaks[0]
    diagnostics.ncc_peak = best.score
    diagnostics.n_tied = len(peaks)
    diagnostics.theta_est = pose_estimate.theta_deg
    diagnostics.scale_est = pose_estimate.scale

    centre_x, centre_y = best.centre(template.shape)

    refinement = matcher.refine_subpixel_detailed(
        template,
        search,
        best,
        surface=surface,
        upsample=config.DEFAULT_UPSAMPLE,
    )
    diagnostics.subpixel_error = refinement.error
    diagnostics.subpixel_method = refinement.method

    return (centre_x + refinement.dx, centre_y + refinement.dy)


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
        resolved: Mode = "fast" if mode == "auto" else mode
        diagnostics.mode_used = resolved

        pose_estimate = _resolve_pose(search_f, reference_f, resolved)
        psf_sigma = _resolve_psf_sigma(search_f, resolved)
        centre_x, centre_y = _run_pipeline(
            search_f, reference_f, pose_estimate, psf_sigma, diagnostics
        )

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
