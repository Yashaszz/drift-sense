"""Shared data structures that cross module boundaries.

This module is the real interface contract between the four roles. The frozen
function signatures agreed in Phase 0 are only meaningful if everyone means the
same thing by ``Peak`` and ``PoseEstimate``; if each role defines its own, the
result is either circular imports or silent field mismatches discovered at
integration time.

Ownership
---------
``Peak``
    Produced by :func:`src.matcher.top_k_peaks` (R4), consumed by
    ``src.disambiguate.select_candidate`` (R3).
``PoseEstimate``
    Produced by ``src.pose.estimate_pose`` (R2), consumed by
    :func:`src.matcher.build_template` (R4).
``Diagnostics`` and ``LocalizationResult``
    Produced by :func:`src.localize.localize` (R4), consumed by
    ``src.evaluate`` (R3).

Changing a field here breaks other people's code. Raise it with the team first.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, TypeAlias

import numpy as np
import numpy.typing as npt

from src import config

# ---------------------------------------------------------------------------
# Array aliases
# ---------------------------------------------------------------------------

AnyArray: TypeAlias = npt.NDArray[Any]
"""Array of unconstrained dtype. Used for inputs we accept permissively."""

FloatArray: TypeAlias = npt.NDArray[np.float32]
"""Single-precision array. The internal working dtype throughout the pipeline."""

Float64Array: TypeAlias = npt.NDArray[np.float64]
"""Double-precision array. Used for feature vectors and fitted coefficients."""

BoolArray: TypeAlias = npt.NDArray[np.bool_]
"""Boolean mask array."""

Shape2D: TypeAlias = tuple[int, int]
"""Two-dimensional array shape as ``(rows, cols)``, matching NumPy."""

Mode: TypeAlias = Literal["auto", "fast", "robust", "ambiguous"]
"""Operating mode for :func:`src.localize.localize`.

``auto``
    Start in ``fast`` and escalate on a weak detection statistic. The default.
``fast``
    Assume nominal scale and zero rotation; single ZNCC pass.
``robust``
    Full pose estimation, multiple pose hypotheses, uniqueness weighting.
``ambiguous``
    Robust, plus re-ranking and the mandated centre tie-break.
"""

FailureMode: TypeAlias = Literal[
    "none",
    "lattice_aliasing",
    "snr_collapse",
    "pose_mis_estimate",
    "aliasing_destruction",
    "tie_break_loss",
    "edge_clipping",
    "internal_error",
]
"""Failure taxonomy from the handbook, used to classify rather than merely count.

Distinguishes *algorithmic* failures, which are fixable, from
*information-theoretic* ones where the data genuinely does not contain the
answer. ``internal_error`` marks a degraded result produced by an exception
inside the pipeline rather than by the imaging physics.
"""


# ---------------------------------------------------------------------------
# Candidate peaks
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Peak:
    """One candidate match position on a correlation surface.

    Notes
    -----
    ``col`` and ``row`` index the correlation surface, which addresses the
    *top-left corner* of the template window — not its centre. The fields are
    named ``col``/``row`` rather than ``x``/``y`` deliberately: ``x``/``y``
    invites the reader to assume a centre position in search-image coordinates,
    which would be wrong by roughly half a template edge.

    Use :meth:`centre` to obtain the search-image centre.

    Attributes
    ----------
    col
        Column index into the correlation surface.
    row
        Row index into the correlation surface.
    score
        Correlation score at that index, in ``[-1, 1]`` for a ZNCC surface.
    """

    col: int
    row: int
    score: float

    def centre(self, template_shape: Shape2D) -> tuple[float, float]:
        """Return the centre of this window in search-image pixels.

        Parameters
        ----------
        template_shape
            Shape of the template that produced the surface, as ``(rows, cols)``.

        Returns
        -------
        tuple of float
            Window centre as ``(x, y)``.
        """
        return config.window_topleft_to_centre(self.col, self.row, template_shape)


# ---------------------------------------------------------------------------
# Pose
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PoseEstimate:
    """Estimated rotation and scale relating the reference to the search image.

    Produced by R2's Fourier/log-polar stage. Note that ``scale`` is the total
    ratio, not the residual: a perfect nominal capture gives
    ``scale == config.NOMINAL_SCALE``, i.e. 10.0.

    Attributes
    ----------
    theta_deg
        Rotation in degrees, positive counter-clockwise.
    scale
        Reference-to-search decimation ratio, near ``config.NOMINAL_SCALE``.
    quality
        Confidence in this estimate, in ``[0, 1]``. Low quality means the
        spectral peaks were too weak to trust and the caller should fall back to
        a bounded grid search over pose.
    """

    theta_deg: float
    scale: float
    quality: float

    @classmethod
    def nominal(cls) -> "PoseEstimate":
        """Return the assumed-nominal pose: no rotation, exactly 10x.

        This is what ``fast`` mode uses, and what callers should fall back to
        when pose estimation is unavailable or untrustworthy.

        Returns
        -------
        PoseEstimate
            Zero rotation, nominal scale, zero quality.
        """
        return cls(theta_deg=0.0, scale=config.NOMINAL_SCALE, quality=0.0)


# ---------------------------------------------------------------------------
# Diagnostics and result
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Diagnostics:
    """Evidence emitted alongside the answer.

    Mutable by design: the fields are filled in progressively as the pipeline
    runs, so that a result degraded partway through still carries whatever
    evidence was gathered before the degradation.

    This structure is 10% of the project score on its own. A system that knows
    when it is guessing is worth more than one that claims certainty, and on a
    real tool ``low_confidence_flag`` is what triggers a wider search or a
    fallback to a different alignment site.

    Attributes
    ----------
    ncc_peak
        Absolute correlation quality of the winning peak.
    psr
        Peak-to-sidelobe ratio — the ambiguity measure.
    n_tied
        Number of candidates statistically tied with the best.
    theta_est
        Estimated rotation in degrees, checkable against the expected range.
    scale_est
        Estimated scale, checkable against ``config.NOMINAL_SCALE``.
    tie_break_used
        Whether the mandated centre rule decided the answer.
    uniqueness_score
        Mean uniqueness of the reference — is there an aperiodic anchor at all?
    mode_used
        Which operating mode actually ran.
    failure_mode
        Classification from the failure taxonomy.
    elapsed_ms
        Wall-clock time for the whole call, in milliseconds.
    notes
        Free-text breadcrumbs for failure analysis. Never parsed.
    """

    ncc_peak: float = 0.0
    psr: float = 0.0
    n_tied: int = 0
    theta_est: float = 0.0
    scale_est: float = config.NOMINAL_SCALE
    tie_break_used: bool = False
    uniqueness_score: float = 0.0
    mode_used: str = "fast"
    failure_mode: FailureMode = "none"
    elapsed_ms: float = 0.0
    notes: tuple[str, ...] = ()

    def with_note(self, note: str) -> None:
        """Append a diagnostic breadcrumb.

        Parameters
        ----------
        note
            Short human-readable message describing what happened.
        """
        self.notes = (*self.notes, note)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of these diagnostics.

        Returns
        -------
        dict
            Field names mapped to plain Python values, with ``notes`` as a list.
        """
        payload = asdict(self)
        payload["notes"] = list(self.notes)
        return payload


@dataclass(frozen=True, slots=True)
class LocalizationResult:
    """The answer returned by :func:`src.localize.localize`.

    Attributes
    ----------
    x
        Centre of the matching region, in search-image pixels (column axis).
    y
        Centre of the matching region, in search-image pixels (row axis).
    confidence
        Calibrated scalar in ``[0, 1]``. Validated on construction.
    low_confidence_flag
        ``True`` when the tool should escalate rather than trust this answer.
    diagnostics
        Supporting evidence. See :class:`Diagnostics`.

    Raises
    ------
    ValueError
        If ``confidence`` is outside ``[0, 1]`` or any coordinate is not finite.
        Callers are expected to clamp before constructing; the check exists to
        surface internal bugs in tests, not to be caught at runtime.
    """

    x: float
    y: float
    confidence: float
    low_confidence_flag: bool
    diagnostics: Diagnostics = field(default_factory=Diagnostics)

    def __post_init__(self) -> None:
        """Validate the invariants this type promises to its consumers."""
        if not 0.0 <= self.confidence <= 1.0:
            msg = f"confidence must lie in [0, 1], got {self.confidence!r}"
            raise ValueError(msg)
        if not (np.isfinite(self.x) and np.isfinite(self.y)):
            msg = f"coordinates must be finite, got ({self.x!r}, {self.y!r})"
            raise ValueError(msg)

    @property
    def centre(self) -> tuple[float, float]:
        """Return the answer as an ``(x, y)`` tuple.

        Returns
        -------
        tuple of float
            The matched centre in search-image pixels.
        """
        return (self.x, self.y)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of this result.

        Returns
        -------
        dict
            Flat mapping with nested diagnostics under ``"diagnostics"``.
        """
        return {
            "x": self.x,
            "y": self.y,
            "confidence": self.confidence,
            "low_confidence_flag": self.low_confidence_flag,
            "diagnostics": self.diagnostics.to_dict(),
        }
