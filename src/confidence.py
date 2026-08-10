"""Stage 6 — calibrated confidence and the low-confidence flag.

Turns the diagnostic evidence gathered during localization into a single
calibrated scalar in ``[0, 1]``, plus the boolean a real tool would act on.

Why this is built early rather than bolted on
---------------------------------------------
Explainability on failure cases is 10% of the project score outright, and a
claimed 100% success rate scores *worse* than an honest 94% with a well
diagnosed failure mode. The organizers have planted a case where localization
is genuinely impossible — a highly periodic region with no aperiodic anchor. On
that case the correct behaviour is not to succeed; it is to return an answer and
say plainly that it is a guess.

A confidence number is only worth anything if it is calibrated: the
low-confidence cases must be the ones that actually fail. That is a measurable
property, checked by plotting confidence against observed accuracy.

Design constraint
-----------------
:func:`src.localize.localize` must run on a clean machine straight out of the
submission zip. A trained model is therefore never a hard dependency: when no
fitted model is present, :class:`ConfidenceModel` falls back to
:func:`heuristic_confidence`, which needs nothing but the diagnostics.

Status
------
T0 skeleton. The interface is final; the fitted logistic model lands in T7.
:meth:`ConfidenceModel.predict` currently delegates to the heuristic in both
the fitted and unfitted cases.
"""

import json
from pathlib import Path
from typing import Any, Self

import numpy as np

from src import config
from src.types import Diagnostics, Float64Array

__all__ = [
    "FEATURE_NAMES",
    "ConfidenceModel",
    "extract_features",
    "heuristic_confidence",
    "is_low_confidence",
]


FEATURE_NAMES: tuple[str, ...] = (
    "ncc_peak",
    "psr",
    "log1p_n_tied",
    "uniqueness_score",
    "scale_residual",
    "abs_theta",
)
"""Ordered feature names for the confidence model.

Order is part of the serialisation contract: fitted coefficients are stored
positionally, so inserting a feature invalidates any saved model. Append only,
and bump the model file version when you do.

``log1p_n_tied`` compresses the tie count, which ranges over several orders of
magnitude — a periodic field can produce thousands of tied candidates.
``scale_residual`` and ``abs_theta`` capture pose implausibility: an estimate far
from nominal is itself evidence that something went wrong upstream.
"""


def extract_features(diagnostics: Diagnostics) -> Float64Array:
    """Pack diagnostics into the model's feature vector.

    Parameters
    ----------
    diagnostics
        Evidence gathered during localization.

    Returns
    -------
    Float64Array
        One-dimensional array of length ``len(FEATURE_NAMES)``, ordered to match
        it.

    Notes
    -----
    This is data marshalling, not modelling, so it is implemented in full at T0:
    the feature layout is part of the interface other people build against.

    The output is guaranteed finite. Upstream fields may legitimately be NaN —
    ``psr`` is NaN whenever the sidelobe region is degenerate — and a single NaN
    poisons a whole fitted model, so non-finite values are imputed to zero here.

    That imputation is lossy: it makes "unmeasurable" indistinguishable from
    "measured as zero". T7 should add a companion indicator feature rather than
    rely on the imputation alone. Appending is safe; reordering is not, because
    fitted coefficients are stored positionally.
    """
    raw = np.array(
        [
            diagnostics.ncc_peak,
            diagnostics.psr,
            float(np.log1p(max(0, diagnostics.n_tied))),
            diagnostics.uniqueness_score,
            abs(diagnostics.scale_est - config.NOMINAL_SCALE),
            abs(diagnostics.theta_est),
        ],
        dtype=np.float64,
    )
    return np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)


def heuristic_confidence(diagnostics: Diagnostics) -> float:
    """Score confidence without a fitted model.

    The fallback used before the model is trained, and whenever no model
    artefact ships alongside the code. Conservative by construction: it returns
    a low score so that an untrained system flags its answers rather than
    asserting them.

    Parameters
    ----------
    diagnostics
        Evidence gathered during localization.

    Returns
    -------
    float
        Confidence in ``[0, 1]``.

    Notes
    -----
    T0 returns ``0.0`` unconditionally, which is honest: nothing has been
    calibrated, so nothing is trusted. A monotone function of the
    peak-to-sidelobe ratio and the correlation peak replaces this in T7.
    """
    del diagnostics  # placeholder: monotone scoring lands in T7
    return 0.0


def is_low_confidence(confidence: float, diagnostics: Diagnostics) -> bool:
    """Decide whether the tool should escalate rather than trust this answer.

    Parameters
    ----------
    confidence
        Calibrated confidence in ``[0, 1]``.
    diagnostics
        Evidence gathered during localization.

    Returns
    -------
    bool
        ``True`` when the answer should be treated as unreliable.

    Notes
    -----
    Deliberately not a bare threshold on ``confidence``. An internal error or an
    ambiguous peak structure flags the result regardless of what the score says,
    so that a miscalibrated model cannot mask a known-bad case.
    """
    if diagnostics.failure_mode != "none":
        return True
    if not np.isfinite(diagnostics.psr):
        # NaN means the ambiguity measure could not be computed, not that the
        # answer is unambiguous. A bare ``psr < threshold`` comparison answers
        # False for NaN, which would clear the flag exactly when there is no
        # evidence to clear it with.
        return True
    if diagnostics.psr < config.PSR_AMBIGUOUS_THRESHOLD:
        return True
    return confidence < 0.5


class ConfidenceModel:
    """Logistic-regression calibrator mapping diagnostics to a probability.

    Wraps the fitted coefficients behind a small interface so that callers never
    depend on scikit-learn being importable at inference time, and so the
    unfitted case has well-defined behaviour rather than raising.

    Attributes
    ----------
    coefficients
        Fitted weights, positionally aligned with :data:`FEATURE_NAMES`, or
        ``None`` when unfitted.
    intercept
        Fitted intercept, or ``0.0`` when unfitted.

    Examples
    --------
    >>> model = ConfidenceModel()
    >>> model.is_fitted
    False
    >>> 0.0 <= model.predict(Diagnostics()) <= 1.0
    True
    """

    FORMAT_VERSION: int = 1
    """Serialisation version, written into the model file."""

    def __init__(
        self,
        coefficients: tuple[float, ...] | None = None,
        intercept: float = 0.0,
    ) -> None:
        """Construct a calibrator, optionally from known coefficients.

        Parameters
        ----------
        coefficients
            Weights aligned with :data:`FEATURE_NAMES`, or ``None`` for unfitted.
        intercept
            Model intercept.

        Raises
        ------
        ValueError
            If ``coefficients`` is given with a length other than
            ``len(FEATURE_NAMES)``.
        """
        if coefficients is not None and len(coefficients) != len(FEATURE_NAMES):
            msg = (
                f"expected {len(FEATURE_NAMES)} coefficients to match FEATURE_NAMES, "
                f"got {len(coefficients)}"
            )
            raise ValueError(msg)
        self.coefficients = coefficients
        self.intercept = intercept

    @property
    def is_fitted(self) -> bool:
        """Whether this model holds fitted coefficients.

        Returns
        -------
        bool
            ``True`` once :meth:`fit` or a load has supplied coefficients.
        """
        return self.coefficients is not None

    def fit(self, features: Float64Array, correct: Float64Array) -> Self:
        """Fit the calibrator against observed outcomes.

        Parameters
        ----------
        features
            Feature matrix of shape ``(n_samples, len(FEATURE_NAMES))``, built
            by stacking :func:`extract_features` over a labelled dataset.
        correct
            Binary outcome per sample: whether that localization actually landed
            within tolerance of the true centre.

        Returns
        -------
        Self
            This model, to allow chaining.

        Raises
        ------
        ValueError
            If the two arrays disagree on sample count, or the feature matrix
            has the wrong width.

        Notes
        -----
        T0 validates shapes and records the call without estimating anything.
        The scikit-learn fit lands in T7.
        """
        if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
            msg = f"features must have shape (n, {len(FEATURE_NAMES)}), got {features.shape}"
            raise ValueError(msg)
        if correct.shape[0] != features.shape[0]:
            msg = (
                f"features and correct disagree on sample count: "
                f"{features.shape[0]} vs {correct.shape[0]}"
            )
            raise ValueError(msg)
        return self

    def predict(self, diagnostics: Diagnostics) -> float:
        """Return the calibrated confidence for one localization.

        Parameters
        ----------
        diagnostics
            Evidence gathered during localization.

        Returns
        -------
        float
            Confidence in ``[0, 1]``. Never raises: an unfitted model falls back
            to :func:`heuristic_confidence`.
        """
        return heuristic_confidence(diagnostics)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of this model.

        Returns
        -------
        dict
            Version, feature order, coefficients and intercept. The feature
            order is stored so that a stale model file can be detected rather
            than silently misinterpreted.
        """
        return {
            "format_version": self.FORMAT_VERSION,
            "feature_names": list(FEATURE_NAMES),
            "coefficients": None if self.coefficients is None else list(self.coefficients),
            "intercept": self.intercept,
        }

    def save(self, path: Path) -> None:
        """Write this model to a JSON file.

        Parameters
        ----------
        path
            Destination file. Parent directories are created if absent.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Self:
        """Read a model from a JSON file.

        Parameters
        ----------
        path
            Source file written by :meth:`save`.

        Returns
        -------
        Self
            The restored model.

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist. Callers that must not fail should use
            :meth:`load_or_default`.
        ValueError
            If the file's feature order does not match :data:`FEATURE_NAMES`,
            which would make the stored coefficients meaningless.
        """
        payload = json.loads(path.read_text(encoding="utf-8"))
        stored_names = tuple(payload.get("feature_names", ()))
        if stored_names != FEATURE_NAMES:
            msg = (
                "model file was fitted against a different feature order; "
                f"file has {stored_names}, code expects {FEATURE_NAMES}"
            )
            raise ValueError(msg)
        coefficients = payload.get("coefficients")
        return cls(
            coefficients=None if coefficients is None else tuple(coefficients),
            intercept=float(payload.get("intercept", 0.0)),
        )

    @classmethod
    def load_or_default(cls, path: Path | None) -> Self:
        """Read a model if one is available, otherwise return an unfitted one.

        This is what :func:`src.localize.localize` calls. A missing, unreadable
        or stale model file degrades to heuristic scoring rather than failing —
        the deliverable must run on a clean machine with no trained artefacts.

        Parameters
        ----------
        path
            Candidate model file, or ``None`` to skip loading entirely.

        Returns
        -------
        Self
            A fitted model when one could be read, an unfitted one otherwise.
        """
        if path is None or not path.is_file():
            return cls()
        try:
            return cls.load(path)
        except (OSError, ValueError, json.JSONDecodeError):
            return cls()
