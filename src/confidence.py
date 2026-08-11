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
Fitted, serialised and wired (T7). On the current data it does **not**
discriminate: cross-validated AUC is 0.504 over 108 pairs, which is chance.

That is a data problem, not a model problem. Four of the six features are
constant because the diagnostics feeding them are still stubs: ``n_tied`` and
the pose residuals never vary, and ``uniqueness_score`` is a constant while R3's
map returns ones. Only ``psr`` (AUC 0.606) and ``ncc_peak`` (0.540) carry
anything at all.

The signal that would work is known and measured. Accuracy splits 77.8% on
anchored references against 0.0% on unanchored ones, and ``uniqueness_score`` is
precisely the feature designed to capture that. Substituting a working map in
the counterfactual takes cross-validated AUC from 0.506 to **0.926**.

So no fitted model ships. ``localize`` falls back to the conservative heuristic,
which is honest about knowing little, rather than to a calibrator that would
look authoritative at chance level - the handbook is explicit that a decorative
confidence earns nothing. ``benchmarks/fit_confidence.py`` produces a real model
the moment the upstream stubs land, and dropping the file at
``config.CONFIDENCE_MODEL_PATH`` activates it with no code change.
"""

import json
from pathlib import Path
from typing import Any, Self

import numpy as np

from src import config
from src.types import Diagnostics, Float64Array

_HEURISTIC_SCALE: float = 2.0
"""Width of the heuristic logistic, in PSR units."""

_MIN_FEATURE_SPREAD: float = 1e-9
"""Standard deviation below which a feature is treated as constant."""

_DEFAULT_THRESHOLD: float = 0.5
"""Decision threshold used until one is calibrated from data."""

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
    A logistic in the peak-to-sidelobe ratio, centred on the acceptance
    threshold: it crosses 0.5 exactly where the escalation logic stops
    escalating, so the two agree by construction rather than by coincidence.

    Deliberately a function of PSR alone. On the current data PSR is the only
    diagnostic carrying meaningful signal, and a heuristic that pretended to
    weigh several would imply a confidence it cannot support. A non-finite PSR
    means the ambiguity could not be measured, which scores zero.
    """
    if not np.isfinite(diagnostics.psr):
        return 0.0
    centred = (diagnostics.psr - config.PSR_ACCEPT_THRESHOLD) / _HEURISTIC_SCALE
    return float(1.0 / (1.0 + np.exp(-centred)))


def is_low_confidence(
    confidence: float,
    diagnostics: Diagnostics,
    threshold: float = _DEFAULT_THRESHOLD,
) -> bool:
    """Decide whether the tool should escalate rather than trust this answer.

    Parameters
    ----------
    confidence
        Calibrated confidence in ``[0, 1]``.
    diagnostics
        Evidence gathered during localization.
    threshold
        Confidence below which the answer is not trusted. Comes from the fitted
        model when one is available, so the decision point is calibrated rather
        than assumed.

    Returns
    -------
    bool
        ``True`` when the answer should be treated as unreliable.

    Notes
    -----
    Deliberately not a bare threshold on ``confidence``. An internal error, an
    unmeasurable statistic, or an answer that correlation evidence did not
    actually decide all flag the result regardless of what the score says, so
    that a miscalibrated model cannot mask a known-bad case.

    Every gate here is one-directional: each can only *raise* the flag, never
    clear it. That is what lets an unfitted or badly fitted calibrator degrade
    the number without degrading the safety property.
    """
    if diagnostics.failure_mode != "none":
        return True
    if not np.isfinite(diagnostics.psr):
        # NaN means the ambiguity measure could not be computed, not that the
        # answer is unambiguous. A bare ``psr < threshold`` comparison answers
        # False for NaN, which would clear the flag exactly when there is no
        # evidence to clear it with.
        return True
    if diagnostics.psr < config.get_thresholds().psr_ambiguous:
        return True
    if diagnostics.tie_break_used and diagnostics.n_tied > 1:
        # The centre tie-break is a prior, not evidence. When it decided between
        # genuinely tied candidates the answer rests on "the stage aimed here",
        # which is exactly the unanchored case that cannot be solved. The
        # correlation surface ranked nothing, so the score above it is not
        # measuring this answer.
        return True
    if not np.isfinite(confidence):
        # Same failure class as the NaN psr above, one level up: a model that
        # returns NaN would sail through ``confidence < threshold`` as False.
        return True
    # Coerced because a numpy scalar confidence yields np.bool_, which json
    # cannot serialise and which the CLI's --json path would fail on.
    return bool(confidence < threshold)


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

    FORMAT_VERSION: int = 2
    """Serialisation version, written into the model file.

    Version 2 adds the feature standardisation and the decision threshold, both
    of which are fitted quantities and meaningless to infer without.
    """

    def __init__(
        self,
        coefficients: tuple[float, ...] | None = None,
        intercept: float = 0.0,
        centre: tuple[float, ...] | None = None,
        spread: tuple[float, ...] | None = None,
        threshold: float = _DEFAULT_THRESHOLD,
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
        self.centre = centre
        self.spread = spread
        self.threshold = threshold

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

        from sklearn.linear_model import LogisticRegression

        clean = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        centre = clean.mean(axis=0)
        spread = clean.std(axis=0)
        # A feature with no variance carries no information and would otherwise
        # divide by zero. Several are constant today because the diagnostics
        # feeding them are still stubs upstream.
        spread = np.where(spread < _MIN_FEATURE_SPREAD, 1.0, spread)

        # `penalty` is deprecated from scikit-learn 1.8; C alone selects the same
        # L2 regularisation and keeps the call forward-compatible.
        estimator = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            random_state=0,
        )
        estimator.fit((clean - centre) / spread, correct.astype(int))

        self.coefficients = tuple(float(c) for c in estimator.coef_[0])
        self.intercept = float(estimator.intercept_[0])
        self.centre = tuple(float(c) for c in centre)
        self.spread = tuple(float(s) for s in spread)
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
        if not self.is_fitted or self.centre is None or self.spread is None:
            return heuristic_confidence(diagnostics)

        assert self.coefficients is not None  # noqa: S101 - narrowed by is_fitted
        standardised = (extract_features(diagnostics) - np.asarray(self.centre)) / np.asarray(
            self.spread
        )
        logit = float(np.dot(standardised, np.asarray(self.coefficients))) + self.intercept
        return float(np.clip(1.0 / (1.0 + np.exp(-logit)), 0.0, 1.0))

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
            "centre": None if self.centre is None else list(self.centre),
            "spread": None if self.spread is None else list(self.spread),
            "threshold": self.threshold,
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
        centre = payload.get("centre")
        spread = payload.get("spread")
        return cls(
            coefficients=None if coefficients is None else tuple(coefficients),
            intercept=float(payload.get("intercept", 0.0)),
            centre=None if centre is None else tuple(centre),
            spread=None if spread is None else tuple(spread),
            threshold=float(payload.get("threshold", _DEFAULT_THRESHOLD)),
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
