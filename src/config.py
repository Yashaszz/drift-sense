"""Project-wide conventions and tunable constants.

Everything in this module is a *convention* or a *knob*. Nothing here computes.

Why one module
--------------
Several conventions in the problem statement are unresolved — the organizers have
not stated the coordinate origin, the axis order, whether pixel centres sit at
integer or half-integer coordinates, or which pixel grid the success tolerance is
measured in. Our working answers are inferences, not fact.

When an answer arrives, exactly one file changes. If these values were inlined at
their call sites instead, a convention change would mean hunting through five
modules written by four people, and a missed site produces a silent half-pixel
bias that looks like an algorithm defect.

Conventions adopted (all provisional)
-------------------------------------
- Origin at the top-left of the image.
- ``x`` is the column index, ``y`` is the row index.
- Zero-indexed.
- Pixel centres at integer coordinates, so a 1000x1000 image has centre
  ``(499.5, 499.5)``.
- Accuracy tolerances measured in *search*-image pixels (10 nm each).
"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Final

# ---------------------------------------------------------------------------
# Imaging geometry — stated in the problem specification
# ---------------------------------------------------------------------------

REF_PX_NM: Final[float] = 1.0
"""Reference-image pixel size in nanometres (the '100x' optic)."""

SEARCH_PX_NM: Final[float] = 10.0
"""Search-image pixel size in nanometres (the '10x' optic)."""

NOMINAL_SCALE: Final[float] = SEARCH_PX_NM / REF_PX_NM
"""Nominal reference-to-search decimation ratio (10.0).

Given, not estimated from scratch. Stage 1 estimates a small *residual* around
this value: the true ratio is near 10x but not exactly 10.000x.
"""

EXPECTED_IMAGE_SHAPE: Final[tuple[int, int]] = (1000, 1000)
"""Expected ``(rows, cols)`` of both inputs. Advisory only — never enforced."""

TEMPLATE_NOMINAL_PX: Final[int] = 100
"""Nominal template edge length in search pixels (1000 nm / 10 nm per px).

Advisory only. Because the true scale is not exactly 10.000x, the constructed
template is *not* exactly 100x100. Read ``template.shape`` — never assume this.
"""

# ---------------------------------------------------------------------------
# Coordinate conventions — provisional, see module docstring
# ---------------------------------------------------------------------------

ORIGIN_TOP_LEFT: Final[bool] = True
"""``True`` if pixel (0, 0) is the top-left of the image."""

X_AXIS_IS_COLUMN: Final[bool] = True
"""``True`` if ``x`` indexes columns and ``y`` indexes rows."""

PIXEL_CENTRE_AT_INTEGER: Final[bool] = True
"""Pixel-centre convention.

``True``  -- centres at integer coordinates; a 1000-px axis has centre 499.5.
``False`` -- centres at half-integers (0.5, 1.5, ...); that axis has centre 500.0.

Flipping this constant is the *only* change required to switch conventions.
"""

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

TOLERANCES_PX: Final[tuple[float, ...]] = (0.5, 1.0, 2.0)
"""Success tolerances in search pixels, equivalently 5 nm / 10 nm / 20 nm."""

HEADLINE_TOLERANCE_PX: Final[float] = 1.0
"""The tolerance quoted as the headline accuracy figure (10 nm)."""

# ---------------------------------------------------------------------------
# Algorithm defaults
#
# Provisional values. Thresholds are tuned against the full stratified dataset
# in Phase 3 and must be validated across the whole noise range, not fitted at
# one noise level.
# ---------------------------------------------------------------------------

DEFAULT_TOP_K: Final[int] = 30
"""Number of candidate peaks retained from the correlation surface."""

DEFAULT_NMS_RADIUS_PX: Final[int] = 8
"""Fallback non-maximum-suppression radius, in search pixels.

Superseded at runtime by a radius derived from the lattice pitch that Stage 1's
spectral analysis produces for free. Used only when no pitch estimate exists.
"""

DEFAULT_UPSAMPLE: Final[int] = 100
"""Upsampling factor for subpixel refinement, giving 1/100 px resolution."""

DEFAULT_PSF_SIGMA_PX: Final[float] = 1.0
"""Fallback Gaussian PSF sigma in search pixels, when estimation is unreliable."""

DEFAULT_UNIQUENESS_TILE_PX: Final[int] = 64
"""Tile edge length for the Stage 4a uniqueness map, in reference pixels."""

UNIQUENESS_CACHE_ENTRIES: Final[int] = 4
"""How many reference uniqueness maps to keep cached process-wide.

Scoring a reference costs about 400 ms at production shapes — 68% of a
``localize`` call — and depends on the reference alone, so a repeat visit to the
same site should not pay it twice. Each entry holds one ``float32`` map the size
of the reference, so four entries is roughly 16 MB at 1000x1000.

Zero disables the cache and restores per-call computation, which is what a
cold-start benchmark wants.
"""

PSR_ACCEPT_THRESHOLD: Final[float] = 8.0
"""Peak-to-sidelobe ratio above which a match is accepted without escalation."""

PSR_AMBIGUOUS_THRESHOLD: Final[float] = 4.0
"""Peak-to-sidelobe ratio below which the result is treated as ambiguous."""


@dataclass(frozen=True, slots=True)
class Thresholds:
    """The decision points that govern escalation and the low-confidence flag.

    Grouped into one object because they are not independent: ``psr_accept``
    sits above ``psr_ambiguous`` by construction, and tuning one without the
    other produces a ladder whose tiers overlap. Validating them together is
    the only way to catch that.

    These are the numbers Phase 3 tuning sweeps over. Holding them in a value
    object rather than reading module constants at each call site means a sweep
    can vary them per iteration without editing source or reimporting, and means
    a fitted set can eventually be loaded from disk the way the confidence model
    already is.
    """

    psr_accept: float = PSR_ACCEPT_THRESHOLD
    psr_ambiguous: float = PSR_AMBIGUOUS_THRESHOLD

    def __post_init__(self) -> None:
        """Reject a ladder whose tiers cannot be ordered."""
        if not self.psr_accept >= self.psr_ambiguous:
            msg = (
                f"psr_accept ({self.psr_accept}) must be >= psr_ambiguous "
                f"({self.psr_ambiguous}); otherwise the robust tier accepts "
                "answers the fast tier would have escalated"
            )
            raise ValueError(msg)

    def for_tier(self, tier: str) -> float:
        """Return the PSR a tier must beat to stop escalating.

        Parameters
        ----------
        tier
            Resolved operating mode.

        Returns
        -------
        float
            ``psr_accept`` for the cheap tier, ``psr_ambiguous`` beyond it. The
            bar drops as the pipeline spends more: having already paid for the
            expensive path, a weaker peak is the best evidence available.
        """
        return self.psr_accept if tier == "fast" else self.psr_ambiguous


_ACTIVE_THRESHOLDS: Thresholds = Thresholds()


def get_thresholds() -> Thresholds:
    """Return the thresholds currently in force.

    Read at each decision point rather than captured at import, so that a sweep
    or a test can change them without reloading the module.
    """
    return _ACTIVE_THRESHOLDS


def set_thresholds(thresholds: Thresholds) -> None:
    """Install a new threshold set process-wide.

    Prefer :func:`override_thresholds` where the change is scoped; this exists
    for a tuned set loaded once at start-up.
    """
    global _ACTIVE_THRESHOLDS  # noqa: PLW0603 - one process-wide policy, by design
    _ACTIVE_THRESHOLDS = thresholds


@contextmanager
def override_thresholds(**changes: float) -> Iterator[Thresholds]:
    """Temporarily replace one or more thresholds.

    Restores the previous set on exit, including when the body raises, so a
    sweep that fails partway cannot leave the process mistuned for every test
    that follows.

    Examples
    --------
    >>> with override_thresholds(psr_accept=3.0) as t:
    ...     t.psr_accept
    3.0
    """
    previous = _ACTIVE_THRESHOLDS
    set_thresholds(replace(previous, **changes))
    try:
        yield _ACTIVE_THRESHOLDS
    finally:
        set_thresholds(previous)


TIE_SIGMA: Final[float] = 0.0
"""Tie width, in units of the statistic the call site converts from.

Zero means *exact ties only*: a candidate goes to the mandated centre tie-break
only when its score equals the maximum bit for bit.

That is deliberately conservative, and it was chosen from measurement. The call
site converts this into score units by multiplying by the sidelobe standard
deviation, and on a periodic layout that deviation is around 0.29 because the
sidelobe region spans nearly the whole correlation range. Meanwhile the top
thirty candidates are separated by roughly 0.025 in total. A width of one
sidelobe sigma therefore declared *every* candidate tied and handed the answer
to the centre rule, discarding a peak standing clearly above the rest.

Measured over 108 generated pairs:

    tie width   accuracy @ 1 px   median candidates tied
    0.000            35.2%                  1
    0.002            32.4%                 30
    0.010            25.0%                 30
    0.291             3.7%                 30   <- one sidelobe sigma

The handbook specifies ties as "within roughly one *noise* sigma of the
maximum". Noise sigma and sidelobe sigma are different quantities: the first is
the scale of random fluctuation between neighbouring candidates, the second is
the spread of the entire surface. Conflating them is what cost the accuracy
above.

Zero is therefore a floor, not the final answer. The proper fix is to estimate
the local fluctuation scale among the top candidates and express the tie width
in that, which restores the mandated tie-break to the cases it was meant for
instead of disabling it in practice. Until then, exact ties only.
"""

CONFIDENCE_MODEL_PATH: Final[str] = "models/confidence.json"
"""Where a fitted Stage 6 calibrator is looked for, relative to the repo root.

Absence is normal and must stay harmless: the deliverable has to run from a
clean unzip with no trained artefacts, so a missing file falls back to the
heuristic rather than failing.
"""

# ---------------------------------------------------------------------------
# Conversions and coordinate helpers
# ---------------------------------------------------------------------------


def search_px_to_nm(pixels: float) -> float:
    """Convert a distance in search-image pixels to nanometres.

    Parameters
    ----------
    pixels
        Distance in search-image pixels.

    Returns
    -------
    float
        The same distance in nanometres.
    """
    return pixels * SEARCH_PX_NM


def nm_to_search_px(nanometres: float) -> float:
    """Convert a distance in nanometres to search-image pixels.

    Parameters
    ----------
    nanometres
        Distance in nanometres.

    Returns
    -------
    float
        The same distance in search-image pixels.
    """
    return nanometres / SEARCH_PX_NM


def image_centre(shape: tuple[int, int]) -> tuple[float, float]:
    """Return the centre of an image as ``(x, y)``.

    The mandated tie-break rule selects the candidate closest to the centre of
    the *search* image, so this value directly decides ambiguous cases. It is
    also the maximum-prior fallback answer when localization fails outright.

    Parameters
    ----------
    shape
        Image shape as ``(rows, cols)``, matching NumPy's convention.

    Returns
    -------
    tuple of float
        Centre as ``(x, y)`` — that is, ``(column, row)``.

    Examples
    --------
    >>> image_centre((1000, 1000))
    (499.5, 499.5)
    """
    rows, cols = shape
    if PIXEL_CENTRE_AT_INTEGER:
        return ((cols - 1) / 2.0, (rows - 1) / 2.0)
    return (cols / 2.0, rows / 2.0)


def window_topleft_to_centre(
    col: int,
    row: int,
    template_shape: tuple[int, int],
) -> tuple[float, float]:
    """Convert a correlation-surface index to the centre of the matched window.

    A correlation surface indexes the *top-left corner* of the template window,
    not its centre: ``surface[r, c]`` scores the template placed with its corner
    at row ``r``, column ``c``. The deliverable must report the window's centre.

    Confusing the two produces a constant offset of roughly half the template
    edge — about 50 px for a 100x100 template. That is a systematic error which
    looks exactly like a broken matcher, so the conversion lives in one tested
    function and is never written inline.

    Parameters
    ----------
    col
        Column index into the correlation surface.
    row
        Row index into the correlation surface.
    template_shape
        Template shape as ``(rows, cols)``. Read this from the actual array; the
        template is not exactly ``TEMPLATE_NOMINAL_PX`` square once the scale
        residual is applied.

    Returns
    -------
    tuple of float
        Centre of the matched window as ``(x, y)`` in search-image pixels.

    Examples
    --------
    >>> window_topleft_to_centre(0, 0, (100, 100))
    (49.5, 49.5)
    >>> window_topleft_to_centre(200, 350, (100, 100))
    (249.5, 399.5)
    """
    t_rows, t_cols = template_shape
    if PIXEL_CENTRE_AT_INTEGER:
        return (col + (t_cols - 1) / 2.0, row + (t_rows - 1) / 2.0)
    return (col + t_cols / 2.0, row + t_rows / 2.0)


# --- Stage 4a: uniqueness weighting -----------------------------------------
UNIQUENESS_PREFILTER_SIGMA_PX: Final[float] = 4.0
UNIQUENESS_FLOOR: Final[float] = 0.05
UNIQUENESS_MIN_TILE_PX: Final[int] = 16

# --- PSR --------------------------------------------------------------------
# Split from DEFAULT_NMS_RADIUS_PX: correlation main-lobe width, not lattice
# pitch. Equal at 8 today by coincidence; must not track each other once R1
# randomises pitch.
PSR_EXCLUSION_RADIUS_PX: Final[int] = 8
