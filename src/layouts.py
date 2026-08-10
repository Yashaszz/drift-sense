"""Continuous-coordinate DRAM and FinFET layouts, in nanometres.

A layout is a *description*, never pixels. The periodic bulk is stated
analytically -- pitch, linewidth, via size -- and only the handful of aperiodic
anchor features are listed individually. :mod:`src.render` evaluates the bulk
with modulo arithmetic in one pass, so render cost is independent of how large
the die region is. An earlier per-shape version took 15-20 s per image.

Why anchors are placed relative to the crop
-------------------------------------------
Anchors are what make localization possible at all: on a pure lattice, thousands
of positions tie for best match. They therefore have to be *inside the reference
crop*, not merely somewhere on the die.

That is not automatic. The reference field of view is 1000 nm across inside a
12,000 nm layout -- 0.69% of the area -- so anchors scattered at random lattice
sites land inside the crop about 2% of the time. Measured on an earlier
revision: 1 of 18 pairs labelled ``anchored`` actually contained one, which made
the anchored/unanchored stratum measure nothing.

Both generators therefore take ``anchor_centre_nm``: the crop centre, decided by
:func:`src.render.plan_pair` *before* the layout exists.
"""

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np

__all__ = [
    "Anchor",
    "Disc",
    "DramPattern",
    "FinfetPattern",
    "Layout",
    "Rect",
    "VerticalLine",
    "generate_dram_layout",
    "generate_finfet_layout",
]

DEFAULT_ANCHOR_HALF_SPAN_NM: float = 340.0
"""Half-width of the box anchors are placed in, in nanometres.

Comfortably inside the 500 nm half-field of the reference crop, leaving room for
the anchor's own extent and for the crop to be sampled at a rotated pose.
"""


# ===========================================================================
# Geometry primitives
# ===========================================================================


@dataclass(frozen=True, slots=True)
class Disc:
    """A filled circle in layout coordinates.

    Attributes
    ----------
    x_nm
        Centre, column axis, in nanometres.
    y_nm
        Centre, row axis, in nanometres.
    r_nm
        Radius in nanometres.
    """

    x_nm: float
    y_nm: float
    r_nm: float


@dataclass(frozen=True, slots=True)
class Rect:
    """An axis-aligned rectangle in layout coordinates.

    Attributes
    ----------
    x0_nm
        Left edge in nanometres.
    x1_nm
        Right edge in nanometres.
    y0_nm
        Top edge in nanometres.
    y1_nm
        Bottom edge in nanometres.
    """

    x0_nm: float
    x1_nm: float
    y0_nm: float
    y1_nm: float


@dataclass(frozen=True, slots=True)
class VerticalLine:
    """A vertical bar of finite width, used for the merged-fin anchor.

    Attributes
    ----------
    x_nm
        Centre line, column axis, in nanometres.
    width_nm
        Full width in nanometres.
    y0_nm
        Start of the bar, row axis, in nanometres.
    y1_nm
        End of the bar, row axis, in nanometres.
    """

    x_nm: float
    width_nm: float
    y0_nm: float
    y1_nm: float


Erase: TypeAlias = Disc | Rect
"""A region subtracted from the pattern field."""

Override: TypeAlias = Disc | VerticalLine
"""A region added to the pattern field after erasure."""


@dataclass(frozen=True, slots=True)
class Anchor:
    """Ground-truth record of one aperiodic feature.

    Attributes
    ----------
    anchor_type
        Defect class, e.g. ``"missing_via"``.
    x_nm
        Location, column axis, in nanometres.
    y_nm
        Location, row axis, in nanometres.
    """

    anchor_type: str
    x_nm: float
    y_nm: float

    def to_dict(self) -> dict[str, float | str]:
        """Return a JSON-serialisable view of this anchor.

        Returns
        -------
        dict
            Field names mapped to plain Python values.
        """
        return {"anchor_type": self.anchor_type, "x_nm": self.x_nm, "y_nm": self.y_nm}


# ===========================================================================
# Patterns
# ===========================================================================


@dataclass(frozen=True, slots=True)
class DramPattern:
    """Word-line / bit-line lattice with a contact via at each crossing.

    Attributes
    ----------
    pitch_nm
        Lattice pitch, identical on both axes.
    line_width_nm
        Width of a word line or bit line.
    via_nm
        Via *diameter*. The rendered radius is half this.
    extent_nm
        Edge length of the square die region.
    staggered
        Whether alternate rows offset their vias by half a pitch.
    """

    pitch_nm: float
    line_width_nm: float
    via_nm: float
    extent_nm: float
    staggered: bool
    kind: Literal["dram"] = "dram"

    def to_dict(self) -> dict[str, float | bool | str]:
        """Return a JSON-serialisable view of this pattern.

        Returns
        -------
        dict
            Field names mapped to plain Python values.
        """
        return {
            "kind": self.kind,
            "pitch_nm": self.pitch_nm,
            "line_width_nm": self.line_width_nm,
            "via_nm": self.via_nm,
            "extent_nm": self.extent_nm,
            "staggered": self.staggered,
        }


@dataclass(frozen=True, slots=True)
class FinfetPattern:
    """Parallel fins crossed by regularly spaced gate bars.

    Attributes
    ----------
    fin_pitch_nm
        Centre-to-centre fin spacing.
    fin_width_nm
        Width of one fin.
    gate_width_nm
        Width of one gate bar.
    gate_pitch_nm
        Centre-to-centre gate spacing.
    extent_nm
        Edge length of the square die region.

    Notes
    -----
    ``gate_pitch_nm`` matters more than it looks. An earlier revision placed two
    gates across the whole 12,000 nm layout, so a 1000 nm reference crop
    contained a gate roughly 17% of the time and FinFET references were, in
    practice, nothing but parallel fins.
    """

    fin_pitch_nm: float
    fin_width_nm: float
    gate_width_nm: float
    gate_pitch_nm: float
    extent_nm: float
    kind: Literal["finfet"] = "finfet"

    def to_dict(self) -> dict[str, float | str]:
        """Return a JSON-serialisable view of this pattern.

        Returns
        -------
        dict
            Field names mapped to plain Python values.
        """
        return {
            "kind": self.kind,
            "fin_pitch_nm": self.fin_pitch_nm,
            "fin_width_nm": self.fin_width_nm,
            "gate_width_nm": self.gate_width_nm,
            "gate_pitch_nm": self.gate_pitch_nm,
            "extent_nm": self.extent_nm,
        }


Pattern: TypeAlias = DramPattern | FinfetPattern
"""Either supported architecture."""


@dataclass(frozen=True, slots=True)
class Layout:
    """A die region: periodic bulk plus its aperiodic anchor features.

    Attributes
    ----------
    pattern
        Analytic description of the periodic bulk.
    erase
        Regions removed from the bulk.
    overrides
        Regions added back after erasure.
    anchors
        Ground-truth locations of the aperiodic features.
    anchored
        Whether this layout carries anchors at all. ``False`` reproduces the
        documented tie/ambiguity failure case.
    """

    pattern: Pattern
    erase: tuple[Erase, ...]
    overrides: tuple[Override, ...]
    anchors: tuple[Anchor, ...]
    anchored: bool


# ===========================================================================
# DRAM
# ===========================================================================


def _dram_sites_near(
    centre_nm: tuple[float, float],
    pitch_nm: float,
    *,
    staggered: bool,
    half_span_nm: float,
) -> list[tuple[float, float]]:
    """List lattice crossings inside a box around a point.

    Parameters
    ----------
    centre_nm
        Box centre as ``(x, y)`` in nanometres.
    pitch_nm
        Lattice pitch.
    staggered
        Whether alternate rows offset by half a pitch.
    half_span_nm
        Half-width of the box.

    Returns
    -------
    list of tuple of float
        Via positions as ``(x, y)`` in nanometres.
    """
    cx, cy = centre_nm
    sites: list[tuple[float, float]] = []
    row_lo = int(np.ceil((cy - half_span_nm) / pitch_nm))
    row_hi = int(np.floor((cy + half_span_nm) / pitch_nm))
    for row in range(row_lo, row_hi + 1):
        offset = pitch_nm / 2.0 if (staggered and row % 2 == 1) else 0.0
        col_lo = int(np.ceil((cx - half_span_nm - offset) / pitch_nm))
        col_hi = int(np.floor((cx + half_span_nm - offset) / pitch_nm))
        sites.extend((col * pitch_nm + offset, row * pitch_nm) for col in range(col_lo, col_hi + 1))
    return sites


def generate_dram_layout(
    extent_nm: float,
    pitch_nm: float,
    line_width_nm: float,
    via_nm: float,
    rng: np.random.Generator,
    *,
    variant: Literal["orthogonal", "staggered"] = "orthogonal",
    anchored: bool = True,
    anchor_centre_nm: tuple[float, float] | None = None,
    anchor_half_span_nm: float = DEFAULT_ANCHOR_HALF_SPAN_NM,
) -> Layout:
    """Build a DRAM word-line/bit-line layout.

    Parameters
    ----------
    extent_nm
        Edge length of the square die region.
    pitch_nm
        Lattice pitch, identical on both axes.
    line_width_nm
        Width of a word line or bit line.
    via_nm
        Via diameter.
    rng
        Seeded generator; the only source of randomness.
    variant
        ``"staggered"`` offsets vias on alternate rows by half a pitch.
    anchored
        ``False`` produces a pure periodic grid with no anchors.
    anchor_centre_nm
        Point to place anchors around, normally the reference crop centre.
        Defaults to the layout centre.
    anchor_half_span_nm
        Half-width of the box anchors are drawn from.

    Returns
    -------
    Layout
        Pattern, erase and override lists, and anchor ground truth.

    Notes
    -----
    The three anchor types are drawn from documented DRAM contact defects:
    a missing via (redundancy or repair site), an oversized via (local process
    excursion) and a shifted via (contact misalignment).

    The missing-via erase radius is the via *radius*, not its diameter. An
    earlier revision erased at the diameter, which removed the via and bit into
    both crossing lines -- a hole punched through the wiring rather than an
    absent contact.
    """
    staggered = variant == "staggered"
    pattern = DramPattern(
        pitch_nm=float(pitch_nm),
        line_width_nm=float(line_width_nm),
        via_nm=float(via_nm),
        extent_nm=float(extent_nm),
        staggered=staggered,
    )
    if not anchored:
        return Layout(pattern=pattern, erase=(), overrides=(), anchors=(), anchored=False)

    centre = anchor_centre_nm if anchor_centre_nm is not None else (extent_nm / 2, extent_nm / 2)
    span = anchor_half_span_nm
    sites = _dram_sites_near(centre, pitch_nm, staggered=staggered, half_span_nm=span)
    if len(sites) < 3:
        # Widening the search would place anchors outside the reference crop,
        # which is the exact failure this parameter exists to prevent -- so say
        # so instead. It means the crop is too small to hold three lattice
        # sites: raise anchor_half_span_nm, or use a coarser pitch.
        msg = (
            f"only {len(sites)} lattice sites within {span:.0f} nm of the crop centre "
            f"at pitch {pitch_nm:.0f} nm; need 3"
        )
        raise ValueError(msg)

    chosen = rng.choice(len(sites), size=3, replace=False)
    (x0, y0), (x1, y1), (x2, y2) = (sites[int(i)] for i in chosen)
    via_r = via_nm / 2.0

    erase: list[Erase] = [
        Disc(x0, y0, via_r),
        Disc(x1, y1, via_r),
        Disc(x2, y2, via_r),
    ]
    overrides: list[Override] = [
        Disc(x1, y1, via_r * 1.8),
        Disc(x2 + pitch_nm * 0.28, y2 + pitch_nm * 0.12, via_r),
    ]
    anchors = (
        Anchor("missing_via", x0, y0),
        Anchor("oversized_via", x1, y1),
        Anchor("shifted_via", x2, y2),
    )
    return Layout(
        pattern=pattern,
        erase=tuple(erase),
        overrides=tuple(overrides),
        anchors=anchors,
        anchored=True,
    )


# ===========================================================================
# FinFET
# ===========================================================================


def generate_finfet_layout(
    extent_nm: float,
    fin_pitch_nm: float,
    fin_width_nm: float,
    gate_width_nm: float,
    rng: np.random.Generator,
    *,
    gate_pitch_nm: float = 420.0,
    anchored: bool = True,
    anchor_centre_nm: tuple[float, float] | None = None,
    anchor_half_span_nm: float = DEFAULT_ANCHOR_HALF_SPAN_NM,
) -> Layout:
    """Build a FinFET fin-and-gate layout.

    Parameters
    ----------
    extent_nm
        Edge length of the square die region.
    fin_pitch_nm
        Centre-to-centre fin spacing.
    fin_width_nm
        Width of one fin.
    gate_width_nm
        Width of one gate bar.
    rng
        Seeded generator; the only source of randomness.
    gate_pitch_nm
        Centre-to-centre gate spacing.
    anchored
        ``False`` produces bare fins and gates with no anchors.
    anchor_centre_nm
        Point to place anchors around, normally the reference crop centre.
    anchor_half_span_nm
        Half-width of the box anchors are drawn from.

    Returns
    -------
    Layout
        Pattern, erase and override lists, and anchor ground truth.

    Notes
    -----
    Gate bars belong to the analytic field, not to the override list. At a
    realistic gate pitch there are tens of them per layout, and one array pass
    per gate is exactly the per-shape cost this renderer exists to avoid.
    Only the *broken* gate needs an explicit shape.
    """
    pattern = FinfetPattern(
        fin_pitch_nm=float(fin_pitch_nm),
        fin_width_nm=float(fin_width_nm),
        gate_width_nm=float(gate_width_nm),
        gate_pitch_nm=float(gate_pitch_nm),
        extent_nm=float(extent_nm),
    )
    if not anchored:
        return Layout(pattern=pattern, erase=(), overrides=(), anchors=(), anchored=False)

    cx, cy = anchor_centre_nm if anchor_centre_nm is not None else (extent_nm / 2, extent_nm / 2)
    span = anchor_half_span_nm

    fin_lo = int(np.ceil((cx - span) / fin_pitch_nm))
    fin_hi = int(np.floor((cx + span) / fin_pitch_nm))
    if fin_hi - fin_lo < 1:
        msg = (
            f"only {fin_hi - fin_lo + 1} fins within {span:.0f} nm of the crop centre "
            f"at pitch {fin_pitch_nm:.0f} nm; need 2"
        )
        raise ValueError(msg)
    indices = list(range(fin_lo, fin_hi + 1))
    chosen = rng.choice(len(indices), size=2, replace=False)
    i_missing, i_double = indices[int(chosen[0])], indices[int(chosen[1])]

    x_missing = i_missing * fin_pitch_nm
    x_double = i_double * fin_pitch_nm
    cut_half = span * 0.8
    fin_half = fin_pitch_nm * 0.6

    # Fin cut: a span, not a pinhole. Erasing a disc from a fin that runs the
    # full die height is a nick, which is not what "missing fin" means.
    erase: list[Erase] = [
        Rect(x_missing - fin_half, x_missing + fin_half, cy - cut_half, cy + cut_half),
        Disc(x_double, cy, fin_half),
    ]
    overrides: list[Override] = [
        VerticalLine(x_double, fin_width_nm * 2.2, -fin_pitch_nm, extent_nm + fin_pitch_nm),
    ]

    gate_y = float(
        np.round((cy - gate_pitch_nm / 2.0) / gate_pitch_nm) * gate_pitch_nm + gate_pitch_nm / 2.0
    )
    x_break = float(cx + rng.uniform(-span * 0.5, span * 0.5))
    erase.append(
        Rect(x_break - 30.0, x_break + 30.0, gate_y - gate_width_nm, gate_y + gate_width_nm)
    )

    anchors = (
        Anchor("missing_fin", x_missing, cy),
        Anchor("merged_fin", x_double, cy),
        Anchor("gate_break", x_break, gate_y),
    )
    return Layout(
        pattern=pattern,
        erase=tuple(erase),
        overrides=tuple(overrides),
        anchors=anchors,
        anchored=True,
    )
