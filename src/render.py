"""Two-scale rasterization of continuous layouts, and the ground-truth geometry.

Three choices here are load-bearing.

**Analytic periodic evaluation.** The lattice is evaluated with vectorized
modulo arithmetic over the coordinate arrays, so cost is O(image size) and
independent of die extent. Only the handful of anchor shapes are handled
individually.

**Rotation applied to the sampling window, not the raster.** The offset vector is
rotated before the pattern is evaluated, rather than rasterizing axis-aligned and
rotating the finished image. Ground truth stays exact by construction, and no
interpolation ringing enters the geometry.

**Area-averaged downsampling.** Each output pixel is the mean of a
``supersample x supersample`` block of the binary membership field. A real SEM
pixel integrates signal over its area; bicubic or Lanczos would inject ringing
that physics does not produce.

Ground-truth coordinates
------------------------
``rasterize`` samples pixel ``k`` at continuous position ``k + 0.5``, so the
window centre falls between two pixels on an even-sized image. The *label* for
that position comes from :func:`src.config.image_centre`, which is the single
place the pixel-centre convention is decided. Deriving it there rather than
writing ``out_size / 2`` locally is deliberate: an earlier revision wrote the
literal and put a constant +0.5 px error on every ground-truth value ever
emitted -- 0.707 px radial, against a 1.0 px success tolerance.
"""

from dataclasses import dataclass
from typing import cast

import numpy as np
from scipy.ndimage import map_coordinates

from src import config
from src.layouts import Disc, DramPattern, FinfetPattern, Layout, Pattern, Rect, VerticalLine
from src.types import BoolArray, FloatArray

__all__ = [
    "GroundTruth",
    "PairPlan",
    "plan_pair",
    "raster_centre_base",
    "rasterize",
    "reconstruct_from_gt",
    "render_pair",
]

DEFAULT_SUPERSAMPLE: int = 4
"""Anti-aliasing factor. Each output pixel averages this many samples per axis."""

DEFAULT_EDGE_PAD_PX: float = 8.0
"""Extra clearance kept between the reference footprint and the search border."""


# ===========================================================================
# Pattern fields
# ===========================================================================


def _dram_field(xs: FloatArray, ys: FloatArray, pattern: DramPattern) -> BoolArray:
    """Evaluate DRAM lattice membership at arbitrary layout coordinates.

    Parameters
    ----------
    xs
        Column-axis coordinates in nanometres.
    ys
        Row-axis coordinates in nanometres.
    pattern
        Lattice description.

    Returns
    -------
    BoolArray
        ``True`` where a word line, bit line or via covers the point.
    """
    pitch = np.float32(pattern.pitch_nm)
    half_line = np.float32(pattern.line_width_nm / 2.0)
    via_r = np.float32(pattern.via_nm / 2.0)

    y_mod = np.mod(ys, pitch)
    horizontal = np.minimum(y_mod, pitch - y_mod) <= half_line
    x_mod = np.mod(xs, pitch)
    vertical = np.minimum(x_mod, pitch - x_mod) <= half_line

    row = np.round(ys / pitch)
    offset = np.where(np.mod(row, 2) != 0, pitch / 2.0, 0.0) if pattern.staggered else 0.0
    x_local = xs - offset
    col = np.round(x_local / pitch)
    via = (x_local - col * pitch) ** 2 + (ys - row * pitch) ** 2 <= via_r**2

    return horizontal | vertical | via


def _finfet_field(xs: FloatArray, ys: FloatArray, pattern: FinfetPattern) -> BoolArray:
    """Evaluate FinFET fin and gate membership at arbitrary layout coordinates.

    Parameters
    ----------
    xs
        Column-axis coordinates in nanometres.
    ys
        Row-axis coordinates in nanometres.
    pattern
        Fin and gate description.

    Returns
    -------
    BoolArray
        ``True`` where a fin or a gate bar covers the point.
    """
    fin_pitch = np.float32(pattern.fin_pitch_nm)
    x_mod = np.mod(xs, fin_pitch)
    fins = np.minimum(x_mod, fin_pitch - x_mod) <= np.float32(pattern.fin_width_nm / 2.0)

    gate_pitch = np.float32(pattern.gate_pitch_nm)
    y_shift = ys - np.float32(pattern.gate_pitch_nm / 2.0)
    y_mod = np.mod(y_shift, gate_pitch)
    gates = np.minimum(y_mod, gate_pitch - y_mod) <= np.float32(pattern.gate_width_nm / 2.0)

    return cast(BoolArray, fins | gates)


def _pattern_field(xs: FloatArray, ys: FloatArray, pattern: Pattern) -> BoolArray:
    """Dispatch to the field function for this architecture.

    Parameters
    ----------
    xs
        Column-axis coordinates in nanometres.
    ys
        Row-axis coordinates in nanometres.
    pattern
        Either supported architecture.

    Returns
    -------
    BoolArray
        Pattern membership.
    """
    if isinstance(pattern, DramPattern):
        return _dram_field(xs, ys, pattern)
    return _finfet_field(xs, ys, pattern)


def _disc_mask(xs: FloatArray, ys: FloatArray, disc: Disc) -> BoolArray:
    """Return the membership mask of a filled circle.

    Parameters
    ----------
    xs
        Column-axis coordinates in nanometres.
    ys
        Row-axis coordinates in nanometres.
    disc
        Circle to evaluate.

    Returns
    -------
    BoolArray
        ``True`` inside the circle.
    """
    return (xs - disc.x_nm) ** 2 + (ys - disc.y_nm) ** 2 <= disc.r_nm**2


def _rect_mask(xs: FloatArray, ys: FloatArray, rect: Rect) -> BoolArray:
    """Return the membership mask of an axis-aligned rectangle.

    Parameters
    ----------
    xs
        Column-axis coordinates in nanometres.
    ys
        Row-axis coordinates in nanometres.
    rect
        Rectangle to evaluate.

    Returns
    -------
    BoolArray
        ``True`` inside the rectangle.
    """
    return (xs >= rect.x0_nm) & (xs <= rect.x1_nm) & (ys >= rect.y0_nm) & (ys <= rect.y1_nm)


def _vline_mask(xs: FloatArray, ys: FloatArray, line: VerticalLine) -> BoolArray:
    """Return the membership mask of a vertical bar.

    Parameters
    ----------
    xs
        Column-axis coordinates in nanometres.
    ys
        Row-axis coordinates in nanometres.
    line
        Bar to evaluate.

    Returns
    -------
    BoolArray
        ``True`` inside the bar.
    """
    half = line.width_nm / 2.0
    return (
        (xs >= line.x_nm - half)
        & (xs <= line.x_nm + half)
        & (ys >= line.y0_nm)
        & (ys <= line.y1_nm)
    )


# ===========================================================================
# Rasterization
# ===========================================================================


def raster_centre_base(out_size: int) -> tuple[float, float]:
    """Return the coordinate label of the sampling window's centre.

    Parameters
    ----------
    out_size
        Edge length of the square output image, in pixels.

    Returns
    -------
    tuple of float
        Centre as ``(x, y)``, delegated to :func:`src.config.image_centre` so
        that the pixel-centre convention lives in exactly one place.

    Examples
    --------
    >>> raster_centre_base(1000)
    (499.5, 499.5)
    """
    return config.image_centre((out_size, out_size))


def rasterize(
    layout: Layout,
    centre_nm: tuple[float, float],
    nm_per_px: float,
    out_size: int,
    *,
    rotation_deg: float = 0.0,
    supersample: int = DEFAULT_SUPERSAMPLE,
) -> FloatArray:
    """Render a rotated sampling window over a layout.

    Parameters
    ----------
    layout
        Continuous description of the die region.
    centre_nm
        Window centre as ``(x, y)`` in nanometres.
    nm_per_px
        Physical sampling pitch of the output image.
    out_size
        Edge length of the square output image, in pixels.
    rotation_deg
        Window rotation, positive counter-clockwise.
    supersample
        Anti-aliasing factor per axis.

    Returns
    -------
    FloatArray
        Image in ``[0, 1]``, shape ``(out_size, out_size)``.

    Raises
    ------
    ValueError
        If ``out_size``, ``nm_per_px`` or ``supersample`` is not positive.
    """
    if out_size <= 0 or nm_per_px <= 0 or supersample < 1:
        msg = (
            f"out_size, nm_per_px and supersample must be positive, got "
            f"{out_size!r}, {nm_per_px!r}, {supersample!r}"
        )
        raise ValueError(msg)

    centre_x, centre_y = centre_nm
    n = out_size * supersample
    offsets = ((np.arange(n, dtype=np.float32) + 0.5) / supersample - out_size / 2.0).astype(
        np.float32
    )
    pitch = np.float32(nm_per_px)

    if abs(rotation_deg) < 1e-9:
        xs = (centre_x + offsets * pitch)[None, :].repeat(n, axis=0).astype(np.float32)
        ys = (centre_y + offsets * pitch)[:, None].repeat(n, axis=1).astype(np.float32)
    else:
        theta = np.deg2rad(rotation_deg)
        cos_t, sin_t = np.float32(np.cos(theta)), np.float32(np.sin(theta))
        dx_px, dy_px = np.meshgrid(offsets, offsets)
        dx_nm, dy_nm = dx_px * pitch, dy_px * pitch
        xs = (centre_x + (cos_t * dx_nm - sin_t * dy_nm)).astype(np.float32)
        ys = (centre_y + (sin_t * dx_nm + cos_t * dy_nm)).astype(np.float32)

    field = _pattern_field(xs, ys, layout.pattern).astype(np.float32)
    for erase in layout.erase:
        mask = _disc_mask(xs, ys, erase) if isinstance(erase, Disc) else _rect_mask(xs, ys, erase)
        field = np.where(mask, np.float32(0.0), field)
    for override in layout.overrides:
        mask = (
            _disc_mask(xs, ys, override)
            if isinstance(override, Disc)
            else _vline_mask(xs, ys, override)
        )
        field = np.maximum(field, mask.astype(np.float32))

    blocks = field.reshape(out_size, supersample, out_size, supersample)
    return cast(FloatArray, blocks.mean(axis=(1, 3)).astype(np.float32))


# ===========================================================================
# Pair geometry
# ===========================================================================


@dataclass(frozen=True, slots=True)
class GroundTruth:
    """The answer a matcher is expected to recover.

    Attributes
    ----------
    x
        Reference-crop centre in search-image pixels, column axis, sub-pixel.
    y
        Reference-crop centre in search-image pixels, row axis, sub-pixel.
    rotation_deg
        Search-image rotation relative to the reference, in degrees.
    scale
        Reference-to-search decimation ratio, near ``config.NOMINAL_SCALE``.
    """

    x: float
    y: float
    rotation_deg: float
    scale: float

    def to_dict(self) -> dict[str, float]:
        """Return a JSON-serialisable view of this ground truth.

        Returns
        -------
        dict
            Field names mapped to plain floats.
        """
        return {
            "x": self.x,
            "y": self.y,
            "rotation_deg": self.rotation_deg,
            "scale": self.scale,
        }


@dataclass(frozen=True, slots=True)
class PairPlan:
    """Geometry of one reference/search pair, decided before any layout exists.

    Attributes
    ----------
    crop_centre_nm
        Reference-crop centre in layout coordinates.
    search_centre_nm
        Search-window centre in layout coordinates.
    search_px_nm
        Effective search sampling pitch, including the scale mismatch.
    ground_truth
        The answer implied by this geometry.

    Notes
    -----
    Planning before building is what lets anchors be placed inside the reference
    field of view. Generating the layout first and cropping afterwards leaves
    anchor placement to chance, and the chance is about 2%.
    """

    crop_centre_nm: tuple[float, float]
    search_centre_nm: tuple[float, float]
    search_px_nm: float
    ground_truth: GroundTruth


def plan_pair(
    rng: np.random.Generator,
    *,
    extent_nm: float,
    out_size: int,
    rotation_deg: float = 0.0,
    scale_mismatch: float = 1.0,
    edge_pad_px: float = DEFAULT_EDGE_PAD_PX,
    allow_edge_clipping: bool = False,
) -> PairPlan:
    """Decide the geometry of one pair without rendering anything.

    Parameters
    ----------
    rng
        Seeded generator; the only source of randomness.
    extent_nm
        Edge length of the square die region.
    out_size
        Edge length of both output images, in pixels.
    rotation_deg
        Search-image rotation relative to the reference.
    scale_mismatch
        Multiplier on the nominal search pitch, typically near 1.0.
    edge_pad_px
        Extra clearance between the reference footprint and the search border.
    allow_edge_clipping
        If ``True``, permit the reference region to overhang the search border.

    Returns
    -------
    PairPlan
        Crop centre, search centre, effective pitch and ground truth.

    Raises
    ------
    ValueError
        If the reference footprint cannot fit inside the search image.

    Notes
    -----
    The ground-truth position is sampled *first*, inside a safe band, and the
    layout-space crop centre is solved backwards from it. Containment is then
    guaranteed rather than probabilistic. Drawing the two centres independently
    -- an earlier revision -- put 2.6% of pairs wholly outside the search image
    and clipped a further 10.5%.
    """
    search_px_nm = config.SEARCH_PX_NM * scale_mismatch
    search_fov_nm = out_size * search_px_nm
    base_x, base_y = raster_centre_base(out_size)
    theta = np.deg2rad(rotation_deg)

    margin = min(search_fov_nm / 2 + 50.0, extent_nm / 2 - 10.0)
    search_cx = float(rng.uniform(margin, extent_nm - margin))
    search_cy = float(rng.uniform(margin, extent_nm - margin))

    footprint_px = (out_size * config.REF_PX_NM) / search_px_nm
    half_diag = 0.0 if allow_edge_clipping else footprint_px * float(np.sqrt(2)) / 2.0
    low = half_diag + edge_pad_px
    high = out_size - half_diag - edge_pad_px
    if high <= low:
        msg = f"reference footprint ({footprint_px:.1f} px) does not fit in {out_size} px"
        raise ValueError(msg)

    gt_x = float(rng.uniform(low, high))
    gt_y = float(rng.uniform(low, high))
    rx_nm = (gt_x - base_x) * search_px_nm
    ry_nm = (gt_y - base_y) * search_px_nm
    cos_t, sin_t = float(np.cos(theta)), float(np.sin(theta))

    return PairPlan(
        crop_centre_nm=(
            search_cx + (cos_t * rx_nm - sin_t * ry_nm),
            search_cy + (sin_t * rx_nm + cos_t * ry_nm),
        ),
        search_centre_nm=(search_cx, search_cy),
        search_px_nm=search_px_nm,
        ground_truth=GroundTruth(
            x=gt_x,
            y=gt_y,
            rotation_deg=float(rotation_deg),
            scale=search_px_nm / config.REF_PX_NM,
        ),
    )


def render_pair(
    layout: Layout,
    plan: PairPlan,
    *,
    out_size: int,
    supersample: int = DEFAULT_SUPERSAMPLE,
) -> tuple[FloatArray, FloatArray]:
    """Render the reference and search images for a planned pair.

    Parameters
    ----------
    layout
        Die region, normally built with anchors around ``plan.crop_centre_nm``.
    plan
        Geometry from :func:`plan_pair`.
    out_size
        Edge length of both output images, in pixels.
    supersample
        Anti-aliasing factor per axis.

    Returns
    -------
    tuple of FloatArray
        ``(reference, search)``, both in ``[0, 1]``.
    """
    reference = rasterize(
        layout,
        plan.crop_centre_nm,
        config.REF_PX_NM,
        out_size,
        rotation_deg=0.0,
        supersample=supersample,
    )
    search = rasterize(
        layout,
        plan.search_centre_nm,
        plan.search_px_nm,
        out_size,
        rotation_deg=plan.ground_truth.rotation_deg,
        supersample=supersample,
    )
    return reference, search


def reconstruct_from_gt(
    search: FloatArray,
    ground_truth: GroundTruth,
    *,
    out_size: int,
) -> FloatArray:
    """Rebuild the reference image from the search image and the ground truth.

    Parameters
    ----------
    search
        Search image.
    ground_truth
        Recorded answer for this pair.
    out_size
        Edge length of the reconstructed image, in pixels.

    Returns
    -------
    FloatArray
        What the reference should look like if the ground truth is correct.

    Notes
    -----
    This is the strongest available check on the coordinate convention, because
    it uses *only* published artefacts -- the search image and the ground-truth
    record -- and never touches the layout generator. A convention that is
    merely self-consistent passes an internal check and fails this one.
    """
    search_px_nm = config.REF_PX_NM * ground_truth.scale
    offsets = (np.arange(out_size) + 0.5) - out_size / 2.0
    dx_px, dy_px = np.meshgrid(offsets, offsets)
    dx_nm, dy_nm = dx_px * config.REF_PX_NM, dy_px * config.REF_PX_NM

    theta = np.deg2rad(ground_truth.rotation_deg)
    cos_t, sin_t = float(np.cos(-theta)), float(np.sin(-theta))
    sx = (cos_t * dx_nm - sin_t * dy_nm) / search_px_nm
    sy = (sin_t * dx_nm + cos_t * dy_nm) / search_px_nm

    sampled = map_coordinates(
        search,
        [ground_truth.y + sy, ground_truth.x + sx],
        order=1,
        mode="constant",
        cval=0.0,
    )
    return cast(FloatArray, sampled.astype(np.float32))
