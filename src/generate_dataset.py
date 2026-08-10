"""Stratified synthetic dataset generation, and the ground-truth JSONL writer.

Produces reference/search pairs across ``architecture x anchor x pose`` and emits
one JSON object per pair. No real fab data exists for this problem, so this
module is the source of every number the accuracy score is measured against --
which is why :func:`validate_record` is fatal rather than advisory.

Ordering matters
----------------
The pair geometry is planned first, the layout is built around the resulting
crop centre, and only then is anything rendered. See :class:`src.render.PairPlan`.
"""

import argparse
import json
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image

from src import config
from src.layouts import Layout, generate_dram_layout, generate_finfet_layout
from src.render import (
    DEFAULT_SUPERSAMPLE,
    GroundTruth,
    PairPlan,
    plan_pair,
    render_pair,
)
from src.types import FloatArray

__all__ = ["PairRecord", "build_dataset", "main", "validate_record"]

DEFAULT_SEED: int = 20260807
"""Frozen base seed. Every pair's seed is this plus its index."""

EXTENT_NM: float = 12_000.0
"""Die-region edge length. Comfortably larger than the 10 um search field."""

OUT_SIZE: int = 1000
"""Edge length of both images, in pixels."""

POSE_RANGES: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {
    "none": ((0.0, 0.0), (1.0, 1.0)),
    "small": ((-5.0, 5.0), (0.97, 1.03)),
    "large": ((-8.0, 8.0), (0.95, 1.05)),
}
"""Rotation and scale-mismatch ranges per pose stratum.

``small`` is the Phase-2 baseline from the work-split document: rotation +-5 deg,
scale +-3%. ``large`` deliberately exceeds it, so the dataset carries stress
cases beyond the range anyone is tuning against.
"""

NOISE_LEVELS: tuple[str, ...] = ("none",)
"""Noise strata generated.

Currently one inert level, because :func:`apply_sem_chain` is a passthrough
until ``src.sem_physics`` lands. Change to ``("low", "medium", "high")`` in the
same commit that wires R2's module in -- the stratification loop and the record
schema already handle it, and :data:`NOISE_SCALING` holds the intended mapping.
Generating three identical noise strata before the physics is real would put a
column in R3's per-stratum table that cannot differ, which is worse than a
column that is honestly absent.
"""

NOISE_SCALING: dict[str, dict[str, float]] = {
    "low": {"dose": 2.0, "read_noise": 0.5},
    "medium": {"dose": 1.0, "read_noise": 1.0},
    "high": {"dose": 0.35, "read_noise": 2.0},
}
"""Provisional multipliers on R2's preset dose and read-noise sigma.

``medium`` is R2's preset unchanged. Dose drives shot noise, so halving it
raises noise as the square root; read-noise sigma is additive and scales
directly. **Not yet confirmed with R2** -- they shipped two captures presets
(reference/search) rather than three severity levels, so this mapping is R1's
proposal pending their answer.
"""


def _tolerance_range(nominal_nm: float, fraction: float) -> tuple[float, float]:
    """Return a symmetric randomisation range around a nominal dimension.

    Parameters
    ----------
    nominal_nm
        Centre of the range, in nanometres.
    fraction
        Half-width as a fraction of nominal, e.g. ``0.20`` for +-20%.

    Returns
    -------
    tuple of float
        ``(low, high)`` in nanometres.
    """
    return (nominal_nm * (1.0 - fraction), nominal_nm * (1.0 + fraction))


DRAM_NOMINAL_NM: dict[str, float] = {"pitch_nm": 180.0, "line_width_nm": 40.0, "via_nm": 60.0}
"""Nominal DRAM dimensions the randomisation is centred on."""

FINFET_NOMINAL_NM: dict[str, float] = {
    "fin_pitch_nm": 90.0,
    "fin_width_nm": 24.0,
    "gate_width_nm": 13.0,
    "gate_pitch_nm": 420.0,
}
"""Nominal FinFET dimensions the randomisation is centred on."""

PITCH_TOLERANCE: float = 0.20
"""Pitch randomisation, +-20% of nominal, per the work-split document."""

WIDTH_TOLERANCE: float = 0.15
"""Linewidth randomisation, +-15% of nominal, per the work-split document."""

DRAM_RANGES: dict[str, tuple[float, float]] = {
    "pitch_nm": _tolerance_range(DRAM_NOMINAL_NM["pitch_nm"], PITCH_TOLERANCE),
    "line_width_nm": _tolerance_range(DRAM_NOMINAL_NM["line_width_nm"], WIDTH_TOLERANCE),
    "via_nm": _tolerance_range(DRAM_NOMINAL_NM["via_nm"], WIDTH_TOLERANCE),
}
"""Domain-randomisation ranges for DRAM, in nanometres."""

FINFET_RANGES: dict[str, tuple[float, float]] = {
    "fin_pitch_nm": _tolerance_range(FINFET_NOMINAL_NM["fin_pitch_nm"], PITCH_TOLERANCE),
    "fin_width_nm": _tolerance_range(FINFET_NOMINAL_NM["fin_width_nm"], WIDTH_TOLERANCE),
    "gate_width_nm": _tolerance_range(FINFET_NOMINAL_NM["gate_width_nm"], WIDTH_TOLERANCE),
    "gate_pitch_nm": _tolerance_range(FINFET_NOMINAL_NM["gate_pitch_nm"], PITCH_TOLERANCE),
}
"""Domain-randomisation ranges for FinFET, in nanometres."""


# ===========================================================================
# The frozen physics seam
# ===========================================================================


def apply_sem_chain(
    image: FloatArray,
    px_nm: float,
    params: dict[str, Any],
    rng: np.random.Generator,
) -> FloatArray:
    """Apply the SEM capture chain to one rendered image.

    Parameters
    ----------
    image
        Geometry-only image in ``[0, 1]``.
    px_nm
        Sampling pitch of this capture, in nanometres.
    params
        Chain parameters, e.g. ``{"noise_level": "medium"}``.
    rng
        Seeded generator.

    Returns
    -------
    FloatArray
        Image of the same shape, with capture physics applied.

    Notes
    -----
    **Frozen seam, owned by R2.** Contracted order once implemented:
    edge brightening, PSF blur, Poisson shot noise, read noise, scan artifacts,
    applied independently per capture -- reference and search are two separate
    physical acquisitions and must not share a noise draw.

    Currently an identity passthrough, so the pipeline runs end to end before
    ``src.sem_physics`` exists. Replacing this body requires no change here.
    """
    del px_nm, params, rng  # consumed by R2's implementation, not by the stub
    return image.copy()


# ===========================================================================
# Records
# ===========================================================================


@dataclass(frozen=True, slots=True)
class PairRecord:
    """One line of ``ground_truth.jsonl``.

    Attributes
    ----------
    pair_id
        Unique identifier, also the image filename stem.
    reference_path
        Reference image path, relative to the dataset root.
    search_path
        Search image path, relative to the dataset root.
    ground_truth
        The answer for this pair.
    plan
        Layout-space geometry, retained so the record is self-auditing.
    layout
        The die region this pair was cut from.
    strata
        Stratification labels.
    seed
        Seed that reproduces this pair exactly.
    anchors_in_reference
        How many anchors fall inside the reference field of view.
    """

    pair_id: str
    reference_path: str
    search_path: str
    ground_truth: GroundTruth
    plan: PairPlan
    layout: Layout
    strata: dict[str, str]
    seed: int
    anchors_in_reference: int

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serialisable form written to the JSONL file.

        Returns
        -------
        dict
            One ground-truth record.
        """
        return {
            "pair_id": self.pair_id,
            "reference_path": self.reference_path,
            "search_path": self.search_path,
            "ground_truth": self.ground_truth.to_dict(),
            "crop_centre_nm": list(self.plan.crop_centre_nm),
            "search_centre_nm": list(self.plan.search_centre_nm),
            "strata": dict(self.strata),
            "layout_params": self.layout.pattern.to_dict(),
            "anchors_gt": [a.to_dict() for a in self.layout.anchors],
            "anchors_in_reference": self.anchors_in_reference,
            "physics_params": {
                "noise_level": self.strata["noise_level"],
                "note": "stub -- R2 pending",
            },
            "seed": self.seed,
        }


def count_anchors_in_reference(record_layout: Layout, plan: PairPlan, out_size: int) -> int:
    """Count anchors falling inside the reference field of view.

    Parameters
    ----------
    record_layout
        The die region this pair was cut from.
    plan
        Geometry of the pair.
    out_size
        Edge length of the reference image, in pixels.

    Returns
    -------
    int
        Number of anchors inside the crop.
    """
    cx, cy = plan.crop_centre_nm
    half = out_size * config.REF_PX_NM / 2.0
    return sum(
        1 for a in record_layout.anchors if abs(a.x_nm - cx) <= half and abs(a.y_nm - cy) <= half
    )


def validate_record(record: PairRecord, out_size: int = OUT_SIZE) -> None:
    """Assert every invariant a published record must satisfy.

    Parameters
    ----------
    record
        Record about to be written.
    out_size
        Edge length of the search image, in pixels.

    Raises
    ------
    ValueError
        If the ground truth is out of bounds, the reference region would be
        clipped, or an ``anchored`` record has no anchor inside the reference
        field of view.

    Notes
    -----
    Each check corresponds to a defect that reached a published dataset once.
    They are fatal rather than warnings because a silently wrong ground truth
    caps every downstream role's accuracy and looks like an algorithm fault.
    """
    gt = record.ground_truth
    if not (0.0 <= gt.x < out_size and 0.0 <= gt.y < out_size):
        msg = f"{record.pair_id}: ground truth ({gt.x:.3f}, {gt.y:.3f}) outside the search image"
        raise ValueError(msg)

    footprint_px = (out_size * config.REF_PX_NM) / (config.REF_PX_NM * gt.scale)
    half_diag = footprint_px * float(np.sqrt(2)) / 2.0
    within = half_diag <= gt.x <= out_size - half_diag and half_diag <= gt.y <= out_size - half_diag
    if not within:
        msg = f"{record.pair_id}: reference region clipped at a search-image edge"
        raise ValueError(msg)

    if record.strata["anchor"] == "anchored" and record.anchors_in_reference == 0:
        msg = f"{record.pair_id}: labelled 'anchored' but no anchor lies in the reference crop"
        raise ValueError(msg)
    if record.strata["anchor"] == "unanchored" and record.layout.anchors:
        msg = f"{record.pair_id}: labelled 'unanchored' but carries anchors"
        raise ValueError(msg)


# ===========================================================================
# Generation
# ===========================================================================


def _sample_layout(
    architecture: str,
    rng: np.random.Generator,
    *,
    anchored: bool,
    anchor_centre_nm: tuple[float, float],
) -> Layout:
    """Draw one randomised layout of the requested architecture.

    Parameters
    ----------
    architecture
        ``"dram"`` or ``"finfet"``.
    rng
        Seeded generator.
    anchored
        Whether to place anchors.
    anchor_centre_nm
        Crop centre to place anchors around.

    Returns
    -------
    Layout
        Randomised die region.
    """
    if architecture == "dram":
        variant: Literal["orthogonal", "staggered"] = (
            "staggered" if rng.random() < 0.5 else "orthogonal"
        )
        return generate_dram_layout(
            EXTENT_NM,
            float(rng.uniform(*DRAM_RANGES["pitch_nm"])),
            float(rng.uniform(*DRAM_RANGES["line_width_nm"])),
            float(rng.uniform(*DRAM_RANGES["via_nm"])),
            rng,
            variant=variant,
            anchored=anchored,
            anchor_centre_nm=anchor_centre_nm,
        )
    return generate_finfet_layout(
        EXTENT_NM,
        float(rng.uniform(*FINFET_RANGES["fin_pitch_nm"])),
        float(rng.uniform(*FINFET_RANGES["fin_width_nm"])),
        float(rng.uniform(*FINFET_RANGES["gate_width_nm"])),
        rng,
        gate_pitch_nm=float(rng.uniform(*FINFET_RANGES["gate_pitch_nm"])),
        anchored=anchored,
        anchor_centre_nm=anchor_centre_nm,
    )


def _save_png(image: FloatArray, path: Path) -> None:
    """Write a ``[0, 1]`` image to disk as 8-bit grayscale PNG.

    Parameters
    ----------
    image
        Image in ``[0, 1]``.
    path
        Destination file.
    """
    as_bytes = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
    Image.fromarray(as_bytes).save(path)


def build_dataset(
    output_dir: Path,
    *,
    seeds_per_cell: int = 9,
    base_seed: int = DEFAULT_SEED,
    out_size: int = OUT_SIZE,
    supersample: int = DEFAULT_SUPERSAMPLE,
) -> tuple[list[PairRecord], float, Path]:
    """Generate the stratified dataset and write ``ground_truth.jsonl``.

    Parameters
    ----------
    output_dir
        Dataset root. ``reference/`` and ``search/`` are created inside it.
    seeds_per_cell
        Pairs per stratification cell. Cells are
        ``architecture x anchor x pose x noise``, so the total is
        ``2 * 2 * 3 * len(NOISE_LEVELS) * seeds_per_cell``.
    base_seed
        Base seed; pair ``i`` uses ``base_seed + i``.
    out_size
        Edge length of both images, in pixels.
    supersample
        Anti-aliasing factor per axis.

    Returns
    -------
    tuple
        ``(records, elapsed_seconds, ground_truth_path)``.
    """
    # Clear before writing. Leaving old files behind silently mixes runs: a
    # smaller --seeds-per-cell overwrites the pairs it regenerates and orphans
    # the rest, so the folder ends up holding two datasets at once and its
    # checksum matches neither.
    if output_dir.exists():
        for stale in sorted(output_dir.rglob("*.png")):
            stale.unlink()
        stale_gt = output_dir / "ground_truth.jsonl"
        if stale_gt.exists():
            stale_gt.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "reference").mkdir(exist_ok=True)
    (output_dir / "search").mkdir(exist_ok=True)

    records: list[PairRecord] = []
    started = time.perf_counter()
    index = 0

    for architecture in ("dram", "finfet"):
        for anchored in (True, False):
            for pose in ("none", "small", "large"):
                rotation_range, scale_range = POSE_RANGES[pose]
                for noise_level in NOISE_LEVELS:
                    for _ in range(seeds_per_cell):
                        seed = base_seed + index
                        rng = np.random.default_rng(seed)
                        rotation = float(rng.uniform(*rotation_range))
                        scale_mismatch = float(rng.uniform(*scale_range))

                        plan = plan_pair(
                            rng,
                            extent_nm=EXTENT_NM,
                            out_size=out_size,
                            rotation_deg=rotation,
                            scale_mismatch=scale_mismatch,
                        )
                        layout = _sample_layout(
                            architecture,
                            rng,
                            anchored=anchored,
                            anchor_centre_nm=plan.crop_centre_nm,
                        )
                        reference, search = render_pair(
                            layout, plan, out_size=out_size, supersample=supersample
                        )

                        reference = apply_sem_chain(
                            reference, config.REF_PX_NM, {"noise_level": noise_level}, rng
                        )
                        search = apply_sem_chain(
                            search, plan.search_px_nm, {"noise_level": noise_level}, rng
                        )

                        tag = "anchored" if anchored else "unanchored"
                        pair_id = f"{architecture}_{tag}_pose-{pose}_{index:04d}"
                        reference_path = Path("reference") / f"{pair_id}.png"
                        search_path = Path("search") / f"{pair_id}.png"
                        _save_png(reference, output_dir / reference_path)
                        _save_png(search, output_dir / search_path)

                        record = PairRecord(
                            pair_id=pair_id,
                            reference_path=reference_path.as_posix(),
                            search_path=search_path.as_posix(),
                            ground_truth=plan.ground_truth,
                            plan=plan,
                            layout=layout,
                            strata={
                                "architecture": architecture,
                                "anchor": tag,
                                "noise_level": noise_level,
                                "pose_condition": pose,
                            },
                            seed=seed,
                            anchors_in_reference=count_anchors_in_reference(layout, plan, out_size),
                        )
                        validate_record(record, out_size)
                        records.append(record)
                        index += 1

    gt_path = output_dir / "ground_truth.jsonl"
    with gt_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict()) + "\n")

    return records, time.perf_counter() - started, gt_path


def main(argv: Sequence[str] | None = None) -> int:
    """Generate the dataset from the command line.

    Parameters
    ----------
    argv
        Argument list; defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit status; ``0`` on success.
    """
    parser = argparse.ArgumentParser(
        prog="drift-generate-dataset",
        description="Generate the Drift-Sense stratified synthetic dataset.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--seeds-per-cell", type=int, default=9)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--supersample", type=int, default=DEFAULT_SUPERSAMPLE)
    args = parser.parse_args(argv)

    records, elapsed, gt_path = build_dataset(
        args.output_dir,
        seeds_per_cell=args.seeds_per_cell,
        base_seed=args.seed,
        supersample=args.supersample,
    )

    print(f"{len(records)} pairs in {elapsed:.1f}s -> {gt_path}")
    for axis in ("architecture", "anchor", "noise_level", "pose_condition"):
        counts = Counter(r.strata[axis] for r in records)
        print(f"  {axis}: {dict(counts)}")
    anchored = [r for r in records if r.strata["anchor"] == "anchored"]
    hits = sum(1 for r in anchored if r.anchors_in_reference > 0)
    print(f"  anchored references containing an anchor: {hits}/{len(anchored)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
