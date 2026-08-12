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
import hashlib
import json
import subprocess
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from PIL import Image

from src import config
from src.layouts import Layout, generate_dram_layout, generate_finfet_layout
from src.render import (
    DEFAULT_SUPERSAMPLE,
    GroundTruth,
    PairPlan,
    plan_pair,
    reconstruct_from_gt,
    render_pair,
)
from src.sem_physics import apply_sem_chain
from src.types import FloatArray

__all__ = [
    "MANIFEST_NAME",
    "PairRecord",
    "build_dataset",
    "OverlayResult",
    "OverlaySummary",
    "image_tree_hash",
    "overlay_check",
    "summarise_overlay",
    "main",
    "validate_record",
    "verify_dataset",
]

MANIFEST_NAME: str = "dataset_manifest.json"
"""Filename of the manifest written alongside the images."""

MANIFEST_SCHEMA_VERSION: int = 2
"""Bumped when the manifest gains or loses a field."""

OVERLAY_BIAS_TOLERANCE_PX: float = 0.2
"""Largest *mean* preferred offset the overlay check tolerates.

The mean, not the worst pair. A ground-truth defect is systematic -- the +0.5 px
base error an earlier revision shipped would have pushed every pair the same way
-- whereas probing at 0.25 px granularity on an image rebuilt from a 10x coarser
capture is noisy, and individual pairs land on neighbouring offsets by chance.
Judging the worst pair fails on correct data; judging the mean asks the question
that actually separates a defect from noise.

Set from the measured noise floor rather than picked. On a known-good dataset
the mean lands within about 0.10 px of zero (12 pairs, 0.25 px granularity, so a
standard error near 0.08); a planted 0.5 px offset reads back as roughly -0.46.
0.2 sits cleanly between them.

The earlier note here -- that the mean ``dy`` ran consistently negative
(-0.08 to -0.10) and deserved a larger sample -- has been resolved. It was
sampling noise. Once the noise strata were wired in, a 12-pair check read
``dy = -0.250`` and failed this tolerance; the same dataset at 48 pairs reads
``(-0.016, +0.036)``. The 12-pair failure was driven by two pairs pinned at the
-1.00 px probe limit, and the pinned pairs across a larger sample split both
ways in sign, which a real convention error cannot do.

Hence the sample-size rule the CLI now enforces: 12 pairs gives a standard
error near 0.12 px, so a 0.2 px tolerance sits inside 2 SE and the gate cannot
distinguish a defect from scatter. Use 48 or more when the answer has to mean
something. Rail-pinned pairs concentrate in FinFET, whose finer pitch aliases
harder against the 10x coarser search capture and flattens the error surface.
"""

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

NOISE_LEVELS: tuple[str, ...] = ("low", "medium", "high")
"""Noise strata generated.

Wired to ``src.sem_physics.apply_sem_chain`` now that R2's module has landed.
``"none"`` is deliberately excluded here -- in R2's module it is a
zero-noise-stage escape hatch for callers, not a severity level meant to sit
in the per-stratum eval table alongside low/medium/high. Scaling itself is
owned by ``src.sem_physics.NOISE_SCALING``; R1 no longer keeps a parallel copy
of those multipliers.
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

ANCHOR_SPAN_FRACTION: float = 0.34
"""Half-width of the anchor placement box, as a fraction of the reference FOV.

Derived rather than fixed: a constant in nanometres silently assumes a 1000 px
reference, and any other ``out_size`` then places anchors outside the crop and
fails ``validate_record``.
"""

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
#
# apply_sem_chain is imported from src.sem_physics (R2's module, owned by
# them). Contracted order: edge brightening, PSF blur, Poisson shot noise,
# read noise, scan artifacts, applied independently per capture -- reference
# and search are two separate physical acquisitions and must not share a
# noise draw. The call sites below pass "reference"/"search" as the preset so
# each capture gets its own degradation profile.


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
    anchor_half_span_nm: float = 340.0,
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
    anchor_half_span_nm
        Half-width of the box anchors are drawn from. Scale this with the
        reference field of view, not with the die extent.

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
            anchor_half_span_nm=anchor_half_span_nm,
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
        anchor_half_span_nm=anchor_half_span_nm,
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

    anchor_span_nm = out_size * config.REF_PX_NM * ANCHOR_SPAN_FRACTION
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
                            anchor_half_span_nm=anchor_span_nm,
                        )
                        reference, search = render_pair(
                            layout, plan, out_size=out_size, supersample=supersample
                        )

                        reference = apply_sem_chain(
                            reference,
                            config.REF_PX_NM,
                            {"preset": "reference", "noise_level": noise_level},
                            rng,
                        )
                        search = apply_sem_chain(
                            search,
                            plan.search_px_nm,
                            {"preset": "search", "noise_level": noise_level},
                            rng,
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

    expected = expected_pair_count(seeds_per_cell)
    if len(records) != expected:
        msg = f"generated {len(records)} pairs but the stratification demands {expected}"
        raise ValueError(msg)

    gt_path = output_dir / "ground_truth.jsonl"
    with gt_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict()) + "\n")

    write_manifest(
        output_dir,
        records,
        seeds_per_cell=seeds_per_cell,
        base_seed=base_seed,
        out_size=out_size,
        supersample=supersample,
    )
    return records, time.perf_counter() - started, gt_path


# ===========================================================================
# Manifest
# ===========================================================================


def image_tree_hash(output_dir: Path) -> str:
    """Return one fingerprint over the *pixel content* of every PNG.

    Parameters
    ----------
    output_dir
        Dataset root.

    Returns
    -------
    str
        Hex SHA-256 over ``"<pixel digest>  <shape>  <relative path>"`` lines,
        sorted by path.

    Notes
    -----
    Decoded pixels, not file bytes. Two machines can encode identical images
    into different PNG files -- compression level, zlib build, Pillow version --
    and a byte-level hash then reports a difference that does not exist. That is
    not hypothetical: it made a Linux and a macOS checkout of the same commit
    look like different datasets.

    Paths are relative to ``output_dir``, so the value does not depend on where
    the dataset lives or what ``--output-dir`` was called.
    """
    digest = hashlib.sha256()
    for path in sorted(output_dir.rglob("*.png")):
        relative = path.relative_to(output_dir).as_posix()
        with Image.open(path) as handle:
            pixels = np.asarray(handle)
        pixel_digest = hashlib.sha256(pixels.tobytes()).hexdigest()
        digest.update(f"{pixel_digest}  {pixels.shape}{pixels.dtype}  {relative}\n".encode())
    return digest.hexdigest()


def file_tree_hash(output_dir: Path) -> str:
    """Return one fingerprint over the encoded PNG bytes.

    Parameters
    ----------
    output_dir
        Dataset root.

    Returns
    -------
    str
        Hex SHA-256 over ``"<file digest>  <relative path>"`` lines.

    Notes
    -----
    Recorded alongside :func:`image_tree_hash` purely so a mismatch is
    diagnosable. Same pixels with different file hashes means the encoders
    differ and the data is fine; different pixel hashes means the data really
    does differ.
    """
    digest = hashlib.sha256()
    for path in sorted(output_dir.rglob("*.png")):
        relative = path.relative_to(output_dir).as_posix()
        file_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        digest.update(f"{file_digest}  {relative}\n".encode())
    return digest.hexdigest()


def _generator_commit() -> str:
    """Return the current git commit, or ``"unknown"`` outside a checkout.

    Returns
    -------
    str
        Short commit hash, suffixed ``-dirty`` when the tree has local edits.

    Notes
    -----
    This is what ties a ``results.csv`` back to the code that produced the data
    it was measured on. Without it, a number in a report is unattributable once
    anyone regenerates.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return f"{commit}-dirty" if dirty else commit


def expected_pair_count(seeds_per_cell: int) -> int:
    """Return how many pairs a full run must produce.

    Parameters
    ----------
    seeds_per_cell
        Pairs per stratification cell.

    Returns
    -------
    int
        ``architecture x anchor x pose x noise x seeds_per_cell``.
    """
    return 2 * 2 * 3 * len(NOISE_LEVELS) * seeds_per_cell


def write_manifest(
    output_dir: Path,
    records: list[PairRecord],
    *,
    seeds_per_cell: int,
    base_seed: int,
    out_size: int,
    supersample: int,
) -> Path:
    """Write ``dataset_manifest.json`` describing this dataset.

    Parameters
    ----------
    output_dir
        Dataset root.
    records
        Records just generated.
    seeds_per_cell
        Pairs per stratification cell.
    base_seed
        Base seed used.
    out_size
        Image edge length in pixels.
    supersample
        Anti-aliasing factor used.

    Returns
    -------
    Path
        The manifest that was written.

    Notes
    -----
    Since ``dataset/`` is gitignored, this file is the only thing tying a set of
    results back to the data and the code that produced them. It also makes
    truncation loud: a run killed partway writes a pair count that disagrees
    with what the stratification demands, and :func:`verify_dataset` fails on it
    instead of everyone comparing checksums by hand.
    """
    counts = {
        axis: dict(Counter(r.strata[axis] for r in records))
        for axis in ("architecture", "anchor", "noise_level", "pose_condition")
    }
    anchored = [r for r in records if r.strata["anchor"] == "anchored"]
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "pair_count": len(records),
        "expected_pair_count": expected_pair_count(seeds_per_cell),
        "image_count": len(records) * 2,
        "image_tree_sha256": image_tree_hash(output_dir),
        "file_tree_sha256": file_tree_hash(output_dir),
        "generator_commit": _generator_commit(),
        "generation": {
            "seeds_per_cell": seeds_per_cell,
            "seed": base_seed,
            "out_size": out_size,
            "supersample": supersample,
            "extent_nm": EXTENT_NM,
            "noise_levels": list(NOISE_LEVELS),
        },
        "strata_counts": counts,
        "anchored_references_with_anchor": sum(1 for r in anchored if r.anchors_in_reference > 0),
        "layout_ranges_nm": {"dram": DRAM_RANGES, "finfet": FINFET_RANGES},
        "pose_ranges": {k: [list(v[0]), list(v[1])] for k, v in POSE_RANGES.items()},
    }
    path = output_dir / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def verify_dataset(output_dir: Path) -> dict[str, Any]:
    """Re-check a dataset on disk against its manifest.

    Parameters
    ----------
    output_dir
        Dataset root, containing a manifest.

    Returns
    -------
    dict
        The manifest, when everything agrees.

    Raises
    ------
    FileNotFoundError
        If no manifest is present.
    ValueError
        If the pair count is short of what the stratification demands, if the
        image count disagrees, or if the image-tree hash has drifted.
    """
    path = output_dir / MANIFEST_NAME
    if not path.exists():
        msg = f"no {MANIFEST_NAME} in {output_dir} -- regenerate the dataset"
        raise FileNotFoundError(msg)
    manifest: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))

    if manifest["pair_count"] != manifest["expected_pair_count"]:
        msg = (
            f"truncated dataset: {manifest['pair_count']} pairs on record but the "
            f"stratification demands {manifest['expected_pair_count']}"
        )
        raise ValueError(msg)

    on_disk = len(list(output_dir.rglob("*.png")))
    if on_disk != manifest["image_count"]:
        msg = (
            f"image count mismatch: {on_disk} PNGs on disk, manifest says {manifest['image_count']}"
        )
        raise ValueError(msg)

    actual = image_tree_hash(output_dir)
    if actual != manifest["image_tree_sha256"]:
        msg = (
            f"image tree has drifted: {actual} on disk, manifest says "
            f"{manifest['image_tree_sha256']}"
        )
        raise ValueError(msg)
    return manifest


def compare_manifests(mine: dict[str, Any], theirs: dict[str, Any]) -> str:
    """Explain how two datasets differ, in one line.

    Parameters
    ----------
    mine
        A manifest.
    theirs
        Another manifest to compare against.

    Returns
    -------
    str
        Human-readable verdict.

    Notes
    -----
    Separating the pixel hash from the file hash turns "our hashes differ" from
    an unanswerable question into a diagnosis: same pixels and different files
    is a harmless encoder difference, different pixels is a real one.
    """
    if mine["pair_count"] != theirs["pair_count"]:
        return f"different sizes: {mine['pair_count']} vs {theirs['pair_count']} pairs"
    if mine["generator_commit"] != theirs["generator_commit"]:
        return f"different code: {mine['generator_commit']} vs {theirs['generator_commit']}"
    if mine["image_tree_sha256"] != theirs["image_tree_sha256"]:
        return "PIXELS DIFFER -- same code and size, so this is a real divergence"
    if mine.get("file_tree_sha256") != theirs.get("file_tree_sha256"):
        return "identical pixels, different PNG encoding -- harmless, the data matches"
    return "identical"


# ===========================================================================
# Overlay verification
# ===========================================================================


@dataclass(frozen=True, slots=True)
class OverlayResult:
    """Outcome of rebuilding one reference from its published ground truth.

    Attributes
    ----------
    pair_id
        Which pair this is.
    mean_abs_error
        Mean absolute intensity difference in ``[0, 1]``.
    dx
        Residual horizontal shift in reference pixels.
    dy
        Residual vertical shift in reference pixels.
    """

    pair_id: str
    mean_abs_error: float
    dx: float
    dy: float


PROBE_OFFSETS_PX: tuple[float, ...] = (-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0)
"""Offsets probed either side of the published ground truth, in reference px."""


def _best_offset(
    reference: FloatArray,
    search: FloatArray,
    truth: GroundTruth,
    *,
    out_size: int,
) -> tuple[float, float]:
    """Find the offset from the published ground truth that fits the data best.

    Parameters
    ----------
    reference
        The published reference image.
    search
        The published search image.
    truth
        The published ground truth for this pair.
    out_size
        Image edge length in pixels.

    Returns
    -------
    tuple of float
        ``(dx, dy)`` in reference pixels. ``(0, 0)`` means the published answer
        explains the data better than any nearby alternative.

    Notes
    -----
    Phase correlation is the obvious tool and the wrong one here. The rebuilt
    image is a 10x upsample of the search capture, so it carries almost no
    high-frequency content, and phase correlation's whitening then amplifies
    noise into a broad ambiguous peak -- it reported hundreds of pixels of drift
    on data that is provably correct, because a periodic lattice puts an equally
    strong peak at every lattice vector.

    Asking directly is both cheaper and harder to fool: rebuild at a few offsets
    around the published position and see which fits best. Axis-separable, so it
    costs 2n rebuilds rather than n squared.
    """

    def error_at(dx: float, dy: float) -> float:
        shifted = GroundTruth(
            x=truth.x + dx / truth.scale,
            y=truth.y + dy / truth.scale,
            rotation_deg=truth.rotation_deg,
            scale=truth.scale,
        )
        rebuilt = reconstruct_from_gt(search, shifted, out_size=out_size)
        return float(np.abs(reference - rebuilt).mean())

    best_dx = min(PROBE_OFFSETS_PX, key=lambda d: error_at(d, 0.0))
    best_dy = min(PROBE_OFFSETS_PX, key=lambda d: error_at(0.0, d))
    return best_dx, best_dy


def overlay_check(
    output_dir: Path,
    *,
    sample: int = 8,
    seed: int = 0,
    out_size: int = OUT_SIZE,
) -> list[OverlayResult]:
    """Rebuild sampled references from published artefacts alone and compare.

    Parameters
    ----------
    output_dir
        Dataset root.
    sample
        How many pairs to check. ``0`` checks every pair.
    seed
        Seed for choosing the sample.
    out_size
        Image edge length in pixels.

    Returns
    -------
    list of OverlayResult
        One result per checked pair.

    Notes
    -----
    This is the strongest check available on the coordinate convention, because
    it touches **only** the search image and the ground-truth record -- never the
    layout generator. A convention that is merely self-consistent passes every
    internal check and fails this one, which is exactly how a constant +0.5 px
    error survived in an earlier revision.

    Worth re-running after any change to the randomisation ranges: wider ranges
    reach parameter combinations the previous dataset never covered.
    """
    records = [
        json.loads(line)
        for line in (output_dir / "ground_truth.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rng = np.random.default_rng(seed)
    if sample <= 0:
        chosen = records
    else:
        picked = rng.choice(len(records), size=min(sample, len(records)), replace=False)
        chosen = [records[int(i)] for i in picked]

    results: list[OverlayResult] = []
    for record in chosen:
        reference = _load_png(output_dir / record["reference_path"])
        search = _load_png(output_dir / record["search_path"])
        truth = GroundTruth(**record["ground_truth"])
        rebuilt = reconstruct_from_gt(search, truth, out_size=out_size)
        dx, dy = _best_offset(reference, search, truth, out_size=out_size)
        results.append(
            OverlayResult(
                pair_id=record["pair_id"],
                mean_abs_error=float(np.abs(reference - rebuilt).mean()),
                dx=dx,
                dy=dy,
            )
        )
    return results


@dataclass(frozen=True, slots=True)
class OverlaySummary:
    """The verdict of an overlay check, and the numbers behind it.

    Attributes
    ----------
    n
        Pairs checked.
    measured, pinned
        Pairs that found an interior minimum, and pairs that ran to the probe
        limit. ``measured + pinned == n``.
    mean_dx, mean_dy
        Mean preferred offset over the measured pairs, in reference pixels.
    se_dx, se_dy
        Standard error of those means.
    bias
        The larger absolute mean, i.e. the number judged against the tolerance.
    verdict
        ``"ok"``, ``"failed"``, or ``"inconclusive"``.
    detail
        One line explaining the verdict.
    """

    n: int
    measured: int
    pinned: int
    mean_dx: float
    mean_dy: float
    se_dx: float
    se_dy: float
    bias: float
    verdict: Literal["ok", "failed", "inconclusive"]
    detail: str


def summarise_overlay(
    results: Sequence[OverlayResult],
    *,
    tolerance_px: float = OVERLAY_BIAS_TOLERANCE_PX,
) -> OverlaySummary:
    """Turn per-pair overlay results into a pass/fail verdict on the ground truth.

    Parameters
    ----------
    results
        Per-pair results from :func:`overlay_check`.
    tolerance_px
        Largest mean offset accepted as unbiased.

    Returns
    -------
    OverlaySummary
        The verdict and the statistics supporting it.

    Notes
    -----
    Two corrections separate this from a plain mean, both learned from a false
    failure after the noise strata landed:

    **Censoring.** :func:`_best_offset` returns an argmin over a bounded probe
    set. A result at the end of that range has not located a minimum, it has run
    out of room -- the true value is "at most -1 px", not "exactly -1 px".
    Averaging such a value as if it were exact pulls the mean toward the rail.
    Periodic lattices produce these (a lattice puts an equal peak at every
    lattice vector) and FinFET produces more of them than DRAM, its finer pitch
    aliasing harder against the 10x coarser search capture. They are excluded
    from the mean and reported separately.

    **Sampling noise.** The probe is quantised to 0.25 px, so a small sample
    scatters widely: at 12 pairs the standard error is near 0.12 px against a
    0.2 px tolerance, and the gate cannot then tell a defect from chance. A bias
    must therefore exceed both the tolerance and twice its own standard error.
    A real convention error clears both -- a planted 0.5 px offset reads back at
    roughly five standard errors even at 12 pairs -- while an underpowered
    sample that happens to lean returns ``"inconclusive"``, which asks for more
    pairs rather than either passing or crying wolf.

    Whole-sample rail pinning is the one case where censored values still fail:
    a ground truth wrong by more than the probe range drives every pair to the
    same rail, which is a defect, not ambiguity.
    """
    n = len(results)
    if n == 0:
        msg = "overlay summary needs at least one result"
        raise ValueError(msg)

    rail = max(PROBE_OFFSETS_PX)
    pinned = [r for r in results if abs(r.dx) >= rail or abs(r.dy) >= rail]
    measured = [r for r in results if abs(r.dx) < rail and abs(r.dy) < rail]

    # Every pair driven to the same rail is a real offset beyond the probe
    # range, not the ambiguity that censoring is meant to forgive.
    if len(pinned) > n / 2:
        for axis, label in ((0, "dx"), (1, "dy")):
            rails = [(r.dx, r.dy)[axis] for r in pinned]
            leaning = [v for v in rails if abs(v) >= rail]
            if leaning and abs(sum(np.sign(leaning))) == len(leaning):
                return OverlaySummary(
                    n=n,
                    measured=len(measured),
                    pinned=len(pinned),
                    mean_dx=float(np.mean([r.dx for r in results])),
                    mean_dy=float(np.mean([r.dy for r in results])),
                    se_dx=0.0,
                    se_dy=0.0,
                    bias=float(rail),
                    verdict="failed",
                    detail=(
                        f"{len(leaning)}/{n} pairs pinned to the same {label} rail -- the "
                        "published ground truth is offset by more than the probe range"
                    ),
                )

    if not measured:
        return OverlaySummary(
            n=n,
            measured=0,
            pinned=len(pinned),
            mean_dx=float("nan"),
            mean_dy=float("nan"),
            se_dx=float("nan"),
            se_dy=float("nan"),
            bias=float("nan"),
            verdict="failed",
            detail="every pair was ambiguous; the check cannot speak to the ground truth",
        )

    offsets = np.array([[r.dx, r.dy] for r in measured], dtype=float)
    mean_dx, mean_dy = (float(v) for v in offsets.mean(axis=0))
    bias = max(abs(mean_dx), abs(mean_dy))

    m = len(measured)
    if m > 1:
        se_dx, se_dy = (float(v) for v in offsets.std(axis=0, ddof=1) / np.sqrt(m))
    else:
        se_dx = se_dy = float("inf")
    se = se_dx if abs(mean_dx) >= abs(mean_dy) else se_dy

    if bias > tolerance_px and bias > 2.0 * se:
        verdict: Literal["ok", "failed", "inconclusive"] = "failed"
        detail = f"systematic bias of {bias:.3f} px -- the published ground truth is offset"
    elif bias > tolerance_px:
        verdict = "inconclusive"
        detail = (
            f"bias {bias:.3f} px exceeds the {tolerance_px} px tolerance but is within two "
            f"standard errors ({2.0 * se:.3f} px); re-run with more than {n} pairs"
        )
    else:
        verdict = "ok"
        detail = "no systematic offset; the published ground truth fits best"

    return OverlaySummary(
        n=n,
        measured=m,
        pinned=len(pinned),
        mean_dx=mean_dx,
        mean_dy=mean_dy,
        se_dx=se_dx,
        se_dy=se_dy,
        bias=bias,
        verdict=verdict,
        detail=detail,
    )


def _load_png(path: Path) -> FloatArray:
    """Read an 8-bit PNG as a ``[0, 1]`` float32 image.

    Parameters
    ----------
    path
        Image file.

    Returns
    -------
    FloatArray
        Image in ``[0, 1]``.
    """
    with Image.open(path) as handle:
        pixels = np.asarray(handle, dtype=np.float32)
    return cast(FloatArray, pixels / np.float32(255.0))


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
    parser.add_argument(
        "--overlay-check",
        type=int,
        default=0,
        metavar="N",
        help="rebuild N sampled references from search image + ground truth and report drift",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="check an existing dataset against its manifest instead of generating",
    )
    args = parser.parse_args(argv)

    if args.overlay_check:
        results = overlay_check(args.output_dir, sample=args.overlay_check)
        for result in results:
            print(
                f"  {result.pair_id:44s} mean|err| {result.mean_abs_error:.4f}  "
                f"best offset ({result.dx:+.2f}, {result.dy:+.2f}) px"
            )
        summary = summarise_overlay(results)

        if summary.pinned:
            print(
                f"\n  {summary.pinned}/{summary.n} pairs pinned at the "
                f"+-{max(PROBE_OFFSETS_PX):.2f} px probe limit: ambiguous fit, "
                "excluded from the bias estimate"
            )
        if summary.measured:
            print(
                f"\nmean preferred offset: ({summary.mean_dx:+.3f}, {summary.mean_dy:+.3f}) px "
                f"over {summary.measured} unambiguous pairs"
            )
            print(f"  standard error:      (+-{summary.se_dx:.3f}, +-{summary.se_dy:.3f}) px")

        print(f"{summary.verdict.upper()}: {summary.detail}")
        return 0 if summary.verdict == "ok" else 1

    if args.verify:
        try:
            manifest = verify_dataset(args.output_dir)
        except (FileNotFoundError, ValueError) as exc:
            print(f"FAILED: {exc}")
            return 1
        print(f"OK: {manifest['pair_count']} pairs, tree {manifest['image_tree_sha256'][:16]}...")
        print(f"    generated by commit {manifest['generator_commit']}")
        return 0

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

    manifest = json.loads((args.output_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    print(f"  image tree sha256: {manifest['image_tree_sha256']}")
    print(f"  generator commit:  {manifest['generator_commit']}")
    print(f"  manifest:          {args.output_dir / MANIFEST_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
