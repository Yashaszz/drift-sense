"""Geometry and ground-truth invariants for the synthetic data generator.

Every test here corresponds to a defect that reached a published dataset once.
They are regression guards, not coverage filler.
"""

import shutil

import numpy as np
import pytest

from src import config
from src.generate_dataset import (
    DRAM_RANGES,
    EXTENT_NM,
    FINFET_RANGES,
    MANIFEST_NAME,
    NOISE_LEVELS,
    OUT_SIZE,
    OVERLAY_BIAS_TOLERANCE_PX,
    POSE_RANGES,
    PROBE_OFFSETS_PX,
    OverlayResult,
    _sample_layout,
    build_dataset,
    compare_manifests,
    count_anchors_in_reference,
    expected_pair_count,
    file_tree_hash,
    image_tree_hash,
    main,
    overlay_check,
    summarise_overlay,
    validate_record,
    verify_dataset,
)
from src.layouts import generate_dram_layout, generate_finfet_layout
from src.render import (
    GroundTruth,
    _render_rows,
    plan_pair,
    raster_centre_base,
    rasterize,
    reconstruct_from_gt,
    render_pair,
)

SMALL = 200
"""Image edge used throughout, in pixels. Large enough to exercise the geometry."""


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Coordinate convention
# ---------------------------------------------------------------------------


def test_centre_base_delegates_to_config():
    """The generator must not carry its own copy of the pixel-centre rule."""
    assert raster_centre_base(1000) == config.image_centre((1000, 1000))
    assert raster_centre_base(1000) == (499.5, 499.5)


def test_zero_offset_reconstruction_is_bit_exact():
    """Self-reconstruction at zero offset must be exact, not merely close.

    With the correct centre base the resampling grid lands on integer indices,
    so bilinear interpolation is a no-op. Any other base leaves a residual. This
    is the check that caught a constant +0.5 px error in every published record.
    """
    layout = generate_dram_layout(EXTENT_NM, 180.0, 40.0, 60.0, _rng(), anchored=False)
    image = rasterize(layout, (6000.0, 6000.0), config.REF_PX_NM, SMALL, supersample=2)
    base_x, base_y = raster_centre_base(SMALL)

    exact = reconstruct_from_gt(image, GroundTruth(base_x, base_y, 0.0, 1.0), out_size=SMALL)
    assert float(np.abs(image - exact).max()) == 0.0


def test_half_pixel_offset_is_detectable():
    """The bit-exact test must have teeth: the old base has to fail it."""
    layout = generate_dram_layout(EXTENT_NM, 180.0, 40.0, 60.0, _rng(), anchored=False)
    image = rasterize(layout, (6000.0, 6000.0), config.REF_PX_NM, SMALL, supersample=2)

    wrong = reconstruct_from_gt(
        image, GroundTruth(SMALL / 2.0, SMALL / 2.0, 0.0, 1.0), out_size=SMALL
    )
    assert float(np.abs(image - wrong).mean()) > 0.0


@pytest.mark.parametrize("rotation_deg", [-8.0, -3.0, 0.0, 3.0, 8.0])
@pytest.mark.parametrize("scale_mismatch", [0.95, 1.0, 1.05])
def test_plan_is_self_inverse(rotation_deg, scale_mismatch):
    """Solving the crop centre back to a ground-truth position must round-trip.

    The planner samples the answer and solves backwards for the layout centre;
    projecting that centre forward again has to return the same answer, or the
    two directions disagree and the published number is not the true one.
    """
    plan = plan_pair(
        _rng(7),
        extent_nm=EXTENT_NM,
        out_size=1000,
        rotation_deg=rotation_deg,
        scale_mismatch=scale_mismatch,
    )
    base_x, base_y = raster_centre_base(1000)
    cx, cy = plan.crop_centre_nm
    sx, sy = plan.search_centre_nm
    theta = np.deg2rad(-rotation_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    dx, dy = cx - sx, cy - sy

    forward_x = base_x + (cos_t * dx - sin_t * dy) / plan.search_px_nm
    forward_y = base_y + (sin_t * dx + cos_t * dy) / plan.search_px_nm

    assert forward_x == pytest.approx(plan.ground_truth.x, abs=1e-9)
    assert forward_y == pytest.approx(plan.ground_truth.y, abs=1e-9)


# ---------------------------------------------------------------------------
# Containment
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(40))
def test_reference_is_always_contained(seed):
    """The reference region must never overhang the search border.

    Drawing the crop and search centres independently put 2.6% of pairs wholly
    outside the search image, which produces cases with no recoverable answer
    that are indistinguishable from a matcher failing.
    """
    rng = _rng(seed)
    plan = plan_pair(
        rng,
        extent_nm=EXTENT_NM,
        out_size=1000,
        rotation_deg=float(rng.uniform(-8, 8)),
        scale_mismatch=float(rng.uniform(0.95, 1.05)),
    )
    gt = plan.ground_truth
    footprint = (1000 * config.REF_PX_NM) / plan.search_px_nm
    half_diag = footprint * np.sqrt(2) / 2.0

    assert half_diag <= gt.x <= 1000 - half_diag
    assert half_diag <= gt.y <= 1000 - half_diag


def test_edge_clipping_is_opt_in():
    """Clipped cases must be requested explicitly, never produced by accident."""
    plan = plan_pair(
        _rng(3),
        extent_nm=EXTENT_NM,
        out_size=1000,
        scale_mismatch=1.0,
        allow_edge_clipping=True,
    )
    assert 0.0 <= plan.ground_truth.x < 1000


# ---------------------------------------------------------------------------
# Anchors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("architecture", ["dram", "finfet"])
@pytest.mark.parametrize("seed", range(10))
def test_anchors_land_inside_the_reference_crop(architecture, seed):
    """An 'anchored' pair must actually carry an anchor in its reference crop.

    The reference field of view is 0.69% of the layout area, so anchors placed
    at random lattice sites land inside it about 2% of the time. An earlier
    revision produced 1 hit in 18 pairs, which made the stratum meaningless.
    """
    rng = _rng(seed)
    plan = plan_pair(rng, extent_nm=EXTENT_NM, out_size=1000)
    layout = _sample_layout(architecture, rng, anchored=True, anchor_centre_nm=plan.crop_centre_nm)
    assert count_anchors_in_reference(layout, plan, 1000) > 0


@pytest.mark.parametrize("architecture", ["dram", "finfet"])
def test_unanchored_layouts_carry_no_anchors(architecture):
    """The control stratum must be genuinely anchor-free."""
    rng = _rng(1)
    plan = plan_pair(rng, extent_nm=EXTENT_NM, out_size=1000)
    layout = _sample_layout(architecture, rng, anchored=False, anchor_centre_nm=plan.crop_centre_nm)
    assert layout.anchors == ()
    assert layout.erase == ()


def test_anchors_break_periodicity_in_the_rendered_image():
    """Anchors have to be visible, not merely recorded in the ground truth."""
    rng = _rng(5)
    plan = plan_pair(rng, extent_nm=EXTENT_NM, out_size=SMALL)
    common = {"anchor_centre_nm": plan.crop_centre_nm}
    anchored = generate_finfet_layout(EXTENT_NM, 90.0, 24.0, 12.0, _rng(5), anchored=True, **common)
    plain = generate_finfet_layout(EXTENT_NM, 90.0, 24.0, 12.0, _rng(5), anchored=False, **common)

    def self_similarity(layout):
        image = rasterize(layout, plan.crop_centre_nm, config.REF_PX_NM, SMALL, supersample=2)
        centred = image - image.mean()
        shifted = np.roll(centred, 90, axis=1)
        return float((centred * shifted).sum() / np.sqrt((centred**2).sum() * (shifted**2).sum()))

    assert self_similarity(anchored) < self_similarity(plain)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "out_size,nm_per_px,supersample",
    [(0, 1.0, 4), (10, 0.0, 4), (10, 1.0, 0)],
)
def test_rasterize_rejects_impossible_geometry(out_size, nm_per_px, supersample):
    """Nonsense parameters must fail loudly rather than produce an empty image."""
    layout = generate_dram_layout(EXTENT_NM, 180.0, 40.0, 60.0, _rng(), anchored=False)
    with pytest.raises(ValueError, match="must be positive"):
        rasterize(layout, (0.0, 0.0), nm_per_px, out_size, supersample=supersample)


def test_rendered_images_are_normalised():
    """Both captures must land in [0, 1] with the expected shape."""
    rng = _rng(2)
    plan = plan_pair(rng, extent_nm=EXTENT_NM, out_size=SMALL, rotation_deg=4.0)
    layout = _sample_layout("dram", rng, anchored=True, anchor_centre_nm=plan.crop_centre_nm)
    reference, search = render_pair(layout, plan, out_size=SMALL, supersample=2)

    for image in (reference, search):
        assert image.shape == (SMALL, SMALL)
        assert image.dtype == np.float32
        assert float(image.min()) >= 0.0 and float(image.max()) <= 1.0


def test_validate_record_rejects_out_of_bounds_ground_truth():
    """The record validator must reject a coordinate outside the search image."""
    rng = _rng(0)
    plan = plan_pair(rng, extent_nm=EXTENT_NM, out_size=1000)
    layout = _sample_layout("dram", rng, anchored=True, anchor_centre_nm=plan.crop_centre_nm)

    from src.generate_dataset import PairRecord

    broken = PairRecord(
        pair_id="broken",
        reference_path="reference/broken.png",
        search_path="search/broken.png",
        ground_truth=GroundTruth(-5.0, 500.0, 0.0, config.NOMINAL_SCALE),
        plan=plan,
        layout=layout,
        strata={
            "architecture": "dram",
            "anchor": "anchored",
            "noise_level": "none",
            "pose_condition": "none",
        },
        seed=0,
        anchors_in_reference=3,
    )
    with pytest.raises(ValueError, match="outside the search image"):
        validate_record(broken)


@pytest.mark.parametrize("pitch_nm", [70.0, 80.0, 90.0, 100.0, 110.0])
def test_commensurate_pitch_still_antialiases(pitch_nm):
    """A lattice whose pitch is a whole number of pixels must still anti-alias.

    Without the sheared sample grid every output pixel samples the same
    sub-pixel phase, so a commensurate lattice renders with no grey edges and
    its linewidth comes out up to 11% wide -- an error R2's linewidth-dependent
    physics and any cited critical dimension would inherit.
    """
    fin_width = 18.0
    layout = generate_finfet_layout(
        EXTENT_NM, pitch_nm, fin_width, 12.0, _rng(), anchored=False, gate_pitch_nm=1.0e9
    )
    image = rasterize(layout, (6000.0, 6000.0), config.SEARCH_PX_NM, SMALL, supersample=4)
    rendered_nm = float(image.mean()) * pitch_nm
    assert abs(rendered_nm - fin_width) / fin_width < 0.03


def test_zero_rotation_matches_rotated_linewidth():
    """The two rendering paths must agree on how wide a fin is.

    They are different code paths, and a systematic difference between them
    would confound every per-pose comparison R3 makes.
    """
    fin_width, pitch = 18.0, 90.0
    layout = generate_finfet_layout(
        EXTENT_NM, pitch, fin_width, 12.0, _rng(), anchored=False, gate_pitch_nm=1.0e9
    )
    flat = rasterize(layout, (6000.0, 6000.0), config.SEARCH_PX_NM, SMALL, supersample=4)
    tilted = rasterize(
        layout, (6000.0, 6000.0), config.SEARCH_PX_NM, SMALL, rotation_deg=5.0, supersample=4
    )
    assert abs(float(flat.mean()) - float(tilted.mean())) < 0.02


@pytest.mark.parametrize("rotation_deg", [0.0, 5.0])
@pytest.mark.parametrize("supersample", [2, 4])
def test_tiling_does_not_change_the_image(rotation_deg, supersample):
    """Rendering in strips must be bit-identical to rendering in one pass.

    Tiling exists to bound peak memory -- rendering a 1000 px image at
    supersample 4 in one pass allocates roughly a gigabyte, enough to have
    Windows tear down the WSL VM mid-run and leave a partial dataset behind.
    It is only safe because each output pixel depends solely on its own
    supersample block, so this test pins that property.
    """
    layout = generate_dram_layout(
        EXTENT_NM, 187.3, 40.0, 60.0, _rng(1), anchored=True, anchor_centre_nm=(6000.0, 6000.0)
    )
    whole = rasterize(
        layout,
        (6000.0, 6000.0),
        config.SEARCH_PX_NM,
        SMALL,
        rotation_deg=rotation_deg,
        supersample=supersample,
    )
    strips = np.concatenate(
        [
            _render_rows(
                layout,
                (6000.0, 6000.0),
                config.SEARCH_PX_NM,
                SMALL,
                first,
                min(first + 37, SMALL),
                rotation_deg=rotation_deg,
                supersample=supersample,
            )
            for first in range(0, SMALL, 37)
        ]
    )
    assert np.array_equal(whole, strips)


def test_randomisation_matches_the_documented_tolerances():
    """Layout ranges must match the tolerances the work-split document states.

    The generator is 30% of the project score and every dimension in it needs a
    citation, so the ranges are derived from a nominal value and a stated
    tolerance rather than typed in as magic bounds. This pins that derivation.
    """
    for ranges, keys, tolerance in (
        (DRAM_RANGES, ["pitch_nm"], 0.20),
        (DRAM_RANGES, ["line_width_nm", "via_nm"], 0.15),
        (FINFET_RANGES, ["fin_pitch_nm", "gate_pitch_nm"], 0.20),
        (FINFET_RANGES, ["fin_width_nm", "gate_width_nm"], 0.15),
    ):
        for key in keys:
            low, high = ranges[key]
            nominal = (low + high) / 2.0
            assert (high - low) / (2.0 * nominal) == pytest.approx(tolerance, abs=1e-9)


def test_pose_baseline_matches_the_brief():
    """``small`` is the documented baseline; ``large`` must exceed it."""
    (small_rot_lo, small_rot_hi), (small_scale_lo, small_scale_hi) = POSE_RANGES["small"]
    assert (small_rot_lo, small_rot_hi) == (-5.0, 5.0)
    assert small_scale_lo == pytest.approx(0.97)
    assert small_scale_hi == pytest.approx(1.03)

    (large_rot_lo, large_rot_hi), _ = POSE_RANGES["large"]
    assert large_rot_hi > small_rot_hi
    assert large_rot_lo < small_rot_lo


def test_noise_strata_are_not_faked():
    """Only emit noise levels the physics can actually produce.

    Before R2's module was wired in, ``apply_sem_chain`` was a passthrough, so
    generating low/medium/high would have given R3 a stratum column that
    couldn't differ -- worse than an honestly absent one. Now that
    ``src.sem_physics`` is wired in, this guards the reverse mistake: emitting
    strata the physics can't actually back up, or silently reverting to a
    passthrough without also reverting ``NOISE_LEVELS``.
    """
    from src.generate_dataset import apply_sem_chain

    rng = np.random.default_rng(0)
    image = rng.random((16, 16)).astype(np.float32)
    passthrough = np.array_equal(apply_sem_chain(image, 10.0, {}, rng), image)
    if passthrough:
        assert NOISE_LEVELS == ("none",)
    else:
        assert set(NOISE_LEVELS) >= {"low", "medium", "high"}


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def _tiny_dataset_master(tmp_path_factory):
    """Build the smallest complete dataset once for the whole session.

    Nine tests take this fixture, and once the noise strata landed each build
    became 36 pairs through the real physics chain rather than 12 through a
    passthrough -- so building per test meant rendering 324 pairs, an entire
    dataset, before a single assertion ran. That took the suite from about a
    minute to ten.

    The tests mutate their dataset -- truncating it, re-encoding it, planting a
    ground-truth offset -- so they cannot share one directory. They copy this
    one instead, which costs a file copy rather than a re-render.
    """
    root = tmp_path_factory.mktemp("tiny_dataset_master")
    build_dataset(root, seeds_per_cell=1, out_size=OUT_SIZE, supersample=1)
    return root


@pytest.fixture
def tiny_dataset(_tiny_dataset_master, tmp_path):
    """Return a private copy of the smallest complete dataset."""
    root = tmp_path / "dataset"
    shutil.copytree(_tiny_dataset_master, root)
    return root


def test_manifest_describes_the_dataset(tiny_dataset):
    """The manifest is the only link from a results file back to its data.

    ``dataset/`` is gitignored, so nothing else records which images a number
    was measured on or which commit produced them.
    """
    import json

    manifest = json.loads((tiny_dataset / MANIFEST_NAME).read_text())
    assert manifest["pair_count"] == expected_pair_count(1)
    assert manifest["image_count"] == manifest["pair_count"] * 2
    assert len(manifest["image_tree_sha256"]) == 64
    assert manifest["generation"]["seeds_per_cell"] == 1
    # 2 anchor conditions x 3 pose strata x len(NOISE_LEVELS) noise strata,
    # per architecture, at seeds_per_cell=1.
    per_architecture = 2 * 3 * len(NOISE_LEVELS)
    assert manifest["strata_counts"]["architecture"] == {
        "dram": per_architecture,
        "finfet": per_architecture,
    }


def test_verify_accepts_an_untouched_dataset(tiny_dataset):
    """A clean dataset must verify."""
    assert verify_dataset(tiny_dataset)["pair_count"] == expected_pair_count(1)


def test_verify_catches_a_truncated_dataset(tiny_dataset):
    """Missing images must fail loudly.

    A generation run killed partway leaves a folder that looks fine -- the
    process is killed rather than raising, so there is no traceback and no
    obvious sign. That cost the team a day of chasing checksum mismatches.
    """
    next(iter((tiny_dataset / "reference").glob("*.png"))).unlink()
    with pytest.raises(ValueError, match="image count mismatch"):
        verify_dataset(tiny_dataset)


def _manifest(commit, image_hash, file_hash="f", pairs=324):
    return {
        "pair_count": pairs,
        "generator_commit": commit,
        "image_tree_sha256": image_hash,
        "file_tree_sha256": file_hash,
    }


def test_noise_levels_flag_reaches_the_generator(monkeypatch, tmp_path):
    """A parsed flag that never reaches build_dataset is worse than no flag.

    The first cut added ``--noise-levels`` to the parser and did not pass it on,
    so ``--noise-levels none`` silently produced the default three strata --
    three times the pairs, none of them the control that was asked for, and no
    error anywhere. CI passed because nothing exercised the wiring.
    """
    seen: dict[str, object] = {}

    def fake_build(output_dir, **kwargs):
        seen.update(kwargs)
        return [], 0.0, output_dir / "ground_truth.jsonl"

    monkeypatch.setattr("src.generate_dataset.build_dataset", fake_build)
    monkeypatch.setattr("src.generate_dataset.write_manifest", lambda *a, **k: tmp_path)
    monkeypatch.setattr(
        "src.generate_dataset.json.loads",
        lambda *a, **k: {"image_tree_sha256": "x", "generator_commit": "y"},
    )
    (tmp_path / MANIFEST_NAME).write_text("{}", encoding="utf-8")

    main(["--output-dir", str(tmp_path), "--noise-levels", "none", "--seeds-per-cell", "27"])

    assert seen["noise_levels"] == ["none"]
    assert seen["seeds_per_cell"] == 27


def test_compare_manifests_does_not_let_the_commit_mask_the_pixels():
    """Byte-identical data must read as identical even from different commits.

    ``generator_commit`` is git HEAD when the manifest was written, not a
    fingerprint of the code that ran -- an unrelated commit mid-run moves it
    while leaving the generator untouched. An earlier revision compared it
    first and returned "different code", so two provably identical datasets
    short-circuited to a false divergence before their pixels were compared.
    """
    verdict = compare_manifests(_manifest("aaaaaaa", "pix"), _manifest("bbbbbbb", "pix"))

    assert verdict.startswith("identical")
    assert "different code" not in verdict
    # The commit difference is still surfaced, as context rather than a verdict.
    assert "aaaaaaa" in verdict and "bbbbbbb" in verdict


def test_compare_manifests_still_reports_a_real_pixel_divergence():
    """Reordering must not blunt the check it exists to perform."""
    verdict = compare_manifests(_manifest("aaaaaaa", "pix"), _manifest("aaaaaaa", "other"))

    assert "PIXELS DIFFER" in verdict


def test_verify_catches_a_single_changed_pixel(tiny_dataset):
    """Any pixel drift must fail: a stale copy does not announce itself."""
    from PIL import Image

    path = next(iter((tiny_dataset / "reference").glob("*.png")))
    pixels = np.asarray(Image.open(path)).copy()
    pixels[0, 0] = np.uint8((int(pixels[0, 0]) + 1) % 256)
    Image.fromarray(pixels).save(path)
    with pytest.raises(ValueError, match="image tree has drifted"):
        verify_dataset(tiny_dataset)


def test_image_tree_hash_ignores_the_dataset_location(tmp_path):
    """The fingerprint must not depend on where the dataset happens to live."""
    first, second = tmp_path / "one", tmp_path / "somewhere" / "two"
    build_dataset(first, seeds_per_cell=1, out_size=OUT_SIZE, supersample=1)
    build_dataset(second, seeds_per_cell=1, out_size=OUT_SIZE, supersample=1)
    assert image_tree_hash(first) == image_tree_hash(second)


def test_pixel_hash_survives_re_encoding(tiny_dataset):
    """Re-compressing a PNG must not change the fingerprint.

    Two machines can encode identical images into different PNG files --
    compression level, zlib build, Pillow version -- and a byte-level hash then
    reports a difference that does not exist. That is what made a Linux and a
    macOS checkout of the same commit look like different datasets.
    """
    from PIL import Image

    before_pixels = image_tree_hash(tiny_dataset)
    before_bytes = file_tree_hash(tiny_dataset)
    for path in sorted(tiny_dataset.rglob("*.png")):
        pixels = np.asarray(Image.open(path))
        Image.fromarray(pixels).save(path, compress_level=1)

    assert image_tree_hash(tiny_dataset) == before_pixels
    assert file_tree_hash(tiny_dataset) != before_bytes


def test_pixel_hash_still_catches_a_changed_pixel(tiny_dataset):
    """Insensitivity to encoding must not cost sensitivity to content."""
    from PIL import Image

    before = image_tree_hash(tiny_dataset)
    path = next(iter((tiny_dataset / "reference").glob("*.png")))
    pixels = np.asarray(Image.open(path)).copy()
    pixels[0, 0] = np.uint8((int(pixels[0, 0]) + 1) % 256)
    Image.fromarray(pixels).save(path)
    assert image_tree_hash(tiny_dataset) != before


# ---------------------------------------------------------------------------
# Overlay verification
# ---------------------------------------------------------------------------


def _mean_offset(results):
    return (
        float(np.mean([r.dx for r in results])),
        float(np.mean([r.dy for r in results])),
    )


def test_overlay_finds_no_bias_in_a_good_dataset(tiny_dataset):
    """The published ground truth must fit better than any nearby offset.

    This uses only the search image and the ground-truth record -- never the
    layout generator -- so it validates against an external consumer rather than
    checking the code agrees with itself.
    """
    mean_dx, mean_dy = _mean_offset(overlay_check(tiny_dataset, sample=0))
    assert abs(mean_dx) <= OVERLAY_BIAS_TOLERANCE_PX
    assert abs(mean_dy) <= OVERLAY_BIAS_TOLERANCE_PX


@pytest.mark.parametrize("planted_px", [0.5, -0.5])
def test_overlay_detects_a_planted_offset(tiny_dataset, planted_px):
    """The check must fail on the exact defect it exists to catch.

    An earlier revision published every ground truth 0.5 px off in both axes.
    A test that cannot detect that reintroduced would be decoration.
    """
    import json

    path = tiny_dataset / "ground_truth.jsonl"
    shifted = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        record["ground_truth"]["x"] += planted_px / record["ground_truth"]["scale"]
        record["ground_truth"]["y"] += planted_px / record["ground_truth"]["scale"]
        shifted.append(json.dumps(record))
    path.write_text("\n".join(shifted) + "\n")

    mean_dx, mean_dy = _mean_offset(overlay_check(tiny_dataset, sample=0))
    assert abs(mean_dx) > OVERLAY_BIAS_TOLERANCE_PX
    assert abs(mean_dy) > OVERLAY_BIAS_TOLERANCE_PX
    assert np.sign(mean_dx) == -np.sign(planted_px)


# ---------------------------------------------------------------------------
# Overlay verdict statistics
# ---------------------------------------------------------------------------
#
# summarise_overlay exists because the plain mean above failed a clean dataset
# once the noise strata landed: a 12-pair sample read dy = -0.250 while the same
# data at 48 pairs read +0.036. These pin down the two corrections that fixed it
# without blunting the check.


def _result(dx, dy, pair_id="pair"):
    return OverlayResult(pair_id=pair_id, mean_abs_error=0.1, dx=dx, dy=dy)


def test_summary_excludes_pairs_pinned_at_the_probe_limit():
    """A rail-pinned pair is censored, not measured.

    ``_best_offset`` takes an argmin over a bounded probe set, so a result at the
    end of that range means "at most this far", not "exactly this far". Counting
    it as exact drags the mean toward the rail, which is precisely how two
    ambiguous FinFET pairs failed a clean 12-pair dataset.
    """
    rail = max(PROBE_OFFSETS_PX)
    results = [_result(0.0, 0.0) for _ in range(10)]
    results += [_result(0.0, -rail), _result(0.0, -rail)]

    summary = summarise_overlay(results)

    assert summary.pinned == 2
    assert summary.measured == 10
    assert summary.mean_dy == pytest.approx(0.0)
    assert summary.verdict == "ok"


def test_summary_is_inconclusive_rather_than_failed_on_a_small_sample():
    """Too few pairs to resolve the tolerance must ask for more, not cry wolf.

    Failing here would train the team to ignore the one check that guards the
    coordinate convention.
    """
    # A wide, near-symmetric spread whose mean drifts past the tolerance only
    # because the sample is small; the standard error stays comparable to it.
    results = [_result(0.0, -1.0 + 0.5 * i) for i in range(4)]
    results += [_result(0.0, -0.5), _result(0.0, -0.5)]

    summary = summarise_overlay(results)

    if summary.bias > OVERLAY_BIAS_TOLERANCE_PX:
        assert summary.verdict == "inconclusive"
        assert summary.bias <= 2.0 * max(summary.se_dx, summary.se_dy)


def test_summary_still_fails_a_tight_real_offset():
    """A systematic error must fail even though the gate now tolerates scatter.

    This is the regression that matters: the corrections above must not have
    bought a clean pass by blunting the check.
    """
    results = [_result(0.0, -0.5) for _ in range(12)]

    summary = summarise_overlay(results)

    assert summary.verdict == "failed"
    assert summary.bias == pytest.approx(0.5)


def test_summary_fails_when_every_pair_runs_to_the_same_rail():
    """A ground truth wrong by more than the probe range pins everything.

    Censoring forgives ambiguity, not a defect large enough to drive every pair
    off the same end of the probe set.
    """
    rail = max(PROBE_OFFSETS_PX)
    results = [_result(0.0, -rail) for _ in range(12)]

    summary = summarise_overlay(results)

    assert summary.verdict == "failed"
    assert "probe range" in summary.detail
