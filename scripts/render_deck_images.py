"""Render the six deck images, sized to the slide slots that hold them.

Every figure is built from tracked evidence -- ``results/*.csv`` and the
regenerable datasets -- so a claim on a slide and the picture beside it come
from the same run. Nothing here is drawn by hand.

Style is fixed by the deck: navy ``#000036``, no baked-in titles (the slide
carries the heading), predicted marked with a lime cross and truth with a white
ring. Sizes are exact pixel targets, so the slots need no rescaling.

    uv run --all-extras python -m scripts.render_deck_images
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src import config, matcher

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "deck_images"

NAVY = "#000036"
LIME = "#AEFF82"
WHITE = "#FFFFFF"
MUTED = "#8892C8"
DPI = 100

# The confident-and-wrong holdout cases are only ~1.5 px off, so the cross and
# the ring land on the same pixel at slide size. The scale-stress miss is the
# legible one, and it carries the better story: the system flagged it.
FAILURE_CASE = "dram_anchored_pose-stress_0002"
FAILURE_SET = "dataset_scale_stress"


def _style(fig: plt.Figure, axes: list[plt.Axes]) -> None:
    """Paint a figure in deck colours and strip every default chrome element."""
    fig.patch.set_facecolor(NAVY)
    for ax in axes:
        ax.set_facecolor(NAVY)
        for spine in ax.spines.values():
            spine.set_color(MUTED)
            spine.set_linewidth(0.6)
        ax.tick_params(colors=MUTED, labelsize=7, length=3)
        ax.xaxis.label.set_color(MUTED)
        ax.yaxis.label.set_color(MUTED)


def _label(ax: plt.Axes, text: str) -> None:
    """Caption a panel from inside, so no vertical space is spent on it."""
    ax.text(
        0.02,
        0.97,
        text,
        transform=ax.transAxes,
        color=WHITE,
        fontsize=9,
        va="top",
        ha="left",
        bbox={"facecolor": NAVY, "edgecolor": "none", "alpha": 0.75, "pad": 2.5},
    )


def _img(path: Path) -> np.ndarray:
    """Read a grayscale PNG as float32."""
    with Image.open(path) as handle:
        return np.asarray(handle.convert("L"), dtype=np.float32)


def _records(dataset: str) -> dict[str, dict]:
    """Index one dataset's ground truth by pair id."""
    lines = (ROOT / dataset / "ground_truth.jsonl").read_text(encoding="utf-8").splitlines()
    return {r["pair_id"]: r for r in (json.loads(line) for line in lines if line.strip())}


def _rows(csv_name: str) -> dict[str, dict[str, str]]:
    """Index one results CSV by case id."""
    with (ROOT / "results" / csv_name).open() as handle:
        return {r["case_id"]: r for r in csv.DictReader(handle)}


def _surface(
    reference: np.ndarray,
    search: np.ndarray,
    theta: float = 0.0,
    weighted: bool = False,
) -> tuple[np.ndarray, np.ndarray, list]:
    """Return the template, ZNCC surface and candidate peaks at a given pose.

    ``theta`` and ``weighted`` exist so a figure can reproduce the *tier that
    actually answered* rather than a nominal-pose approximation of it. A figure
    explaining why one case failed has to be built at the configuration that
    failed: on a rotated scene the nominal-pose surface is a different surface
    with a different winner, and showing it would invite the obvious question of
    why the pipeline did not simply pick that one.
    """
    template = matcher.build_template(
        reference,
        theta=theta,
        scale=config.NOMINAL_SCALE,
        psf_sigma_px=config.DEFAULT_PSF_SIGMA_PX,
    )
    weight = None
    if weighted:
        from src.localize import _uniqueness_for

        weight = matcher.build_weight(_uniqueness_for(reference), theta, config.NOMINAL_SCALE)
    surface = matcher.zncc_surface(template, search, weight=weight)
    peaks = matcher.top_k_peaks(
        surface, k=config.DEFAULT_TOP_K, nms_radius=config.DEFAULT_NMS_RADIUS_PX
    )
    return template, surface, peaks


def _mark_truth(ax: plt.Axes, x: float, y: float, size: float = 260) -> None:
    """Ring the true centre in white."""
    ax.scatter([x], [y], s=size, facecolors="none", edgecolors=WHITE, linewidths=1.8, zorder=5)


def _mark_prediction(ax: plt.Axes, x: float, y: float, size: float = 150) -> None:
    """Cross the predicted centre in lime."""
    ax.scatter([x], [y], s=size, c=LIME, marker="X", linewidths=0.8, edgecolors=NAVY, zorder=6)


def _blank_ticks(axes: list[plt.Axes]) -> None:
    """Drop ticks from image panels, where pixel indices say nothing."""
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])


# ---------------------------------------------------------------------------
# 1 -- ZNCC correlation surface  (slide 2, 1980x600)
# ---------------------------------------------------------------------------


def image_1_zncc_surface(case: str = "dram_anchored_pose-none_0000") -> Path:
    """Search image, correlation surface, and a zoom on the winning peak.

    The middle panel is the argument the whole project rests on: a periodic
    template against a periodic field produces a *lattice* of near-equal
    maxima, so the failure mode is ambiguity rather than noise.
    """
    record = _records("dataset")[case]
    truth = record["ground_truth"]
    reference = _img(ROOT / "dataset" / record["reference_path"])
    search = _img(ROOT / "dataset" / record["search_path"])
    template, surface, peaks = _surface(reference, search)

    fig, axes = plt.subplots(1, 3, figsize=(19.8, 6.0), dpi=DPI)
    _style(fig, list(axes))

    axes[0].imshow(search, cmap="gray", interpolation="nearest")
    _mark_truth(axes[0], truth["x"], truth["y"])
    _label(axes[0], "10x search image  |  1000 x 1000 px")

    # Surface indices address the template's top-left corner, so truth has to be
    # shifted back by half a template before it can be drawn on a surface panel.
    half_r, half_c = (template.shape[0] - 1) / 2.0, (template.shape[1] - 1) / 2.0

    axes[1].imshow(surface, cmap="inferno", interpolation="nearest")
    _mark_truth(axes[1], truth["x"] - half_c, truth["y"] - half_r)
    _label(axes[1], f"ZNCC surface  |  {len(peaks)} candidate maxima")

    best = peaks[0]
    pad = 45
    r0, r1 = max(0, best.row - pad), min(surface.shape[0], best.row + pad)
    c0, c1 = max(0, best.col - pad), min(surface.shape[1], best.col + pad)
    axes[2].imshow(
        surface[r0:r1, c0:c1],
        cmap="inferno",
        interpolation="nearest",
        extent=(c0, c1, r1, r0),
    )
    _mark_truth(axes[2], truth["x"] - half_c, truth["y"] - half_r, size=5200)
    _label(axes[2], f"winning peak  |  ZNCC {best.score:.3f}")

    _blank_ticks(list(axes))
    fig.tight_layout(pad=0.6)
    out = OUT / "01_zncc_surface.png"
    fig.savefig(out, facecolor=NAVY, dpi=DPI)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 2 -- reference / search pairs with truth  (slide 4, 1920x615)
# ---------------------------------------------------------------------------


def image_2_pairs(
    dram: str = "dram_anchored_pose-none_0000",
    finfet: str = "finfet_anchored_pose-none_0162",
) -> Path:
    """Both architectures, reference beside search, truth ringed on the search."""
    records = _records("dataset")
    fig, axes = plt.subplots(1, 4, figsize=(19.2, 6.15), dpi=DPI)
    _style(fig, list(axes))

    for slot, (case, family) in enumerate(((dram, "DRAM"), (finfet, "FinFET"))):
        record = records[case]
        truth = record["ground_truth"]
        ref_ax, search_ax = axes[slot * 2], axes[slot * 2 + 1]

        ref_ax.imshow(_img(ROOT / "dataset" / record["reference_path"]), cmap="gray")
        _label(ref_ax, f"{family} reference  |  100x  |  1 nm/px")

        search_ax.imshow(_img(ROOT / "dataset" / record["search_path"]), cmap="gray")
        _mark_truth(search_ax, truth["x"], truth["y"])
        _label(search_ax, f"{family} search  |  10x  |  10 nm/px")

    _blank_ticks(list(axes))
    fig.tight_layout(pad=0.6)
    out = OUT / "02_pair_ground_truth.png"
    fig.savefig(out, facecolor=NAVY, dpi=DPI)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 3 -- noise strip  (slide 5, 1800x300)
# ---------------------------------------------------------------------------


def image_3_noise_strip(case: str = "dram_anchored_pose-none_0000") -> Path:
    """One scene at three noise strata.

    The source is the *noise-free control*, which shares scene, geometry and
    seed with the shipped pair, so the only thing separating these panels is
    ``noise_level`` -- the same comparison the control set was built for.
    """
    from src.sem_physics import apply_sem_chain

    record = _records("dataset_control")[case]
    clean = _img(ROOT / "dataset_control" / record["search_path"]) / 255.0
    # A full-field crop renders the three strata indistinguishable, and that is
    # honest rather than a rendering fault: the residual against the clean plate
    # is dominated by PSF blur, which is held constant across strata by design,
    # and the noise itself separates the panels by an rms of only 0.020 (low to
    # medium) and 0.030 (low to high) on a [0, 1] scale. Crop small and let the
    # panel upsample, so one source pixel becomes a visible block and the grain
    # -- which is the thing the slide is about -- is actually on screen.
    crop = clean[470:520, 430:530]

    fig, axes = plt.subplots(1, 3, figsize=(18.0, 3.0), dpi=DPI)
    _style(fig, list(axes))
    for ax, level in zip(axes, ("low", "medium", "high"), strict=True):
        # One seed for all three, so the panels differ by stratum and nothing else.
        noisy = apply_sem_chain(
            crop,
            px_nm=config.SEARCH_PX_NM,
            params={"preset": "search", "noise_level": level},
            rng=np.random.default_rng(7),
        )
        ax.imshow(noisy, cmap="gray", interpolation="nearest")
        _label(ax, level)

    _blank_ticks(list(axes))
    fig.tight_layout(pad=0.4)
    out = OUT / "03_noise_strip.png"
    fig.savefig(out, facecolor=NAVY, dpi=DPI)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 4 -- error CDF  (slide 9, 1500x1000)
# ---------------------------------------------------------------------------


def image_4_error_cdf() -> Path:
    """Cumulative error for the anchored stratum against all pairs.

    Log x because the distribution is bimodal: the anchored half resolves to
    hundredths of a pixel, the unanchored half sits hundreds of pixels away,
    and a linear axis renders the interesting half as a single vertical line.
    """
    rows = list(_rows("full_324.csv").values())
    every = np.sort([float(r["err_px"]) for r in rows])
    anchored = np.sort([float(r["err_px"]) for r in rows if r["anchored"] == "anchored"])

    fig, ax = plt.subplots(figsize=(15.0, 10.0), dpi=DPI)
    _style(fig, [ax])

    for data, colour, name in (
        (anchored, LIME, f"anchored  (n={len(anchored)})"),
        (every, MUTED, f"all pairs  (n={len(every)})"),
    ):
        ax.step(
            np.maximum(data, 1e-4),
            np.arange(1, len(data) + 1) / len(data),
            where="post",
            color=colour,
            linewidth=2.4,
            label=name,
        )

    # 4 px and 5 px are almost the same place on a log axis, so their labels
    # overlap at a shared height. Alternate the rows instead of dropping one --
    # the problem statement asks for all four.
    for index, tol in enumerate(config.TOLERANCES_PX):
        ax.axvline(tol, color=WHITE, linestyle=":", linewidth=1.0, alpha=0.45)
        ax.text(
            tol,
            1.012 if index % 2 == 0 else 1.042,
            f"{tol:g} px",
            color=WHITE,
            fontsize=8,
            ha="center",
            alpha=0.75,
        )

    ax.set_xscale("log")
    ax.set_xlim(1e-3, 2e3)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("euclidean error (search px, log scale)", fontsize=9)
    ax.set_ylabel("fraction of cases at or below", fontsize=9)
    ax.grid(True, which="both", color=MUTED, alpha=0.18, linewidth=0.5)
    legend = ax.legend(loc="upper left", frameon=False, fontsize=10)
    for text in legend.get_texts():
        text.set_color(WHITE)

    fig.tight_layout(pad=0.8)
    out = OUT / "04_error_cdf.png"
    fig.savefig(out, facecolor=NAVY, dpi=DPI)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 5 -- prediction overlay  (slide 9, 1400x1030)
# ---------------------------------------------------------------------------


def image_5_prediction_overlay(case: str = "dram_anchored_pose-large_0077") -> Path:
    """Show a clean anchored result: prediction on truth, with the error stated."""
    record = _records("dataset")[case]
    row = _rows("full_324.csv")[case]
    truth = record["ground_truth"]
    search = _img(ROOT / "dataset" / record["search_path"])
    px, py, err = float(row["pred_x"]), float(row["pred_y"]), float(row["err_px"])

    fig, ax = plt.subplots(figsize=(14.0, 10.3), dpi=DPI)
    _style(fig, [ax])
    ax.imshow(search, cmap="gray", interpolation="nearest")
    _mark_truth(ax, truth["x"], truth["y"], size=700)
    _mark_prediction(ax, px, py, size=260)

    ax.text(
        0.985,
        0.03,
        f"error {err:.3f} px  ({err * config.SEARCH_PX_NM:.2f} nm)\n"
        f"true (white ring)  {truth['x']:.2f}, {truth['y']:.2f}\n"
        f"predicted (lime cross)  {px:.2f}, {py:.2f}",
        transform=ax.transAxes,
        color=WHITE,
        fontsize=9,
        va="bottom",
        ha="right",
        linespacing=1.5,
        bbox={"facecolor": NAVY, "edgecolor": MUTED, "alpha": 0.85, "pad": 5},
    )
    _blank_ticks([ax])
    fig.tight_layout(pad=0.5)
    out = OUT / "05_prediction_overlay.png"
    fig.savefig(out, facecolor=NAVY, dpi=DPI)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 6 -- failure case  (slide 11, 1800x600) -- mandatory per section 4D
# ---------------------------------------------------------------------------


def image_6_failure_case(case: str = FAILURE_CASE) -> Path:
    """Show the reference, the wrong answer, and the peaks that caused it.

    The third panel is the root cause made visible: the candidates the winner
    beat are separated from it by hundredths of a ZNCC point, which is what
    "the evidence does not identify a position" looks like on a surface.
    """
    record = _records(FAILURE_SET)[case]
    row = _rows("scale_stress_48_pose.csv")[case]
    truth = record["ground_truth"]
    reference = _img(ROOT / FAILURE_SET / record["reference_path"])
    search = _img(ROOT / FAILURE_SET / record["search_path"])
    # Rebuild the surface the losing run actually saw: its estimated rotation,
    # and weighting on, because it answered from the `ambiguous` tier.
    template, _surface_map, peaks = _surface(
        reference, search, theta=float(row["theta_est"]), weighted=row["mode_used"] != "fast"
    )
    px, py, err = float(row["pred_x"]), float(row["pred_y"]), float(row["err_px"])

    fig, axes = plt.subplots(1, 3, figsize=(18.0, 6.0), dpi=DPI)
    _style(fig, list(axes))

    axes[0].imshow(reference, cmap="gray")
    _label(axes[0], "reference  |  100x")

    axes[1].imshow(search, cmap="gray", interpolation="nearest")
    _mark_truth(axes[1], truth["x"], truth["y"], size=420)
    _mark_prediction(axes[1], px, py, size=220)
    _label(axes[1], f"error {err:.1f} px  |  flagged, confidence {float(row['confidence']):.3f}")

    axes[2].imshow(search, cmap="gray", alpha=0.32, interpolation="nearest")
    half_r, half_c = (template.shape[0] - 1) / 2.0, (template.shape[1] - 1) / 2.0
    xs = [p.col + half_c for p in peaks]
    ys = [p.row + half_r for p in peaks]
    scores = [p.score for p in peaks]
    dots = axes[2].scatter(
        xs, ys, c=scores, cmap="inferno", s=95, edgecolors=NAVY, linewidths=0.6, zorder=4
    )
    # Labelling the top four scores says little -- they are all the same number
    # to three places. The pair that explains the failure is the winner against
    # the best candidate near the true site: that is the margin the pipeline had
    # to work with, and it is nothing.
    near_truth = min(
        range(len(peaks)), key=lambda i: (xs[i] - truth["x"]) ** 2 + (ys[i] - truth["y"]) ** 2
    )
    margin = scores[0] - scores[near_truth]
    for index, note, offset in (
        (0, f"winner  {scores[0]:.4f}   (+{margin:.4f} over truth)", (16, -22)),
        (
            near_truth,
            f"best near truth  {scores[near_truth]:.4f}  (rank {near_truth + 1})",
            (16, 18),
        ),
    ):
        axes[2].annotate(
            note,
            (xs[index], ys[index]),
            textcoords="offset points",
            xytext=offset,
            color=WHITE,
            fontsize=8,
            ha="right" if xs[index] > search.shape[1] * 0.6 else "left",
            arrowprops={"arrowstyle": "-", "color": WHITE, "lw": 0.5, "alpha": 0.6},
        )
    _mark_truth(axes[2], truth["x"], truth["y"], size=420)
    _label(axes[2], f"top {len(peaks)} candidates  |  ZNCC spread {max(scores) - min(scores):.4f}")
    bar = fig.colorbar(dots, ax=axes[2], fraction=0.046, pad=0.02)
    bar.ax.tick_params(colors=MUTED, labelsize=7)
    bar.outline.set_edgecolor(MUTED)

    _blank_ticks(list(axes))
    fig.tight_layout(pad=0.6)
    out = OUT / "06_failure_case.png"
    fig.savefig(out, facecolor=NAVY, dpi=DPI)
    plt.close(fig)
    return out


def main() -> int:
    """Render every deck image and report its path and pixel size."""
    OUT.mkdir(parents=True, exist_ok=True)
    builders = (
        image_1_zncc_surface,
        image_2_pairs,
        image_3_noise_strip,
        image_4_error_cdf,
        image_5_prediction_overlay,
        image_6_failure_case,
    )
    for build in builders:
        path = build()
        with Image.open(path) as handle:
            width, height = handle.size
        print(f"  {path.relative_to(ROOT)}  {width}x{height}  ({width / height:.2f}:1)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
