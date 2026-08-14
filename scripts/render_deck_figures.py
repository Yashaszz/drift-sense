"""Render R1's deck figures: the ground-truth overlay, and DRAM vs FinFET.

Two figures, both built from the shipped dataset rather than from freshly
invented examples, so what a judge sees on the slide is a pair that exists in
``dataset/`` and can be checked by opening it.

``ground_truth_overlay.png``
    The claim that the published ground truth is correct, made visually. Shows
    the search image with the reference's footprint drawn at the published
    (x, y), the reference itself, and the reference rebuilt from the search
    image using only the ground-truth record. If the convention were off by
    half a pixel the third panel would not line up with the second.

``dram_vs_finfet.png``
    Why the two architectures behave differently. Same field, same scale, with
    the periods that matter annotated.

Uses PIL rather than matplotlib: PIL is already a runtime dependency and the
submission has to install from the committed lockfile on a clean machine.

Run from the repository root:

    python scripts/render_deck_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.render import GroundTruth, reconstruct_from_gt  # noqa: E402

DATA = ROOT / "dataset"
OUT_DIR = ROOT / "docs"
PANEL = 300
MARGIN = 14
LABEL_H = 30


def _load(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path), dtype=np.float32) / 255.0


def _panel(array: np.ndarray) -> Image.Image:
    scaled = np.clip(array, 0.0, 1.0) * 255.0
    return Image.fromarray(scaled.astype(np.uint8), mode="L").resize(
        (PANEL, PANEL), Image.Resampling.LANCZOS
    )


def _records() -> list[dict]:
    return [
        json.loads(line)
        for line in (DATA / "ground_truth.jsonl").read_text().splitlines()
        if line.strip()
    ]


def _pick(records: list[dict], pair_id: str) -> dict:
    for record in records:
        if record["pair_id"] == pair_id:
            return record
    msg = f"{pair_id} not in {DATA}; regenerate the dataset first"
    raise SystemExit(msg)


def ground_truth_overlay(records: list[dict], pair_id: str) -> Path:
    """Draw the search image with the reference located on it, and the rebuild."""
    record = _pick(records, pair_id)
    reference = _load(DATA / record["reference_path"])
    search = _load(DATA / record["search_path"])
    truth = GroundTruth(**record["ground_truth"])
    rebuilt = reconstruct_from_gt(search, truth, out_size=reference.shape[0])

    # The reference footprint inside the search image: 1000 px at 1 nm/px is a
    # 1 um field, which at 10 nm/px covers 100 px. Drawn at the published centre.
    footprint = reference.shape[0] * config.REF_PX_NM / truth.scale
    marked = _panel(search).convert("RGB")
    draw = ImageDraw.Draw(marked)
    k = PANEL / search.shape[0]
    half = footprint * k / 2.0
    cx, cy = truth.x * k, truth.y * k
    draw.rectangle([cx - half, cy - half, cx + half, cy + half], outline=(255, 60, 60), width=2)
    draw.line([cx - 8, cy, cx + 8, cy], fill=(255, 60, 60), width=1)
    draw.line([cx, cy - 8, cx, cy + 8], fill=(255, 60, 60), width=1)

    panels = [
        (marked, f"search 10 nm/px - footprint at published ({truth.x:.1f}, {truth.y:.1f})"),
        (_panel(reference).convert("RGB"), "reference 1 nm/px - what we must locate"),
        (_panel(rebuilt).convert("RGB"), "rebuilt from search + ground truth alone"),
    ]

    width = MARGIN + len(panels) * (PANEL + MARGIN)
    canvas = Image.new("RGB", (width, MARGIN + PANEL + LABEL_H + MARGIN), "white")
    text = ImageDraw.Draw(canvas)
    for index, (panel, label) in enumerate(panels):
        left = MARGIN + index * (PANEL + MARGIN)
        canvas.paste(panel, (left, MARGIN))
        text.text((left, MARGIN + PANEL + 6), label, fill="black")

    text.text(
        (MARGIN, MARGIN + PANEL + 18),
        f"{pair_id}   panel 3 uses only the search image and ground_truth.jsonl, so "
        "a half-pixel convention error would misalign it against panel 2",
        fill="black",
    )

    out = OUT_DIR / "ground_truth_overlay.png"
    canvas.save(out)
    return out


def dram_vs_finfet(records: list[dict], pairs: dict[str, str]) -> Path:
    """Show both architectures at reference and search scale, with periods named."""
    rows = []
    for arch, pair_id in pairs.items():
        record = _pick(records, pair_id)
        params = record["layout_params"]
        if arch == "dram":
            detail = f"pitch {params['pitch_nm']:.0f} nm, line {params['line_width_nm']:.0f} nm"
        else:
            detail = (
                f"fin pitch {params['fin_pitch_nm']:.0f} nm, "
                f"gate pitch {params['gate_pitch_nm']:.0f} nm"
            )
        rows.append(
            (
                arch,
                detail,
                _load(DATA / record["reference_path"]),
                _load(DATA / record["search_path"]),
            )
        )

    width = MARGIN + 2 * (PANEL + MARGIN)
    height = MARGIN + len(rows) * (PANEL + LABEL_H + MARGIN)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    for index, (arch, detail, reference, search) in enumerate(rows):
        top = MARGIN + index * (PANEL + LABEL_H + MARGIN)
        canvas.paste(_panel(reference), (MARGIN, top))
        canvas.paste(_panel(search), (MARGIN + PANEL + MARGIN, top))
        draw.text((MARGIN, top + PANEL + 4), f"{arch} reference 1 nm/px - {detail}", fill="black")
        draw.text(
            (MARGIN + PANEL + MARGIN, top + PANEL + 4),
            f"{arch} search 10 nm/px - reference covers ~100x100 px of this",
            fill="black",
        )

    out = OUT_DIR / "dram_vs_finfet.png"
    canvas.save(out)
    return out


def main() -> int:
    """Render both figures into ``docs/``."""
    if not (DATA / "ground_truth.jsonl").exists():
        msg = f"no dataset at {DATA}; run generate_dataset first"
        raise SystemExit(msg)

    records = _records()
    for path in (
        ground_truth_overlay(records, "dram_anchored_pose-none_0000"),
        dram_vs_finfet(
            records,
            {"dram": "dram_anchored_pose-none_0000", "finfet": "finfet_anchored_pose-none_0162"},
        ),
    ):
        print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
