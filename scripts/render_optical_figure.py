"""Render the RGB-bonus figure: the same layout through SEM and through optics.

Produces one PNG for the deck showing why visible light cannot address this
problem. Left panel is the geometry as the SEM sees it; the three right panels
are the red, green and blue channels of a diffraction-limited capture of the
same field.

Uses PIL rather than matplotlib deliberately: PIL is already a runtime
dependency, and the submission has to install and run on a clean machine from
the committed lockfile. A figure script is not worth a new dependency.

Run from the repository root:

    python scripts/render_optical_figure.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.generate_dataset import EXTENT_NM, _sample_layout  # noqa: E402
from src.optical import CHANNEL_WAVELENGTHS_NM, cutoff_period_nm, render_rgb  # noqa: E402
from src.render import plan_pair, rasterize  # noqa: E402

PANEL_PX = 320
"""Edge length of each rendered panel, in pixels."""

MARGIN = 12
LABEL_H = 22
OUT_PATH = ROOT / "docs" / "rgb_optical_bonus.png"


def _to_image(array: np.ndarray) -> Image.Image:
    """Convert a float array in [0, 1] to an 8-bit greyscale panel."""
    scaled = np.clip(array, 0.0, 1.0) * 255.0
    return Image.fromarray(scaled.astype(np.uint8), mode="L").resize(
        (PANEL_PX, PANEL_PX), Image.Resampling.LANCZOS
    )


def _stretch(array: np.ndarray) -> np.ndarray:
    """Normalise a panel to its own range, so faint structure is visible.

    The optical panels retain a few percent of the geometry's contrast. Shown at
    a common scale they are flat grey rectangles, which reads as "the renderer is
    broken" rather than "the signal is gone". Stretching each panel to its own
    range shows what *is* there, and the caption carries the real numbers.
    """
    lo, hi = float(array.min()), float(array.max())
    if hi - lo < 1e-9:
        return np.full_like(array, 0.5)
    return (array - lo) / (hi - lo)


def main() -> int:
    """Render the figure and write it to ``docs/``."""
    architectures = ("dram", "finfet")
    columns = 4
    width = MARGIN + columns * (PANEL_PX + MARGIN)
    height = MARGIN + len(architectures) * (PANEL_PX + LABEL_H + MARGIN) + LABEL_H

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    for row, arch in enumerate(architectures):
        rng = np.random.default_rng(7)
        plan = plan_pair(rng, extent_nm=EXTENT_NM, out_size=512)
        layout = _sample_layout(arch, rng, anchored=True, anchor_centre_nm=plan.crop_centre_nm)
        geometry = rasterize(layout, plan.crop_centre_nm, config.REF_PX_NM, 512, supersample=2)
        rgb = render_rgb(geometry, config.REF_PX_NM)

        panels = [("geometry (SEM sees this)", geometry, None)]
        for index, band in enumerate(("r", "g", "b")):
            retained = float(rgb[..., index].std()) / float(geometry.std())
            label = (
                f"{band.upper()} {CHANNEL_WAVELENGTHS_NM[band]:.0f}nm  "
                f"cut {cutoff_period_nm(CHANNEL_WAVELENGTHS_NM[band]):.0f}nm  "
                f"{retained:.1%}"
            )
            panels.append((label, rgb[..., index], retained))

        top = MARGIN + row * (PANEL_PX + LABEL_H + MARGIN)
        for col, (label, array, retained) in enumerate(panels):
            left = MARGIN + col * (PANEL_PX + MARGIN)
            shown = array if retained is None else _stretch(array)
            canvas.paste(_to_image(shown), (left, top))
            draw.text((left, top + PANEL_PX + 4), f"{arch}  {label}", fill="black")

    footer = (
        "Optical panels are contrast-stretched to their own range; the percentage is the "
        "signal actually retained. NA 0.95. Fine structure is removed, not attenuated."
    )
    draw.text((MARGIN, height - LABEL_H + 4), footer, fill="black")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT_PATH)
    print(f"wrote {OUT_PATH} ({OUT_PATH.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
