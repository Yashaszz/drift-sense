"""Write real 3-channel RGB PNGs of the optical captures.

The composite figure in ``docs/rgb_optical_bonus.png`` is contrast-stretched per
panel so the surviving structure is visible at all. These are the honest files:
true RGB, unstretched, exactly what a diffraction-limited colour capture of
these layouts would deliver. Most of them look like flat grey rectangles, which
is the point.

Both are written for each pair so the difference is checkable side by side:

    <pair>_geometry.png   the layout as rendered, for reference
    <pair>_optical.png    the same field through the optic, true RGB
    <pair>_optical_x8.png the same, contrast boosted 8x about mid grey

The x8 version exists because "flat grey" and "broken renderer" look identical
on a slide. It is labelled in the filename so it can never be mistaken for the
real thing.

Run from the repository root:

    python scripts/render_rgb_samples.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.generate_dataset import EXTENT_NM, _sample_layout  # noqa: E402
from src.optical import render_rgb  # noqa: E402
from src.render import plan_pair, rasterize  # noqa: E402

OUT_DIR = ROOT / "docs" / "rgb_samples"
SIZE = 512
BOOST = 8.0


def _save_rgb(array: np.ndarray, path: Path) -> None:
    Image.fromarray((np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB").save(path)


def main() -> int:
    """Write geometry, true-RGB and boosted-RGB PNGs for both architectures."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for arch in ("dram", "finfet"):
        for anchored in (True, False):
            rng = np.random.default_rng(7 if anchored else 11)
            plan = plan_pair(rng, extent_nm=EXTENT_NM, out_size=SIZE)
            layout = _sample_layout(
                arch,
                rng,
                anchored=anchored,
                anchor_centre_nm=plan.crop_centre_nm,
            )
            geometry = rasterize(layout, plan.crop_centre_nm, config.REF_PX_NM, SIZE, supersample=2)
            optical = render_rgb(geometry, config.REF_PX_NM)

            tag = f"{arch}_{'anchored' if anchored else 'unanchored'}"
            grey = np.repeat(geometry[..., None], 3, axis=-1)
            boosted = 0.5 + (optical - optical.mean()) * BOOST

            for suffix, image in (
                ("geometry", grey),
                ("optical", optical),
                (f"optical_x{BOOST:.0f}", boosted),
            ):
                path = OUT_DIR / f"{tag}_{suffix}.png"
                _save_rgb(image, path)
                written.append(path)

            retained = [float(optical[..., c].std()) / float(geometry.std()) for c in range(3)]
            print(
                f"{tag:20s} retained R {retained[0]:.1%}  G {retained[1]:.1%}  B {retained[2]:.1%}"
            )

    print(f"\nwrote {len(written)} files to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
