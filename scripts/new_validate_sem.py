import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Make src importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sem_physics import apply_sem_chain


# ------------------------------------------------------------
# R1 images
# ------------------------------------------------------------

reference_path = ROOT / "dataset" / "reference" / "dram_anchored_pose-large_0018.png"
search_path = ROOT / "dataset" / "search" / "dram_anchored_pose-large_0018.png"


# ------------------------------------------------------------
# Load images
# ------------------------------------------------------------

reference = np.asarray(
    Image.open(reference_path).convert("L")
)

search = np.asarray(
    Image.open(search_path).convert("L")
)


# ------------------------------------------------------------
# Apply ECE-2 SEM physics
# ------------------------------------------------------------

reference_sem = apply_sem_chain(
    reference,
    px_nm=1.0,
    params="reference",
    rng=np.random.default_rng(42),
)

search_sem = apply_sem_chain(
    search,
    px_nm=10.0,
    params="search",
    rng=np.random.default_rng(43),
)


# ------------------------------------------------------------
# Basic validation
# ------------------------------------------------------------

for name, original, degraded in [
    ("Reference", reference, reference_sem),
    ("Search", search, search_sem),
]:
    assert original.shape == degraded.shape
    assert degraded.dtype == np.float32
    assert np.all(np.isfinite(degraded))
    assert degraded.min() >= 0.0
    assert degraded.max() <= 1.0

    print(f"\n{name}")
    print("-" * 40)
    print("Input shape :", original.shape)
    print("Input dtype :", original.dtype)
    print("Output shape:", degraded.shape)
    print("Output dtype:", degraded.dtype)
    print("Output min  :", degraded.min())
    print("Output max  :", degraded.max())


# ------------------------------------------------------------
# Display
# ------------------------------------------------------------

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12, 10),
)

images = [
    (axes[0, 0], reference, "R1 Reference - Clean"),
    (axes[0, 1], reference_sem, "ECE-2 Reference - SEM"),
    (axes[1, 0], search, "R1 Search - Clean"),
    (axes[1, 1], search_sem, "ECE-2 Search - SEM"),
]

for ax, image, title in images:
    ax.imshow(
        image,
        cmap="gray",
        vmin=0,
        vmax=255 if image.dtype != np.float32 else 1,
    )
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
plt.show()

print("\nR1 → ECE-2 SEM validation passed.")