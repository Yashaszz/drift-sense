import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# ------------------------------------------------------------
# Make src importable
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sem_physics import apply_sem_chain


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

IMAGE_PATH = (
    ROOT
    / "dataset"
    / "reference"
    / "dram_anchored_pose-large_0018.png"
)

PX_NM = 1.0

NOISE_LEVELS = [
    "none",
    "low",
    "medium",
    "high",
]

SEEDS = [
    42,
    43,
    44,
    45,
    46,
]


# ------------------------------------------------------------
# Load image
# ------------------------------------------------------------

if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"Could not find image:\n{IMAGE_PATH}"
    )

img = np.asarray(
    Image.open(IMAGE_PATH).convert("L"),
    dtype=np.float32,
)

img /= 255.0


print("=" * 60)
print("R2 SEM NOISE STRATA VALIDATION")
print("=" * 60)

print()
print("Input:")
print("  file :", IMAGE_PATH)
print("  shape:", img.shape)
print("  dtype:", img.dtype)
print(
    "  range:",
    float(img.min()),
    "to",
    float(img.max()),
)


# ------------------------------------------------------------
# Helper metrics
# ------------------------------------------------------------

def compute_metrics(
    output,
    baseline,
):
    difference = output - baseline

    abs_difference = np.abs(
        difference
    )

    mae = float(
        np.mean(abs_difference)
    )

    rmse = float(
        np.sqrt(
            np.mean(
                difference ** 2
            )
        )
    )

    diff_std = float(
        np.std(difference)
    )

    p95 = float(
        np.percentile(
            abs_difference,
            95,
        )
    )

    p99 = float(
        np.percentile(
            abs_difference,
            99,
        )
    )

    return {
        "mae": mae,
        "rmse": rmse,
        "diff_std": diff_std,
        "p95": p95,
        "p99": p99,
    }


# ------------------------------------------------------------
# Generate outputs for multiple seeds
# ------------------------------------------------------------

results = {
    level: {}
    for level in NOISE_LEVELS
}


for level in NOISE_LEVELS:

    print()
    print("-" * 60)
    print(f"NOISE LEVEL: {level.upper()}")
    print("-" * 60)

    for seed in SEEDS:

        rng = np.random.default_rng(
            seed
        )

        output = apply_sem_chain(
            img,
            px_nm=PX_NM,
            params={
                "preset": "reference",
                "noise_level": level,
            },
            rng=rng,
        )

        # ----------------------------------------------------
        # Basic validation
        # ----------------------------------------------------

        assert output.shape == img.shape

        assert output.dtype == np.float32

        assert np.all(
            np.isfinite(output)
        )

        assert (
            output.min() >= 0.0
            and output.max() <= 1.0
        )

        results[level][seed] = output

        print(
            f"seed={seed} | "
            f"mean={output.mean():.6f} | "
            f"std={output.std():.6f}"
        )


# ------------------------------------------------------------
# 1. Reproducibility test
# ------------------------------------------------------------

print()
print("=" * 60)
print("1. REPRODUCIBILITY TEST")
print("=" * 60)

for level in NOISE_LEVELS:

    rng_a = np.random.default_rng(999)

    rng_b = np.random.default_rng(999)

    output_a = apply_sem_chain(
        img,
        px_nm=PX_NM,
        params={
            "preset": "reference",
            "noise_level": level,
        },
        rng=rng_a,
    )

    output_b = apply_sem_chain(
        img,
        px_nm=PX_NM,
        params={
            "preset": "reference",
            "noise_level": level,
        },
        rng=rng_b,
    )

    identical = np.array_equal(
        output_a,
        output_b,
    )

    print(
        f"{level:>6}: "
        f"{'PASS' if identical else 'FAIL'}"
    )

    assert identical


# ------------------------------------------------------------
# 2. Different-seed stochasticity test
# ------------------------------------------------------------

print()
print("=" * 60)
print("2. DIFFERENT-SEED STOCHASTICITY TEST")
print("=" * 60)

for level in NOISE_LEVELS:

    a = results[level][42]

    b = results[level][43]

    difference = np.mean(
        np.abs(a - b)
    )

    print(
        f"{level:>6}: "
        f"MAE(seed42, seed43) = "
        f"{difference:.6f}"
    )

    # For the actual stochastic levels, outputs should differ.
    if level != "none":
        assert difference > 0.0


# ------------------------------------------------------------
# 3. Compare each output against input
# ------------------------------------------------------------

print()
print("=" * 60)
print("3. DEGRADATION METRICS")
print("=" * 60)

metrics = {
    level: []
    for level in NOISE_LEVELS
}


for level in NOISE_LEVELS:

    for seed in SEEDS:

        output = results[level][seed]

        m = compute_metrics(
            output,
            img,
        )

        metrics[level].append(m)

    mae_mean = np.mean(
        [m["mae"] for m in metrics[level]]
    )

    rmse_mean = np.mean(
        [m["rmse"] for m in metrics[level]]
    )

    diff_std_mean = np.mean(
        [m["diff_std"] for m in metrics[level]]
    )

    p95_mean = np.mean(
        [m["p95"] for m in metrics[level]]
    )

    p99_mean = np.mean(
        [m["p99"] for m in metrics[level]]
    )

    print()
    print(level.upper())

    print(
        f"  MAE      : {mae_mean:.6f}"
    )

    print(
        f"  RMSE     : {rmse_mean:.6f}"
    )

    print(
        f"  diff std : {diff_std_mean:.6f}"
    )

    print(
        f"  p95      : {p95_mean:.6f}"
    )

    print(
        f"  p99      : {p99_mean:.6f}"
    )


# ------------------------------------------------------------
# 4. Noise ordering test
# ------------------------------------------------------------

print()
print("=" * 60)
print("4. NOISE ORDERING TEST")
print("=" * 60)


mean_mae = {}

for level in NOISE_LEVELS:

    mean_mae[level] = np.mean(
        [
            m["mae"]
            for m in metrics[level]
        ]
    )


for level in NOISE_LEVELS:

    print(
        f"{level:>6}: "
        f"MAE = {mean_mae[level]:.6f}"
    )


print()

print(
    "Expected ordering:"
)

print(
    "  low < medium < high"
)

# ------------------------------------------------------------
# Important:
# "none" currently means baseline preset noise,
# not zero noise.
# Therefore do not include "none" in
# the monotonic severity assertion.
# ------------------------------------------------------------

assert (
    mean_mae["low"]
    <= mean_mae["medium"]
)

assert (
    mean_mae["medium"]
    <= mean_mae["high"]
)

print(
    "Noise severity ordering: PASS"
)


# ------------------------------------------------------------
# 5. Detect none == medium semantic issue
# ------------------------------------------------------------

print()
print("=" * 60)
print("5. NONE / MEDIUM SEMANTIC CHECK")
print("=" * 60)

none_output = results["none"][42]

medium_output = results["medium"][42]

none_medium_difference = np.mean(
    np.abs(
        none_output
        - medium_output
    )
)

print(
    "none -> medium MAE:",
    f"{none_medium_difference:.8f}"
)

if none_medium_difference == 0.0:

    print()
    print(
        "WARNING:"
    )

    print(
        "'none' and 'medium' produce "
        "identical outputs."
    )

    print(
        "This is because the current "
        "sem_physics.py treats 'none' "
        "as no additional scaling, "
        "which is equivalent to the "
        "reference medium baseline."
    )

else:

    print(
        "none and medium are distinct."
    )


# ------------------------------------------------------------
# 6. Pairwise distance matrix
# ------------------------------------------------------------

print()
print("=" * 60)
print("6. PAIRWISE NOISE-STRATA DISTANCE")
print("=" * 60)

pairwise = {}

for level_a in NOISE_LEVELS:

    pairwise[level_a] = {}

    for level_b in NOISE_LEVELS:

        distances = []

        for seed in SEEDS:

            a = results[level_a][seed]

            b = results[level_b][seed]

            distances.append(
                np.mean(
                    np.abs(
                        a - b
                    )
                )
            )

        pairwise[level_a][level_b] = np.mean(
            distances
        )


print()

print(
    "             "
    + "".join(
        f"{level:>12}"
        for level in NOISE_LEVELS
    )
)

for level_a in NOISE_LEVELS:

    print(
        f"{level_a:>12}",
        end="",
    )

    for level_b in NOISE_LEVELS:

        print(
            f"{pairwise[level_a][level_b]:12.6f}",
            end="",
        )

    print()


# ------------------------------------------------------------
# 7. Clipping check
# ------------------------------------------------------------

print()
print("=" * 60)
print("7. CLIPPING CHECK")
print("=" * 60)

for level in NOISE_LEVELS:

    clipping_fractions = []

    for seed in SEEDS:

        output = results[level][seed]

        clipped = np.logical_or(
            output <= 0.0,
            output >= 1.0,
        )

        fraction = np.mean(
            clipped
        )

        clipping_fractions.append(
            fraction
        )

    mean_clipping = np.mean(
        clipping_fractions
    )

    print(
        f"{level:>6}: "
        f"{mean_clipping * 100:.4f}% "
        "pixels at bounds"
    )


# ------------------------------------------------------------
# 8. Difference-image visualization
# ------------------------------------------------------------

print()
print("=" * 60)
print("8. SAVING DIFFERENCE VISUALIZATION")
print("=" * 60)

seed = 42

fig, axes = plt.subplots(
    2,
    4,
    figsize=(16, 8),
)

for column, level in enumerate(
    NOISE_LEVELS
):

    output = results[level][seed]

    # Original output
    axes[0, column].imshow(
        output,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[0, column].set_title(
        f"{level}\noutput"
    )

    axes[0, column].axis("off")

    # Difference from original R1 image
    difference = np.abs(
        output - img
    )

    axes[1, column].imshow(
        difference,
        cmap="magma",
        vmin=0,
        vmax=np.percentile(
            difference,
            99,
        ),
    )

    axes[1, column].set_title(
        f"{level}\n|output - input|"
    )

    axes[1, column].axis("off")


plt.tight_layout()

output_path = (
    ROOT
    / "noise_strata_validation_strong.png"
)

plt.savefig(
    output_path,
    dpi=150,
    bbox_inches="tight",
)

plt.close()

print(
    "Saved:",
    output_path,
)


# ------------------------------------------------------------
# Final result
# ------------------------------------------------------------

print()
print("=" * 60)
print("NOISE STRATA VALIDATION PASSED")
print("=" * 60)