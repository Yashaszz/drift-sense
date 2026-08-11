import sys
import json
from pathlib import Path

import numpy as np
from PIL import Image

# ------------------------------------------------------------
# Make src importable
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

sys.path.insert(0, str(SRC))

from sem_physics import apply_sem_chain


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

DATASET = ROOT / "dataset"

REFERENCE_DIR = DATASET / "reference"
SEARCH_DIR = DATASET / "search"

GROUND_TRUTH_FILE = DATASET / "ground_truth.jsonl"

NOISE_LEVELS = [
    "none",
    "low",
    "medium",
    "high",
]

# Reference pixel scale is fixed by the R1 pipeline.
REFERENCE_PX_NM = 1.0


# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------

def load_image(path):
    """Load image as grayscale float32 in [0, 1]."""

    image = np.asarray(
        Image.open(path).convert("L"),
        dtype=np.float32,
    )

    image /= 255.0

    return image


def validate_input(image):
    """Validate R1-generated input image."""

    problems = []

    if image.ndim != 2:
        problems.append(
            f"shape is not 2-D: {image.shape}"
        )

    if image.dtype != np.float32:
        problems.append(
            f"dtype is {image.dtype}, expected float32"
        )

    if not np.all(np.isfinite(image)):
        problems.append(
            "contains NaN/Inf"
        )

    if image.min() < 0.0 or image.max() > 1.0:
        problems.append(
            "range outside [0,1]"
        )

    return problems


def validate_output(output, input_image):
    """Validate SEM physics output contract."""

    problems = []

    if output.shape != input_image.shape:
        problems.append(
            f"shape changed: "
            f"{input_image.shape} -> {output.shape}"
        )

    if output.dtype != np.float32:
        problems.append(
            f"dtype is {output.dtype}, expected float32"
        )

    if not np.all(np.isfinite(output)):
        problems.append(
            "contains NaN/Inf"
        )

    if output.min() < 0.0 or output.max() > 1.0:
        problems.append(
            "range outside [0,1]"
        )

    return problems


def mae(a, b):
    return float(
        np.mean(
            np.abs(a - b)
        )
    )


def rmse(a, b):
    return float(
        np.sqrt(
            np.mean(
                (a - b) ** 2
            )
        )
    )


# ------------------------------------------------------------
# Load R1 ground truth
# ------------------------------------------------------------

print()
print("=" * 70)
print("R2 LARGE-SCALE SEM DATASET VALIDATION")
print("=" * 70)

if not GROUND_TRUTH_FILE.exists():
    print()
    print("ERROR: ground_truth.jsonl not found:")
    print(f"  {GROUND_TRUTH_FILE}")
    sys.exit(1)


ground_truth = {}

with open(
    GROUND_TRUTH_FILE,
    "r",
    encoding="utf-8",
) as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        record = json.loads(line)

        pair_id = record["pair_id"]

        ground_truth[pair_id] = record


print()
print("Ground truth:")
print(
    f"  Records loaded: {len(ground_truth)}"
)


# ------------------------------------------------------------
# Discover dataset
# ------------------------------------------------------------

reference_files = sorted(
    REFERENCE_DIR.glob("*.png")
)

search_files = sorted(
    SEARCH_DIR.glob("*.png")
)

print()
print("Dataset:")
print("  Reference images:", len(reference_files))
print("  Search images   :", len(search_files))


# ------------------------------------------------------------
# Basic dataset checks
# ------------------------------------------------------------

dataset_failures = []

if len(reference_files) != 108:
    dataset_failures.append(
        f"Expected 108 reference images, found {len(reference_files)}"
    )

if len(search_files) != 108:
    dataset_failures.append(
        f"Expected 108 search images, found {len(search_files)}"
    )


reference_names = {
    path.name
    for path in reference_files
}

search_names = {
    path.name
    for path in search_files
}

missing_search = sorted(
    reference_names - search_names
)

missing_reference = sorted(
    search_names - reference_names
)

if missing_search:
    dataset_failures.append(
        f"Missing search counterparts: {missing_search}"
    )

if missing_reference:
    dataset_failures.append(
        f"Missing reference counterparts: {missing_reference}"
    )


# ------------------------------------------------------------
# Ground-truth correspondence checks
# ------------------------------------------------------------

for path in reference_files + search_files:

    pair_id = path.stem

    if pair_id not in ground_truth:

        dataset_failures.append(
            f"No ground-truth record for {pair_id}"
        )


print()
print("-" * 70)
print("1. DATASET STRUCTURE")
print("-" * 70)

if dataset_failures:

    print("FAIL")

    for failure in dataset_failures:
        print("  ", failure)

else:

    print("PASS")
    print("  108 reference images found")
    print("  108 search images found")
    print("  Matching filenames confirmed")
    print("  Ground-truth records confirmed")


# ------------------------------------------------------------
# Validate pixel scales from ground truth
# ------------------------------------------------------------

print()
print("-" * 70)
print("2. GROUND-TRUTH PIXEL SCALE VALIDATION")
print("-" * 70)

px_nm_by_pair = {}

scale_failures = []

for path in reference_files:

    pair_id = path.stem

    if pair_id not in ground_truth:
        continue

    record = ground_truth[pair_id]
    gt = record["ground_truth"]

    reference_px_nm = 1.0
    search_px_nm = float(gt["scale"])

    if not np.isfinite(search_px_nm) or search_px_nm <= 0:
        scale_failures.append(
            (
                pair_id,
                f"invalid search px_nm: {search_px_nm}"
            )
        )
        continue

    px_nm_by_pair[pair_id] = {
        "reference": reference_px_nm,
        "search": search_px_nm,
    }


if scale_failures:

    print("FAIL")

    for failure in scale_failures[:20]:
        print(" ", failure)

else:

    print("PASS")

    print(
        "  Reference px_nm = 1.0 nm/pixel"
    )

    search_scales = [
        value["search"]
        for value in px_nm_by_pair.values()
    ]

    print(
        f"  Search px_nm range = "
        f"{min(search_scales):.6f} - "
        f"{max(search_scales):.6f} nm/pixel"
    )

    print(
        "  Search pixel scale taken from ground_truth.jsonl"
    )


# ------------------------------------------------------------
# Storage
# ------------------------------------------------------------

all_results = {
    "reference": {},
    "search": {},
}

input_failures = []
output_failures = []


# ------------------------------------------------------------
# Process every image
# ------------------------------------------------------------

for capture_type, files in [
    ("reference", reference_files),
    ("search", search_files),
]:

    print()
    print("-" * 70)
    print(
        f"3. PROCESSING {capture_type.upper()} DATASET"
    )
    print("-" * 70)

    for index, path in enumerate(
        files,
        start=1,
    ):

        pair_id = path.stem

        # ----------------------------------------------------
        # Get correct pixel scale
        # ----------------------------------------------------

        if pair_id not in px_nm_by_pair:

            input_failures.append(
                (
                    capture_type,
                    path.name,
                    "missing pixel-scale ground truth",
                )
            )

            continue

        px_nm = px_nm_by_pair[
            pair_id
        ][capture_type]

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        try:

            image = load_image(path)

        except Exception as exc:

            input_failures.append(
                (
                    capture_type,
                    path.name,
                    f"load error: {exc}",
                )
            )

            continue

        # ----------------------------------------------------
        # Validate input
        # ----------------------------------------------------

        problems = validate_input(
            image
        )

        if problems:

            for problem in problems:

                input_failures.append(
                    (
                        capture_type,
                        path.name,
                        problem,
                    )
                )

            continue

        all_results[
            capture_type
        ][path.name] = {
            "input": image,
            "px_nm": px_nm,
            "outputs": {},
        }

        # ----------------------------------------------------
        # Generate every noise stratum
        # ----------------------------------------------------

        for noise_level in NOISE_LEVELS:

            rng = np.random.default_rng(42)

            try:

                output = apply_sem_chain(
                    image,
                    px_nm=px_nm,
                    params={
                        "preset": capture_type,
                        "noise_level": noise_level,
                    },
                    rng=rng,
                )

            except Exception as exc:

                output_failures.append(
                    (
                        capture_type,
                        path.name,
                        noise_level,
                        f"execution error: {exc}",
                    )
                )

                continue

            problems = validate_output(
                output,
                image,
            )

            if problems:

                for problem in problems:

                    output_failures.append(
                        (
                            capture_type,
                            path.name,
                            noise_level,
                            problem,
                        )
                    )

                continue

            all_results[
                capture_type
            ][path.name]["outputs"][
                noise_level
            ] = output

        if (
            index % 20 == 0
            or index == len(files)
        ):

            print(
                f"  processed "
                f"{index}/{len(files)}"
            )


# ------------------------------------------------------------
# Input validation result
# ------------------------------------------------------------

print()
print("-" * 70)
print("4. INPUT VALIDATION")
print("-" * 70)

if input_failures:

    print("FAIL")

    for failure in input_failures[:20]:
        print(" ", failure)

    if len(input_failures) > 20:
        print(
            f"  ... and "
            f"{len(input_failures) - 20} more"
        )

else:

    print("PASS")
    print("  All images readable")
    print("  All images converted to grayscale float32")
    print("  All pixels finite")
    print("  All pixels inside [0,1]")


# ------------------------------------------------------------
# Output contract validation
# ------------------------------------------------------------

print()
print("-" * 70)
print("5. OUTPUT CONTRACT VALIDATION")
print("-" * 70)

if output_failures:

    print("FAIL")

    for failure in output_failures[:20]:
        print(" ", failure)

    if len(output_failures) > 20:
        print(
            f"  ... and "
            f"{len(output_failures) - 20} more"
        )

else:

    print("PASS")
    print("  All outputs preserve shape")
    print("  All outputs are float32")
    print("  All outputs are finite")
    print("  All outputs remain inside [0,1]")


# ------------------------------------------------------------
# Reproducibility test
# ------------------------------------------------------------

print()
print("-" * 70)
print("6. REPRODUCIBILITY TEST")
print("-" * 70)

repro_failures = []

for capture_type in [
    "reference",
    "search",
]:

    for name, record in all_results[
        capture_type
    ].items():

        image = record["input"]
        px_nm = record["px_nm"]

        for noise_level in NOISE_LEVELS:

            out1 = apply_sem_chain(
                image,
                px_nm=px_nm,
                params={
                    "preset": capture_type,
                    "noise_level": noise_level,
                },
                rng=np.random.default_rng(1234),
            )

            out2 = apply_sem_chain(
                image,
                px_nm=px_nm,
                params={
                    "preset": capture_type,
                    "noise_level": noise_level,
                },
                rng=np.random.default_rng(1234),
            )

            if not np.array_equal(
                out1,
                out2,
            ):

                repro_failures.append(
                    (
                        capture_type,
                        name,
                        noise_level,
                    )
                )


if repro_failures:

    print("FAIL")

    for failure in repro_failures[:20]:
        print(" ", failure)

else:

    print("PASS")
    print(
        "  Same image + same seed = identical output"
    )


# ------------------------------------------------------------
# Different-seed stochasticity
# ------------------------------------------------------------

print()
print("-" * 70)
print("7. DIFFERENT-SEED STOCHASTICITY")
print("-" * 70)

stochastic_failures = []

stochastic_distances = {
    "reference": {
        level: []
        for level in NOISE_LEVELS
    },
    "search": {
        level: []
        for level in NOISE_LEVELS
    },
}


for capture_type in [
    "reference",
    "search",
]:

    for name, record in all_results[
        capture_type
    ].items():

        image = record["input"]
        px_nm = record["px_nm"]

        for noise_level in NOISE_LEVELS:

            out1 = apply_sem_chain(
                image,
                px_nm=px_nm,
                params={
                    "preset": capture_type,
                    "noise_level": noise_level,
                },
                rng=np.random.default_rng(42),
            )

            out2 = apply_sem_chain(
                image,
                px_nm=px_nm,
                params={
                    "preset": capture_type,
                    "noise_level": noise_level,
                },
                rng=np.random.default_rng(43),
            )

            distance = mae(
                out1,
                out2,
            )

            stochastic_distances[
                capture_type
            ][noise_level].append(
                distance
            )

            if distance <= 0.0:

                stochastic_failures.append(
                    (
                        capture_type,
                        name,
                        noise_level,
                    )
                )


if stochastic_failures:

    print("FAIL")

    for failure in stochastic_failures[:20]:
        print(" ", failure)

else:

    print("PASS")

    for capture_type in [
        "reference",
        "search",
    ]:

        print()
        print(
            f"  {capture_type.upper()}"
        )

        for level in NOISE_LEVELS:

            values = stochastic_distances[
                capture_type
            ][level]

            print(
                f"    {level:>6}: "
                f"mean MAE = "
                f"{np.mean(values):.6f}"
            )


# ------------------------------------------------------------
# Noise-strata statistics
# ------------------------------------------------------------

print()
print("-" * 70)
print("8. NOISE STRATA AGGREGATE TEST")
print("-" * 70)

strata_stats = {}

for capture_type in [
    "reference",
    "search",
]:

    strata_stats[capture_type] = {}

    print()
    print(capture_type.upper())

    for level in NOISE_LEVELS:

        means = []
        stds = []
        maes = []
        rmses = []

        for name, record in all_results[
            capture_type
        ].items():

            if level not in record["outputs"]:
                continue

            image = record["input"]
            output = record["outputs"][level]

            means.append(
                float(output.mean())
            )

            stds.append(
                float(output.std())
            )

            maes.append(
                mae(output, image)
            )

            rmses.append(
                rmse(output, image)
            )

        if not means:
            continue

        strata_stats[
            capture_type
        ][level] = {
            "mean": float(np.mean(means)),
            "std": float(np.mean(stds)),
            "mae": float(np.mean(maes)),
            "rmse": float(np.mean(rmses)),
        }

        stats = strata_stats[
            capture_type
        ][level]

        print(
            f"  {level:>6} | "
            f"mean={stats['mean']:.6f} | "
            f"std={stats['std']:.6f} | "
            f"MAE={stats['mae']:.6f} | "
            f"RMSE={stats['rmse']:.6f}"
        )


# ------------------------------------------------------------
# Noise-strata ordering test
# ------------------------------------------------------------

print()
print("-" * 70)
print("9. NOISE STRATA ORDERING")
print("-" * 70)

ordering_failures = []

for capture_type in [
    "reference",
    "search",
]:

    values = {
        level: np.mean(
            stochastic_distances[
                capture_type
            ][level]
        )
        for level in NOISE_LEVELS
    }

    # Current design:
    #
    # none == medium
    # low < medium
    # medium < high

    if not np.isclose(
        values["none"],
        values["medium"],
        rtol=0.0,
        atol=1e-7,
    ):

        ordering_failures.append(
            (
                capture_type,
                "none and medium should be equivalent",
            )
        )

    if not (
        values["low"]
        < values["medium"]
        < values["high"]
    ):

        ordering_failures.append(
            (
                capture_type,
                "expected low < medium < high",
            )
        )


if ordering_failures:

    print("FAIL")

    for failure in ordering_failures:
        print(" ", failure)

else:

    print("PASS")

    print(
        "  Noise severity increases as:"
    )

    print(
        "    low < medium < high"
    )

    print(
        "  none == medium by design"
    )


# ------------------------------------------------------------
# Search vs reference noise asymmetry
# ------------------------------------------------------------

print()
print("-" * 70)
print("10. SEARCH VS REFERENCE NOISE ASYMMETRY")
print("-" * 70)

asymmetry_failures = []

for level in [
    "low",
    "medium",
    "high",
]:

    ref_values = stochastic_distances[
        "reference"
    ][level]

    search_values = stochastic_distances[
        "search"
    ][level]

    ref_mean = float(
        np.mean(ref_values)
    )

    search_mean = float(
        np.mean(search_values)
    )

    print(
        f"  {level:>6}: "
        f"reference={ref_mean:.6f} | "
        f"search={search_mean:.6f}"
    )

    if search_mean <= ref_mean:

        asymmetry_failures.append(
            (
                level,
                ref_mean,
                search_mean,
            )
        )


if asymmetry_failures:

    print()
    print("FAIL")

    print(
        "  Search should remain noisier than "
        "reference at every level."
    )

else:

    print()
    print("PASS")

    print(
        "  Search remains noisier than "
        "reference at low/medium/high."
    )


# ------------------------------------------------------------
# Final summary
# ------------------------------------------------------------

print()
print("=" * 70)
print("FINAL VALIDATION SUMMARY")
print("=" * 70)

checks = {
    "Dataset structure": not dataset_failures,
    "Ground-truth scales": not scale_failures,
    "Input validation": not input_failures,
    "Output contract": not output_failures,
    "Reproducibility": not repro_failures,
    "Stochasticity": not stochastic_failures,
    "Noise ordering": not ordering_failures,
    "Search/reference asymmetry": not asymmetry_failures,
}


for name, passed in checks.items():

    status = "PASS" if passed else "FAIL"

    print(
        f"{name:<30}: {status}"
    )


hard_failures = [
    name
    for name, passed in checks.items()
    if not passed
]


print()

if hard_failures:

    print(
        "LARGE-SCALE VALIDATION: FAIL"
    )

    print()
    print("Failed checks:")

    for failure in hard_failures:

        print(
            f"  - {failure}"
        )

    sys.exit(1)

else:

    print(
        "LARGE-SCALE VALIDATION: PASS"
    )

    print()
    print(
        "All 216 R1-generated images passed the"
    )

    print(
        "SEM physics validation contract."
    )