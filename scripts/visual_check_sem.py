import sys
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
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

GROUND_TRUTH = DATASET / "ground_truth.jsonl"

OUTPUT_DIR = ROOT / "validation_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

NOISE_LEVELS = [
    "low",
    "medium",
    "high",
]

SEED = 42

# Number of image pairs to inspect
NUM_PAIRS = 4


# ------------------------------------------------------------
# Load ground truth
# ------------------------------------------------------------

def load_ground_truth(path):
    """
    Load ground_truth.jsonl using the actual DriftSense schema.

    Returns:
        dict keyed by pair_id
    """

    records = {}

    with open(path, "r", encoding="utf-8") as f:

        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            # ------------------------------------------------
            # Required fields
            # ------------------------------------------------

            if "pair_id" not in record:
                raise ValueError(
                    f"Ground-truth line {line_number} "
                    f"has no pair_id field"
                )

            if "reference_path" not in record:
                raise ValueError(
                    f"Ground-truth line {line_number} "
                    f"has no reference_path field"
                )

            if "search_path" not in record:
                raise ValueError(
                    f"Ground-truth line {line_number} "
                    f"has no search_path field"
                )

            if "ground_truth" not in record:
                raise ValueError(
                    f"Ground-truth line {line_number} "
                    f"has no ground_truth field"
                )

            if "scale" not in record["ground_truth"]:
                raise ValueError(
                    f"Ground-truth line {line_number} "
                    f"has no ground_truth.scale field"
                )

            # ------------------------------------------------
            # Extract information
            # ------------------------------------------------

            pair_id = record["pair_id"]

            reference_name = Path(
                record["reference_path"]
            ).name

            search_name = Path(
                record["search_path"]
            ).name

            search_px_nm = float(
                record["ground_truth"]["scale"]
            )

            records[pair_id] = {
                "pair_id": pair_id,
                "reference_name": reference_name,
                "search_name": search_name,
                "reference_px_nm": 1.0,
                "search_px_nm": search_px_nm,
            }

    if not records:
        raise ValueError(
            "ground_truth.jsonl contains no records"
        )

    return records


# ------------------------------------------------------------
# Image loading
# ------------------------------------------------------------

def load_image(path):
    """
    Load grayscale image as float32 in [0, 1].
    """

    image = np.asarray(
        Image.open(path).convert("L"),
        dtype=np.float32,
    )

    image /= 255.0

    return image


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    print()
    print("=" * 70)
    print("SEM VISUAL SANITY CHECK")
    print("=" * 70)

    # --------------------------------------------------------
    # Load ground truth
    # --------------------------------------------------------

    if not GROUND_TRUTH.exists():
        raise FileNotFoundError(
            f"Could not find:\n{GROUND_TRUTH}"
        )

    ground_truth = load_ground_truth(
        GROUND_TRUTH
    )

    print()
    print(
        "Ground-truth records:",
        len(ground_truth),
    )

    # --------------------------------------------------------
    # Discover images
    # --------------------------------------------------------

    reference_files = sorted(
        REFERENCE_DIR.glob("*.png")
    )

    search_files = sorted(
        SEARCH_DIR.glob("*.png")
    )

    if not reference_files:
        raise RuntimeError(
            f"No reference images found in:\n"
            f"{REFERENCE_DIR}"
        )

    if not search_files:
        raise RuntimeError(
            f"No search images found in:\n"
            f"{SEARCH_DIR}"
        )

    reference_map = {
        path.name: path
        for path in reference_files
    }

    search_map = {
        path.name: path
        for path in search_files
    }

    # --------------------------------------------------------
    # Match ground-truth records to actual image files
    # --------------------------------------------------------
    #
    # IMPORTANT:
    #
    # We do NOT assume reference filename == search filename.
    #
    # Instead:
    #
    # pair_id
    #    |
    #    +--> reference_name
    #    |
    #    +--> search_name
    #
    # This matches the actual ground_truth.jsonl structure.
    # --------------------------------------------------------

    valid_pairs = []

    for pair_id, record in ground_truth.items():

        reference_name = record["reference_name"]
        search_name = record["search_name"]

        if reference_name not in reference_map:
            continue

        if search_name not in search_map:
            continue

        valid_pairs.append(
            {
                "pair_id": pair_id,
                "reference_name": reference_name,
                "search_name": search_name,
                "reference_path": reference_map[
                    reference_name
                ],
                "search_path": search_map[
                    search_name
                ],
                "reference_px_nm": record[
                    "reference_px_nm"
                ],
                "search_px_nm": record[
                    "search_px_nm"
                ],
            }
        )

    # --------------------------------------------------------
    # Validate matching
    # --------------------------------------------------------

    if not valid_pairs:
        raise RuntimeError(
            "No valid image pairs were found.\n\n"
            "Ground truth records exist, but their "
            "reference_path/search_path files were not "
            "found in dataset/reference and dataset/search."
        )

    print(
        f"Matched image pairs: {len(valid_pairs)}"
    )

    if len(valid_pairs) != len(ground_truth):

        print(
            f"WARNING: "
            f"{len(ground_truth) - len(valid_pairs)} "
            f"ground-truth records could not be matched."
        )

    # --------------------------------------------------------
    # Select representative pairs
    # --------------------------------------------------------

    num_pairs = min(
        NUM_PAIRS,
        len(valid_pairs),
    )

    selected_pairs = valid_pairs[
        :num_pairs
    ]

    print(
        "Image pairs selected:",
        len(selected_pairs),
    )

    print()

    for pair in selected_pairs:

        print(
            f"  {pair['pair_id']}"
        )

        print(
            f"    reference: "
            f"{pair['reference_name']}"
        )

        print(
            f"    search:    "
            f"{pair['search_name']}"
        )

        print(
            f"    search px_nm: "
            f"{pair['search_px_nm']:.6f}"
        )

    # --------------------------------------------------------
    # Generate visual checks
    # --------------------------------------------------------

    for pair_index, pair in enumerate(
        selected_pairs,
        start=1,
    ):

        pair_id = pair["pair_id"]

        reference_path = pair[
            "reference_path"
        ]

        search_path = pair[
            "search_path"
        ]

        reference_px_nm = pair[
            "reference_px_nm"
        ]

        search_px_nm = pair[
            "search_px_nm"
        ]

        # ----------------------------------------------------
        # Load images
        # ----------------------------------------------------

        reference_image = load_image(
            reference_path
        )

        search_image = load_image(
            search_path
        )

        # ----------------------------------------------------
        # Generate SEM outputs
        # ----------------------------------------------------

        reference_outputs = {}
        search_outputs = {}

        for noise_level in NOISE_LEVELS:

            reference_outputs[
                noise_level
            ] = apply_sem_chain(
                reference_image,
                px_nm=reference_px_nm,
                params={
                    "preset": "reference",
                    "noise_level": noise_level,
                },
                rng=np.random.default_rng(
                    SEED
                ),
            )

            search_outputs[
                noise_level
            ] = apply_sem_chain(
                search_image,
                px_nm=search_px_nm,
                params={
                    "preset": "search",
                    "noise_level": noise_level,
                },
                rng=np.random.default_rng(
                    SEED
                ),
            )

        # ----------------------------------------------------
        # Plot
        # ----------------------------------------------------

        fig, axes = plt.subplots(
            4,
            2,
            figsize=(10, 16),
        )

        fig.suptitle(
            (
                f"SEM Visual Sanity Check\n"
                f"Pair: {pair_id}\n"
                f"Reference: "
                f"{reference_px_nm:.4f} nm/pixel | "
                f"Search: "
                f"{search_px_nm:.4f} nm/pixel"
            ),
            fontsize=14,
        )

        # ----------------------------------------------------
        # Row 1: Original
        # ----------------------------------------------------

        axes[0, 0].imshow(
            reference_image,
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )

        axes[0, 0].set_title(
            "Reference — Original"
        )

        axes[0, 1].imshow(
            search_image,
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )

        axes[0, 1].set_title(
            "Search — Original"
        )

        # ----------------------------------------------------
        # Rows 2–4: Noise levels
        # ----------------------------------------------------

        for row, noise_level in enumerate(
            NOISE_LEVELS,
            start=1,
        ):

            axes[row, 0].imshow(
                reference_outputs[
                    noise_level
                ],
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
            )

            axes[row, 0].set_title(
                f"Reference — {noise_level}"
            )

            axes[row, 1].imshow(
                search_outputs[
                    noise_level
                ],
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
            )

            axes[row, 1].set_title(
                f"Search — {noise_level}"
            )

        # ----------------------------------------------------
        # Formatting
        # ----------------------------------------------------

        for ax in axes.flat:
            ax.axis("off")

        plt.tight_layout(
            rect=[
                0,
                0,
                1,
                0.96,
            ]
        )

        output_path = (
            OUTPUT_DIR
            / f"sem_visual_check_{pair_index}.png"
        )

        fig.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight",
        )

        plt.close(fig)

        print()
        print(
            f"Saved: {output_path}"
        )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("VISUAL SANITY CHECK COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Generated {num_pairs} comparison figures."
    )

    print()
    print(
        "Output directory:"
    )

    print(
        f"  {OUTPUT_DIR}"
    )

    print()
    print(
        "Inspect the figures for:"
    )

    print(
        "  1. Low < Medium < High noise severity"
    )

    print(
        "  2. Search visibly noisier than Reference"
    )

    print(
        "  3. No geometric displacement"
    )

    print(
        "  4. No extreme clipping"
    )

    print(
        "  5. No pathological scan artifacts"
    )


if __name__ == "__main__":
    main()