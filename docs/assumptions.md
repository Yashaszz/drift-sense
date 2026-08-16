# Assumptions — generator geometry and sampling

Owner: R1. Every number the generator uses to place a shape or sample a pixel is
listed here with where it lives, what it feeds, and what justifies it.

The point is not to make the dataset look authoritative. It is that a judge
reading `layouts.py` can ask "why 180 nm?" and get an answer that is either a
citation, a derivation, or an honest "we chose this, here is the consequence if
it is wrong". A number with none of those three is a defect.

Physics parameters — edge brightening, PSF, shot and read noise, scan artifacts
— are R2's. They are **not yet cited**: `src/sem_physics.py` states at line 6
that its presets "must be tuned against the evaluation strata and replaced (or
supported) by literature citations before final submission", and that remains
open. This file covers geometry and sampling only, and does not stand in for it.

Layout references are collected in `docs/citations_layout.md`, to be folded into
the project-wide `citations.md` as its layout section.

## How to read the Source column

| Class | Meaning |
|---|---|
| **Spec** | Given in the problem statement. Not ours to justify, but ours to state. |
| **Derived** | Follows from another number. The derivation is the justification; no citation applies. |
| **Literature** | A real process dimension we chose to imitate. **Needs a citation.** |
| **Engineering** | Our choice, defensible by consequence rather than by source. |

Only the **Literature** rows need an external reference. Everything else needs a
sentence, and most of those sentences are already in the code.

---

## 1. Imaging geometry — Spec

| Constant | Value | Where | Feeds |
|---|---|---|---|
| `REF_PX_NM` | 1.0 nm/px | `config.py:33` | Reference sampling; the "100x" optic |
| `SEARCH_PX_NM` | 10.0 nm/px | `config.py:36` | Search sampling; the "10x" optic |
| `NOMINAL_SCALE` | 10.0 | `config.py:39` | Reference-to-search decimation ratio |
| `EXPECTED_IMAGE_SHAPE` | (1000, 1000) | `config.py:46` | Both image sizes |
| `OUT_SIZE` | 1000 px | `generate_dataset.py:98` | Rendered edge length |

**Source:** stated in the problem specification. Cite the problem statement by
section, not a paper.

> **TODO:** quote the exact line from the problem statement that fixes the two
> pixel sizes, so the claim is checkable rather than asserted.

These are the reason the whole task is hard: a 1000 px reference at 1 nm/px sees
1 µm, and the search image at 10 nm/px sees 10 µm, so the reference occupies
0.69% of the search area.

---

## 2. Die region and framing — Derived / Engineering

| Constant | Value | Where | Justification |
|---|---|---|---|
| `EXTENT_NM` | 12 000 nm | `generate_dataset.py:95` | **Derived.** Must exceed the 10 µm search field so a crop never runs off the layout. 12 µm gives 2 µm of margin. |
| `ANCHOR_SPAN_FRACTION` | 0.34 | `generate_dataset.py:154` | **Derived.** Half-width of the anchor placement box as a fraction of the reference FOV. Expressed as a fraction rather than nanometres so it tracks `out_size`; a fixed nm value silently assumes a 1000 px reference and puts anchors outside the crop at any other size. |
| `DEFAULT_ANCHOR_HALF_SPAN_NM` | 340.0 nm | `layouts.py:42` | **Derived.** `1000 px x 1 nm/px x 0.34`. The default exists so `layouts.py` is usable standalone; the generator always passes an explicit value. |
| `DEFAULT_SUPERSAMPLE` | 4 | `render.py:53` | **Engineering.** Area-averaged 4x4 per output pixel. Consequence if wrong: too low aliases edges and biases the sub-pixel ground truth; higher costs render time with no measurable accuracy gain. |

No citation applies to any of these. The derivation *is* the justification, and
each is already stated in the code.

---

## 3. Domain randomisation — Engineering, sourced to the team spec

| Constant | Value | Where | Justification |
|---|---|---|---|
| `PITCH_TOLERANCE` | ±20% | `generate_dataset.py:162` | Work-split document, Phase 2 |
| `WIDTH_TOLERANCE` | ±15% | `generate_dataset.py:165` | Work-split document, Phase 2 |
| `POSE_RANGES["small"]` | ±5°, ±3% | `generate_dataset.py:101` | Work-split document, Phase 2 baseline |
| `POSE_RANGES["large"]` | ±8°, ±5% | `generate_dataset.py:101` | **Engineering.** Deliberately exceeds the spec so the dataset carries stress cases beyond the range anyone tunes against. |

**Honest framing:** these are our own planning document, not literature. Say so.
"We chose ±20% to exercise the matcher across process variation" is defensible.
"±20% is the industry figure" is not, unless sourced.

> **TODO (optional, upgrades the claim):** real process variation is specified
> as 3σ CD tolerance in published process-control literature. If a figure is
> found, state ours as "wider than typical 3σ CD variation, deliberately" and
> cite it. If not, keep the honest version — do not imply a source that is not
> there.

---

## 4. Layout dimensions — deliberately relaxed, not a node replica

**These are the numbers a judge will ask about, and the honest answer is not a
citation.** Our dimensions do not match any production node. They are larger by
**1.5x to 5x depending on which dimension you compare** — fin pitch is the
tightest at 1.5x against Intel 22 nm, gate pitch the loosest at 4.7x — and that
is a deliberate consequence of the sampling the problem statement fixes. Claiming a node reference for them would be misattribution:
the source would not support the number, and R3 audits citations against the
code.

### What we use

| Architecture | Parameter | Nominal | Range in data |
|---|---|---|---|
| DRAM | `pitch_nm` | 180 nm | 144–216 |
| DRAM | `line_width_nm` | 40 nm | 34–46 |
| DRAM | `via_nm` | 60 nm | 51–69 |
| FinFET | `fin_pitch_nm` | 90 nm | 72–108 |
| FinFET | `fin_width_nm` | 24 nm | 20.4–27.6 |
| FinFET | `gate_width_nm` | 13 nm | 11.05–14.95 |
| FinFET | `gate_pitch_nm` | 420 nm | 336–504 |

Defined at `generate_dataset.py:143` and `:146`; ranges follow from
`PITCH_TOLERANCE` and `WIDTH_TOLERANCE` in section 3.

### What real silicon uses

| Process | Fin pitch | Gate pitch (CPP) | Source | Status |
|---|---|---|---|---|
| Intel 22 nm SoC | **60 nm** | **90 nm** (108 relaxed) | Jan et al., IEDM 2012 | **confirmed from primary** |
| DRAM, current | — | **60–96 nm pitch** (30–48 nm half-pitch) | IRDS 2023 More Moore §5.1 p28 | **confirmed from primary** |

Earlier drafts also listed Intel 14 nm (42 nm fin pitch) and 22FFL (45 nm /
~108 nm). Both came from search summaries and WikiChip refused connections on
three attempts, so they have been **dropped rather than carried unverified**.
The argument does not need them: the two rows above are primary, and both are
already tighter than anything we generate.

The Intel 22 nm paper also gives M1 pitch 90 nm and gate lengths 30/34/40 nm.
IRDS adds that some DRAM lines require ~20 nm half-pitch, and defines the cell
size factor `a = [cell size]/[half pitch]^2`, 6F2 today against a 4F2 limit.

> **Verification status:** every real-silicon figure above was extracted from a
> primary PDF and is quoted verbatim in `docs/citations_layout.md`. Nothing
> unverified remains in this section. The Intel table rows (fin, M1, Lgate) came
> from table extraction rather than prose, so re-read that table visually before
> the deck; the two prose figures are quoted exactly. Every number here is used
> only as *contrast* to our values, so an error weakens the framing without
> corrupting the dataset.

### Why we relaxed them — the actual justification

The problem statement fixes the search capture at 10 nm/px. Combined with R2's
search PSF (σ = 12 nm), a Gaussian MTF of `exp(-2π²σ²/Λ²)` gives the contrast
surviving into the search image:

| Feature | Period | px/period | Contrast retained |
|---|---|---|---|
| Intel 22 nm fin pitch *(confirmed)* | 60 nm | 6.0 | **45.4%** |
| Intel 22 nm gate / M1 pitch *(confirmed)* | 90 nm | 9.0 | 70.4% |
| IRDS DRAM, 30 nm half-pitch *(confirmed)* | 60 nm | 6.0 | **45.4%** |
| IRDS DRAM, 48 nm half-pitch *(confirmed)* | 96 nm | 9.6 | 73.5% |
| **Ours, FinFET fin (min)** | 72 nm | 7.2 | **57.8%** |
| **Ours, FinFET fin (nom)** | 90 nm | 9.0 | 70.4% |
| **Ours, FinFET fin (max)** | 108 nm | 10.8 | 78.4% |
| **Ours, DRAM pitch (min)** | 144 nm | 14.4 | 87.2% |
| **Ours, DRAM pitch (max)** | 216 nm | 21.6 | 94.1% |

Every confirmed real dimension lands at **45–74% contrast**, against our
**58–94%**. At the tighter end of the DRAM ground rules — 30 nm half-pitch, a
60 nm pitch — the lattice reaches the search image at 45% contrast *before any
noise is added*. Under the `high` stratum, where dose is scaled to 0.35 of R2's
search preset, that margin is thin. Tighter still -- and leading-edge FinFET is
tighter still -- the lattice falls to or below the shot-noise floor entirely.
The task would not be hard, it would be ill-posed, and every unanchored FinFET
pair would fail for reasons that say nothing about the matcher.

Relaxing keeps the periodic structure resolvable while preserving the ordering
that matters: our FinFET fins (58–78%) still sit below our DRAM (87–94%), so
FinFET remains the harder architecture for the reason we claim rather than by
accident.

**The one-sentence version for the deck:** *dimensions are relaxed 1.5–5x
from current silicon so periodic structure survives the 10 nm/px search optic
the problem specifies; at real DRAM ground rules the lattice reaches the search
image at 45% contrast and at leading-edge fin pitch at 20%, where the task stops
being well-posed rather than merely hard.*

Nyquist is not the binding constraint anywhere — it needs ≥2 px/period and even
42 nm clears it at 4.2 — so nothing in the dataset aliases. Contrast, not
sampling, is what separates the architectures.

### Consequence if this choice is wrong

If a judge considers the relaxation unrealistic, the affected claim is
"architecture-agnostic across representative geometry", not the accuracy
numbers: ground truth, the coordinate convention and the matcher are all
independent of absolute feature size. The fix would be a second dataset at true
node dimensions, which the generator supports today by changing two dicts.

---

## 5. Ground-truth convention — Spec / Derived

| Constant | Value | Where | Justification |
|---|---|---|---|
| `ORIGIN_TOP_LEFT` | True | `config.py:60` | **Spec.** Image convention. |
| `X_AXIS_IS_COLUMN` | True | `config.py:63` | **Spec.** |
| `PIXEL_CENTRE_AT_INTEGER` | True | `config.py:66` | **Spec/Derived.** Pixel *i* centre is at coordinate *i*, so a 1000 px image has centre base 499.5, not 500. A published revision once used 500 and capped team accuracy at ~1 px. |
| `OVERLAY_BIAS_TOLERANCE_PX` | 0.2 px | `generate_dataset.py:62` | **Derived from measurement.** Set from the measured noise floor: a known-good dataset lands within ~0.10 px of zero, a planted 0.5 px offset reads back as ~-0.46. 0.2 sits between them. |

No literature applies. These are conventions plus one empirically calibrated
threshold, and the calibration is recorded beside the constant.

---

## 6. Seeds and reproducibility — Engineering

| Constant | Value | Where | Justification |
|---|---|---|---|
| `DEFAULT_SEED` | 20260807 | `generate_dataset.py:92` | Frozen base seed for the shipped set. Pair *i* uses `base_seed + i`. |
| Held-out seed | 389722107 | not in code | Drawn with `secrets`, deliberately unpredictable and outside the tuning seed range, so no pair is shared with the shipped set. |

Generation is reproducible from seed: an independent regeneration on macOS
reproduced the held-out set's `image_tree_sha256` (`d51df27b…`) against a value
published in advance from Windows. Pixel-level cross-platform reproducibility is
therefore established by pre-registered prediction rather than by comparing a
file against itself.

**Byte-level was never confirmed, and is deliberately not claimed.** Recorded
2026-08-16, closing the TODO that stood here.

The 2026-08-12 macOS run compared `image_tree_sha256` only. It could not have
done otherwise: `verify_dataset` checks the image-tree hash and never reads
`file_tree_sha256` (`generate_dataset.py:850`), and the generator CLI prints
only the image hash (`generate_dataset.py:1362`), so `dcdcb969…` never appeared
in any output an operator saw. It appears exactly once in this repository — in
the TODO that used to sit here, asking for it.

It is also not a defect that it is unconfirmed. `file_tree_hash` is a
*diagnostic*, by its own docstring: "same pixels with different file hashes
means the encoders differ and the data is fine". `compare_manifests` agrees,
classifying that case as "identical pixels, different PNG encoding — harmless,
the data matches". PNG bytes are zlib-compressed, and the zlib build ships
inside each platform's Pillow wheel, so byte-identity across platforms is not
merely unmeasured but not expected.

What *is* established:

| claim | status |
|---|---|
| same platform, same seed → identical pixels **and** bytes | verified, two runs |
| cross-platform → identical pixels (`d51df27b…`) | verified by pre-registered prediction |
| cross-platform → identical PNG bytes (`dcdcb969…`) | **not claimed**, not measured |

Three documents previously stated the byte-level claim as done. They now say
pixel-level, which is what the evidence supports and what every downstream
number actually needs.

> **If you want to close the gap properly:** print `file_tree_sha256` alongside
> the image hash in the generator CLI, then regenerate on a second platform.
> Until the CLI surfaces it, the comparison stays impractical rather than
> merely undone.

---

## Open items

- [ ] Quote the problem statement lines fixing `REF_PX_NM` and `SEARCH_PX_NM`
- [ ] Source the seven DRAM/FinFET dimensions, or replace each with an explicit
      engineering rationale
- [ ] Optional: source a 3σ CD variation figure to upgrade the tolerance claim
- [x] Record the byte-level reproducibility result — done 2026-08-16: not
      confirmed, not claimed, and not expected to hold (section 6)
