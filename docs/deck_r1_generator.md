# Deck — R1's generator slides

Four slides covering the generator, to be folded into `docs/deck_outline.md`.
Written separately to avoid a whole-file collision with that document; R3 owns
the running order.

**Why these exist:** the 12-slide outline covers inference thoroughly and the
generator only in passing (slide 11, reproducibility). Scoring puts **30% on the
generator and augmentation with literature citations** and a bonus on RGB. As
drafted, roughly a third of the marks has no slide.

**Suggested placement:** A and B after slide 2 (they set up why the problem is
hard before the pipeline explains how it is solved); C alongside slide 11; D at
the end as the bonus.

Same rule as the rest of the deck: every number below is traceable to a tracked
file, named on the slide.

---

## A — The generator is the product

No real fab data exists for this problem, so the dataset *is* the deliverable
that everything else is measured against.

**324 pairs, fully crossed:**

```
2 architectures  x  2 anchoring  x  3 pose  x  3 noise  x  9 seeds  =  324
DRAM / FinFET      anchored /      none /     low /
                   unanchored      small /    medium /
                                   large      high
```

Balanced exactly: 162 per architecture, 162 per anchoring condition, 108 per
pose and per noise level.

**Domain randomisation, per pair:** pitch ±20%, linewidth ±15%, rotation ±5°
(`small`) or ±8° (`large`), scale ±3% / ±5%. The `large` pose stratum
deliberately exceeds the specification, so the dataset carries stress cases
beyond the range anyone tuned against.

**Every stratum answers a question:**

| Stratum | The question it isolates |
|---|---|
| anchored vs unanchored | is there aperiodic information at all? |
| noise low/medium/high | how much of the failure is dose? |
| pose none/small/large | how much is rotation and scale? |
| DRAM vs FinFET | does the method depend on the architecture? |

**Source:** `src/generate_dataset.py`, `dataset/dataset_manifest.json`.

> Speaker note: the unanchored half is unsolvable by construction. That is not
> an oversight — it is the control condition that makes the anchored number
> mean something.

---

## B — DRAM and FinFET are hard in different ways

**Figure:** `docs/dram_vs_finfet.png`

Left column is the reference at 1 nm/px; right is the search field at 10 nm/px,
in which the reference occupies about 100×100 of 1000×1000 pixels — 0.69% of the
area, and every part of it looks like every other part.

The two architectures fail differently, and the reason is contrast rather than
sampling. Through the search optic (10 nm/px, R2's 12 nm PSF):

| Feature | Period | Contrast retained |
|---|---|---|
| FinFET fin pitch | 72–108 nm | **58–78%** |
| DRAM pitch | 144–216 nm | **87–94%** |
| FinFET gate pitch | 336–504 nm | 97–99% |

Nothing aliases — Nyquist needs 2 px per period and even 72 nm clears it at 7.2.
What separates them is that FinFET carries *two* periodicities: a strongly
attenuated fine lattice on top of a nearly intact coarse one. That is the
leading explanation for its flatter correlation surfaces.

**Our dimensions are deliberately ~2–3× relaxed from current silicon**, and the
reason is quantitative: at DRAM's own ground rules (30–48 nm half-pitch, IRDS
2023 More Moore §5.1) the lattice reaches the search image at 45% contrast
before any noise, and the `high` stratum scales dose to 0.35. Tighter still and
the task stops being well-posed rather than merely hard.

**Source:** `docs/assumptions.md` §4; citations in `docs/citations_layout.md`.

---

## C — The ground truth is checkable, not asserted

**Figure:** `docs/ground_truth_overlay.png`

Panel 3 is the reference rebuilt from **only** the search image and the
ground-truth record — never the layout generator. A convention that is merely
self-consistent passes every internal check and fails this one. That is exactly
how a constant +0.5 px error was caught: it had been published in every record.

**The gate, at 48 pairs:** mean preferred offset `(-0.006, -0.054)` px over 42
unambiguous pairs, standard error ±0.045 / ±0.040.

It is powered rather than decorative:

- A planted **0.5 px** offset reads back as −0.417, about five standard errors
- A planted **0.3 px** offset — barely above the 0.2 px tolerance — is still caught
- Pairs pinned at the probe limit are **excluded**, not averaged in: an argmin at
  the edge of its range has run out of room rather than found a minimum
- Under-powered samples return **`inconclusive`**, not a pass. At 12 pairs the
  standard error is ~0.12 px against a 0.2 px tolerance, so the gate cannot tell
  a defect from scatter and says so

**Generation is byte-reproducible across platforms.** The held-out set's image
hash was published *before* anyone regenerated it; a macOS run reproduced
`d51df27b…` exactly against the Windows original. Pre-registered prediction, not
a file compared with itself.

**Source:** `src/generate_dataset.py` (`summarise_overlay`), `tests/test_geometry.py`,
`dataset_holdout/dataset_manifest.json`.

---

## D — Bonus: what visible light cannot see

**Figure:** `docs/rgb_optical_bonus.png`

The bonus asks for three channels with diffraction-limited blur. Working the
physics first changed the deliverable:

> **Visible light cannot resolve this layout.** Not "resolves it poorly" —
> transmits it at exactly zero contrast.

Incoherent imaging passes nothing above `f_c = 2·NA/λ`, so the finest surviving
period is `λ/(2·NA)`. At NA 0.95 that is **237 nm** in blue and **337 nm** in
red. FinFET fins (72–108 nm) and DRAM pitch (144–216 nm) are below it at any
wavelength and at NA 1.35 too. Only FinFET's gate pitch survives.

Measured on rendered layouts, the best channel retains **5.7%** of the geometry's
contrast on DRAM and **6.8%** on FinFET.

So the bonus is delivered as **a measured limit with a working renderer**, not an
accuracy table — a number on optical pairs would be a number about noise. Three
claims, all checkable from the code:

1. Optical alignment cannot address this problem at these dimensions
2. The wavelength dependence is real: cutoff scales as `1/λ`, so blue carries
   strictly more structure than red
3. This is *why* the SEM chain exists — the same layout retains 45–94% through
   the SEM optic and ~6% optically

**Source:** `src/optical.py`, `tests/test_optical.py`, `docs/rgb_optical_bonus.md`.

> Speaker note: if asked why not a Gaussian blur — a Gaussian never reaches zero,
> so a matcher could lock onto a lattice at 0.1% contrast that no real objective
> delivers. The circular-aperture OTF is exactly zero above cutoff, which is what
> makes "cannot resolve" a claim rather than a phrase.

---

## Number provenance

| slide | claim | file |
|---|---|---|
| A | 324 pairs, stratum balance, randomisation ranges | `dataset/dataset_manifest.json`, `src/generate_dataset.py` |
| B | contrast retained per feature | `docs/assumptions.md` §4 |
| B | DRAM ground rules 30–48 nm half-pitch | `docs/citations_layout.md` [5a], IRDS 2023 More Moore §5.1 p28 |
| B | Intel fin pitch 60 nm, gate pitch 90 nm | `docs/citations_layout.md` [3], Jan et al., IEDM 2012 |
| C | overlay gate figures, planted-offset detection | `src/generate_dataset.py`, `tests/test_geometry.py` |
| C | cross-platform hash reproduction | `dataset_holdout/dataset_manifest.json` |
| D | Abbe limits, retained contrast | `src/optical.py`, `tests/test_optical.py` |
