# Deck outline — Drift-Sense, PS-02

Twelve slides. Every number below is traceable to a tracked file, named on the
slide it appears on, so any figure can be checked live without anyone
reconstructing where it came from.

**Status:** draft for review. Numbers verified against `results/*.csv` on
2026-08-13. Regenerate the figures before the final build if any harness re-runs.

---

## Rules for anyone editing this deck

Four ways to be wrong in front of a judge, all of which have already happened
once in this repo:

1. **Never quote a 108-set number.** The 108-pair set was physics-free and is
   superseded. Anchored 77.8%, 105 ms and 0.01 px sub-pixel are all from it.
   The current set is 324 pairs with live SEM physics.
2. **Never quote the all-pairs accuracy without the anchored one.** 0.426
   averages a stratum we solve to 0.042 px with one that is unsolvable by
   construction. Lead with anchored 0.852 and explain the ceiling.
3. **Never quote a latency without its machine.** 212 ms is this Mac; the same
   324 pairs read 356 ms on R4's laptop with identical accuracy.
   `results/full_324.meta.json` records the machine for the tracked run.
4. **Never say PSR is capped near 3.4.** That is the 108-set figure. On 324 the
   maximum is 12.314. Opening the CSV falsifies the claim in one step.

---

## 1 — The problem

A wafer-inspection tool revisits the same microscopic site. The stage lands
roughly a micrometre off, and because every die is a printed copy of the same
design, **the wrong location looks identical to the right one.**

Given the zoomed-in `reference` from visit one and a zoomed-out `search` of
wherever the stage actually landed, return the reference's centre in search
pixels. That offset times the pixel size is the stage correction.

> Speaker note: the second sentence is the whole problem. Everything after this
> slide follows from "identical by design".

## 2 — What comes in, what goes out

```
reference : uint8 (1000, 1000)   1 nm/px  ->  1 um field
search    : uint8 (1000, 1000)  10 nm/px  -> 10 um field
(x, y)    : float — centre of the match, in SEARCH pixels
```

Two separate physical captures: independent noise, different point-spread
function, possibly different detector gain, small rotation and scale error.
The reference occupies about 100x100 px inside the search image.

**Source:** `README.md`, `src/config.py`.

## 3 — Why it is hard: a lattice, not a peak

Correlating a periodic template against a periodic field produces a *lattice of
near-identical peaks*, not one peak. For a 40 nm pitch the search image spans
about 250 lattice cells and the template 25 — on the order of 50,000 placements
that look essentially the same.

**Recall is easy. Picking the right one is the problem.**

> Visual: one correlation surface, showing the lattice of maxima. This slide
> earns the rest of the deck.

## 4 — The pipeline

```
Stage 0  normalise
Stage 1  pose            rotation and scale, Fourier/log-polar
Stage 2  template        rotate, PSF-match, area-average
Stage 3  ZNCC surface    FFT matched filter
Stage 3b peaks + NMS     one candidate per lattice cell
Stage 4  disambiguate    uniqueness weighting, PSR, centre tie-break
Stage 5  subpixel        upsampled-DFT phase correlation
Stage 6  confidence      calibrated scalar + escalation flag
```

Compute escalates rather than being spent up front: `fast` assumes nominal pose;
`robust` and `ambiguous` are entered only when the detection statistic says the
cheap path was not enough.

## 5 — The idea that does the work

**Not all of the reference is equally informative.** Periodic regions match
everywhere; aperiodic ones match in one place. So weight the matched filter
toward the parts that can actually localise.

The score is a tile's autocorrelation sidelobe. A signal's autocorrelation *is*
its ambiguity function under matched filtering (Woodward, 1953), so this
measures positional ambiguity directly rather than by proxy.

One design decision matters more than the rest: **the scale is absolute, never
per-image normalised.** Rescaling so the best tile reads 1.0 would promote the
least-periodic tile of an unanchored reference into a false anchor. A flat map
degrades to unweighted correlation and escalation fires — which is the correct
behaviour when there is nothing to anchor to.

**Source:** `src/uniqueness.py`, `docs/citations.md` §3.

## 6 — Results

| | plain NCC | full pipeline |
|---|---|---|
| **anchored** (n=162) | 0.753 | **0.852** |
| unanchored (n=162) | 0.000 | 0.000 |
| all pairs (n=324) | 0.377 | 0.426 |
| median error, anchored | 0.454 px | **0.042 px** |
| latency, median / p95 | | 212 / 215 ms *(this Mac)* |

324 stratified pairs: 2 architectures x 2 anchor states x 3 pose conditions x
3 noise strata x 9 seeds, at a 1 px (10 nm) tolerance, with live SEM physics.

**Source:** `results/full_324.csv`, `results/baseline_324.csv`, machine in
`results/full_324.meta.json`. The README's table is generated from these files
and a test fails if it drifts.

## 7 — The zero is a ceiling, not a failure

An unanchored reference is a window onto a purely periodic region. **The
generator asserts that such a layout carries an empty anchor list** — the
stratum is built to contain no distinguishing feature.

- Ceiling on this dataset is therefore **0.500**. We are at **0.852 of what is
  achievable.**
- The true peak is absent from the top 30 candidates in **160 of 162**
  unanchored cases. Not a near miss — the information is not there.
- All 162 escalate, return the mandated centre-prior answer, and carry the
  low-confidence flag.

> Speaker note: if a judge pushes on the 0.426, this is the slide. Do not
> apologise for the zero — explain why a correct system must produce it.

## 8 — What each stage actually buys

Anchored stratum, n=162:

| stage | accuracy | median error |
|---|---|---|
| ZNCC only | 0.753 | 0.454 px |
| + uniqueness weighting | 0.833 | 0.456 px |
| + candidate selection | 0.833 | 0.456 px |
| + sub-pixel | **0.852** | **0.042 px** |

**Weighting is the entire disambiguation gain (+0.080). Selection buys exactly
0.000** — with `TIE_SIGMA = 0.0` exact ties never occur on this dataset, so the
mandated tie-break is correct and never fires. Sub-pixel buys +0.019 and an 11x
error reduction; 0.454 px is the integer-peak quantisation floor.

**Source:** `results/ablation_324.csv`. Cite the anchored stratum — across all
324 the unanchored half dilutes the effect to invisibility.

## 9 — The honest slide

**Of the 324 cases, the system was confident about 2. One of those was wrong**
(`finfet_anchored_pose-large_0226`, 2.28 px error, PSR 8.13 against an accept
threshold of 8.0). The other 322 escalated and were flagged.

That is not a bug we failed to fix — it is why the thresholds are deliberately
untuned. **PSR does not separate correct from incorrect on periodic layouts**,
because the sidelobe region it normalises against contains genuine lattice peaks
rather than noise. Lowering the thresholds would buy speed by manufacturing
confident wrong answers.

> Speaker note: lead with this before a judge finds it. A team that reports its
> one confident error is more credible on the other 323.

## 10 — Where the remaining errors live

Anchored stratum, by condition:

| axis | best | worst |
|---|---|---|
| pose | none 0.963 | **large 0.667** |
| noise | high 0.889 | low/medium 0.833 |
| architecture | DRAM 0.889 | FinFET 0.815 |

**Pose is the dominant axis, not noise.** Rotation and scale estimation returns
nominal pose today, so large-pose pairs are matched at the wrong orientation.
Noise barely moves the result — ZNCC is normalised, and the accuracy at the
*high* stratum is the best of the three.

**Source:** `results/full_324.csv`.

## 11 — Reproducibility

- **Seed-reproducible generation, verified across platforms.** The holdout's
  image and file tree hashes were pre-registered before generation and a Mac run
  reproduced both exactly against the Windows values.
- **Clean-room check** (`scripts/clean_room_check.sh`): unpacks `git archive` —
  what a grader unzips, no venv, no dataset, no artefacts — resolves from the
  lockfile, runs 497 tests, the linters, and the CLI. It found five stale CSVs
  shipping at the archive root; they are gone.
- **The README's numbers are generated** from the tracked CSVs, with the machine
  attached, and a test fails when they drift.
- **`docs/citations.md`** tags every parameter: cited mechanism *and* magnitude,
  cited mechanism with our number, or declared assumption. Most of the SEM chain
  is the last two, and the document says so.

## 12 — What we would do next

1. **Pose estimation.** The single largest accuracy gain available — slide 10.
2. **Re-fit the confidence calibrator** now that `uniqueness_score` populates.
3. **Geometric scan distortion.** Our SEM model is intensity-only, so the
   dataset is easier than a real tool by exactly the amount of its drift. First
   thing to add before instrument data.

---

## Number provenance

| slide | claim | file |
|---|---|---|
| 6 | all headline accuracy and latency | `results/full_324.csv`, `results/baseline_324.csv`, `results/full_324.meta.json` |
| 7 | 160 of 162 top-30 misses | `results/recall_324.csv` |
| 8 | stage ablation | `results/ablation_324.csv` |
| 9 | the confident error | `results/full_324.csv`, row `finfet_anchored_pose-large_0226` |
| 10 | per-condition splits | `results/full_324.csv` |
| 11 | clean-room, hashes | `scripts/clean_room_check.sh`, `dataset_holdout` manifest |
| 5, 11 | citation tiers | `docs/citations.md` |
