# Failure analysis

Where the system fails, why, and which failures are defects versus correct
readings of a genuinely underdetermined problem.

All numbers come from `results/uniqueness_on.csv` over 108 stratified pairs
(2 architectures x 2 anchor states x 3 pose conditions x 9 seeds), 1 px success
tolerance. Reproduction command in section 6.

**Every figure on this page is measured on physics-free renders.** R2's
`apply_sem_chain` is still an identity passthrough, so the noise stratum is a
label on clean images. Nothing here should be read as performance under SEM
noise; it is an upper bound that the physics chain will move.

---

## Headline

| stratum | n | success@1px | median error (px) | mean PSR |
|---|---|---|---|---|
| **overall** | 108 | **0.417** | 115.4 | 2.46 |
| anchored | 54 | **0.833** | **0.028** | 2.76 |
| unanchored | 54 | **0.000** | 455.3 | 2.17 |

The overall figure is the less informative of the two. It averages a stratum the
system solves to a hundredth of a pixel with one that is not solvable at all.

---

## 1. The unanchored stratum is a ceiling, not a defect

Fifty-four of the 108 cases score zero, and no amount of tuning moves them.

An unanchored reference is a window onto a purely periodic region — a word-line
array with no aperiodic feature in frame. Correlating a periodic template
against a periodic field produces a *lattice* of near-identical peaks rather
than one peak. On a bare lattice, 903 candidate positions score exactly the
maximum. The correlation evidence does not identify a position, because there
is no information in the images that distinguishes those placements.

This is confirmed by construction rather than inferred: R1's generator asserts
that an unanchored layout carries an empty anchor list. The stratum is built to
contain no distinguishing feature.

**The ceiling on this dataset is therefore 0.500.** Measured against that
ceiling, the system is at 0.833 of what is achievable.

The correct response to an unsolvable case is to answer and flag, not to
succeed. Every unanchored case escalates to the ambiguous tier, returns the
centre-prior answer mandated by the problem statement, and carries a low
confidence. None returns a confident wrong answer. That is designed behaviour.

---

## 2. Uniqueness weighting: what it fixed and what it could not

Stage 4a weights the matched filter toward reference regions that identify a
position uniquely, measured by the peak-to-sidelobe ratio of each tile's own
autocorrelation.

| | before | after |
|---|---|---|
| overall | 0.389 | **0.417** |
| anchored | 0.778 | **0.833** |
| unanchored | 0.000 | 0.000 |

The gain lands entirely on the anchored stratum, which is exactly the
prediction: weighting can only concentrate the filter on distinguishing
features that are *present*. Where no anchor is in frame the map correctly
returns near-flat, masked correlation degrades to unmasked, and nothing changes.

That degradation is deliberate. The map is absolute-scale and is **not**
renormalised per image. Rescaling so the best tile reads 1.0 would, on a
reference with no anchor, promote whichever tile is marginally least periodic
by chance into a false anchor — manufacturing a confident wrong answer out of
noise. Returning a flat map keeps the failure visible.

---

## 3. PSR does not separate correct from incorrect

The peak-to-sidelobe ratio is the detection statistic gating escalation. It
currently carries little discriminative signal:

| stratum | mean PSR |
|---|---|
| anchored (83% correct) | 2.76 |
| unanchored (0% correct) | 2.17 |

The distributions overlap heavily. In earlier per-case measurement the highest
PSR in the set belonged to a *wrong* answer.

The cause is structural rather than a coding error. PSR compares the winning
peak against the surrounding surface, but on a periodic layout the sidelobe
region contains genuine lattice peaks rather than noise. The background the
statistic normalises against is itself signal, which inflates the sidelobe
standard deviation and compresses PSR toward the same value regardless of
correctness.

**Consequence:** with accept thresholds at 8.0 (fast) and 4.0 (robust) and
observed PSR never exceeding about 3.4, no case is ever accepted early. Every
pair escalates to the ambiguous tier. Slow, but never falsely confident — and
the thresholds are deliberately left untuned, because lowering them to buy
speed would buy it by making wrong answers confident.

One measurement bug was found and fixed in this pass: sidelobe statistics
excluded a **Euclidean disc** around the peak while peak extraction suppresses
over a **Chebyshev square**. The two regions disagree at the corners, so part of
the peak's own shoulder was counted as background, biasing PSR low. Now matched.

---

## 4. Secondary failure modes

**Architecture.** DRAM 0.463 vs FinFET 0.370. FinFET layouts carry finer, more
regular structure, which survives the 10x decimation less well and leaves fewer
distinguishing features in the template.

**Pose.** Unrotated 0.500, small rotation 0.389, large rotation 0.361.
Degradation with pose is expected: rotation is estimated before template
construction, and residual angular error compounds through correlation. Note
that R2's pose estimator still returns nominal pose with zero quality, so
rotated pairs are currently matched at nominal — this number will move when
pose estimation lands.

**PSF estimation declines on periodic layouts.** The estimator fits a Gaussian
rolloff to the radial power spectrum and gates on fit quality. On a
lattice-dominated spectrum it correctly refuses and returns the documented
default. Correct behaviour, but it means the PSF is effectively never measured
on this data.

**Ties never fire.** Tie rate is 0.000 and the tied-set size is always 1, both
consequences of TIE_SIGMA = 0.0. That value is correct — at the previous 1.0 a
sigma width was being passed as a raw score tolerance, tying all 30 candidates
on 102 of 105 cases and handing the answer to the centre prior blind. Fixing it
recovered accuracy by close to an order of magnitude. The side effect is that
the tie-break flag
carries no information as a confidence feature.

**Latency.** Median 351 ms with uniqueness weighting against 105 ms without.
The map depends only on the reference and is currently recomputed per case; it
should be cached alongside the template.

---

## 5. What would move the numbers

In order of expected value:

1. **Anchors in the unanchored stratum.** The 0.500 ceiling is a property of the
   dataset, not the algorithm. Nothing else can lift overall accuracy past it.
2. **A PSR variant robust to periodic backgrounds** — estimating sidelobe
   statistics from the aperiodic residual rather than the raw surface — would
   make early acceptance possible and cut latency substantially.
3. **R2's physics chain.** Every number here is measured on unrealistically
   clean data and must be re-measured once noise, PSF, and edge brightening land.
4. **Caching the uniqueness map** per reference.

---

## 6. How to reproduce

Run: `uv run python -m src.evaluate --data dataset_full --out results/uniqueness_on.csv`

Per-case rows, including psr, n_tied, tie_break_used, uniqueness_score and the
failure mode, are in the CSV. Dataset integrity is verifiable against
`dataset_full/dataset_manifest.json`.
