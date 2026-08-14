# Failure analysis

Where the system fails, why, and which failures are defects versus correct
readings of a genuinely underdetermined problem.

All numbers come from `results/full_324.csv` and `results/baseline_324.csv`
over **324 stratified pairs** (2 architectures x 2 anchor states x 3 pose
conditions x 3 noise strata x 9 seeds), 1 px success tolerance. Reproduction
commands in section 8.

**These renders carry real SEM physics.** R2's `apply_sem_chain` is no longer an
identity passthrough — the noise strata modify Poisson shot noise and read
noise, with geometry-affecting parameters held constant across strata. Earlier
revisions of this document were measured on physics-free renders at 108 pairs;
**none of those figures appear here, and none should be quoted alongside these.**

---

> **Revised 14 August, after R2's rotation estimator landed.** Every figure
> below is re-measured with pose live. The previous revision reported anchored
> 0.852 at 212 ms; those numbers described a pipeline that assumed nominal pose
> and no longer exists. Sections 4 and 6 are marked where they still describe
> the nominal-pose behaviour deliberately.

## Headline

| stratum | n | plain NCC | full pipeline | median error (px) |
|---|---|---|---|---|
| **overall** | 324 | 0.377 | **0.469** | 104.3 |
| anchored | 162 | 0.753 | **0.938** | **0.035** |
| unanchored | 162 | 0.000 | **0.000** | 487.6 |

The overall figure is the less informative of the two. It averages a stratum the
system solves to four hundredths of a pixel with one that is not solvable at all.

Median end-to-end latency is **389 ms** (p95 422 ms), measured on
macOS-26.5.2-arm64 — see `results/full_324.meta.json`, which now travels with
every results CSV. Latency is a property of the machine: the same 324 pairs
report 356 ms on R4's Windows laptop and the identical accuracy, and repeat runs
on one machine vary by a few ms. Accuracy figures are deterministic and
reproduce to the digit.

---

## 1. The unanchored stratum is a ceiling, not a defect

162 of the 324 cases score zero, and no amount of tuning moves them.

An unanchored reference is a window onto a purely periodic region — an array
with no aperiodic feature in frame. Correlating a periodic template against a
periodic field produces a *lattice* of near-identical peaks rather than one
peak. The correlation evidence does not identify a position, because there is no
information in the images that distinguishes those placements.

This is confirmed by construction rather than inferred: R1's generator asserts
that an unanchored layout carries an empty anchor list, and raises if an
anchored case has no anchor in its reference. The stratum is built to contain no
distinguishing feature.

**The ceiling on this dataset is therefore 0.500.** Measured against that
ceiling, the system is at **0.938 of what is achievable**.

The correct response to an unsolvable case is to answer and flag, not to
succeed. Every unanchored case escalates to the ambiguous tier, returns the
centre-prior answer mandated by the problem statement, and carries a low
confidence. None returns a confident wrong answer. That is designed behaviour,
and the recall data in section 6 shows it is not a near miss: the true peak is
absent from the top 30 candidates in 160 of 162 unanchored cases.

---

## 2. Why FinFET is harder: two periodicities, not "fine features"

The intuitive explanation — FinFET fails because its features are fine — is
wrong, and a judge can falsify it. At `SEARCH_PX_NM = 10.0` the Nyquist period
is 20 nm, or 2 px. The finest generated feature, a 72 nm fin pitch, samples at
7.2 px. **Nothing aliases.** What happens is MTF attenuation from the PSF.

Recomputing `exp(-2 pi^2 sigma^2 / Lambda^2)` at `psf.sigma_nm = 12.0`:

| feature | period | px/period | PSF MTF | area-avg | total |
|---|---|---|---|---|---|
| FinFET fin | 72 nm | 7.2 | 0.578 | 0.969 | **0.560** |
| FinFET fin | 108 nm | 10.8 | 0.784 | 0.986 | **0.773** |
| DRAM pitch | 144 nm | 14.4 | 0.872 | 0.992 | 0.865 |
| DRAM pitch | 216 nm | 21.6 | 0.941 | 0.996 | 0.938 |
| FinFET gate | 336 nm | 33.6 | 0.975 | 0.999 | **0.974** |
| FinFET gate | 504 nm | 50.4 | 0.989 | 0.999 | **0.988** |

The area-average column is `sinc(10 / Lambda)` and contributes at most 3%
anywhere. **It is overwhelmingly the PSF.**

The real mechanism is that FinFET carries *two* periodicities at once: fins
retaining 56–77% contrast superimposed on gates retaining 97–99%. A strongly
attenuated fine lattice sitting on a nearly intact coarse one flattens the
correlation surface in a way that a single periodicity does not. DRAM has one
periodicity, at 87–94%.

Two caveats to state whenever this table is shown:

- It is **single-axis 1-D MTF**, and it describes contrast *amplitude*
  retention. It is not an accuracy prediction.
- **Pitch is a range, not a nominal.** `PITCH_TOLERANCE = 0.20` randomises it:
  FinFET 72–108 nm, DRAM 144–216 nm. No generated pair sits at 90 or 180.

**The measured split now runs the other way, and rotation estimation is why.**
DRAM 0.444 against FinFET 0.494 overall, or 0.889 against 0.988 on the anchored
stratum. Before pose estimation landed, FinFET was the harder family exactly as
this MTF argument predicts. It is now the easier one: FinFET's coarse gate
lattice, retaining 97–99% contrast, gives the spectral rotation estimator a
strong unambiguous peak, and that gain outweighs the fine-fin attenuation the
table describes. DRAM's single, finer periodicity gives the estimator less to
lock onto.

The median-error inversion survives and has the same cause — FinFET 56.1 px
against DRAM 159.3 px. FinFET fails *less badly* when it fails, because its
intact gate lattice still constrains the answer to a coarse grid.

The MTF table above therefore explains *correlation contrast*, which is real,
but it no longer predicts the accuracy ordering on its own. Say both.

---

## 3. What disambiguation actually buys

The plain-NCC baseline in `src/baseline_ncc.py` is the incumbent to beat. The
overall gap is +0.049. **That number overstates the contribution of
disambiguation and should not be quoted as it.**

| stratum | plain NCC | full pipeline | delta |
|---|---|---|---|
| ALL | 0.377 | 0.469 | +0.093 |
| **pose = none** | 0.463 | 0.472 | **+0.009** |
| pose = small | 0.398 | 0.472 | +0.074 |
| pose = large | 0.269 | 0.463 | +0.194 |

The baseline is handed nominal pose (θ = 0, s = 10). That is *correct* on the
`pose-none` stratum and wrong on the other two, so most of the aggregate gap is
the baseline's absence of pose estimation rather than anything disambiguation
does. On the stratum where the baseline's pose is right, disambiguation is worth
**+0.009**. That is the honest figure for disambiguation, and the per-stratum
breakdown is what separates the two claims.

The gap widens with pose severity — +0.009, +0.074, +0.194 — which is now a
direct readout of what rotation estimation contributes, since that is the only
capability the baseline lacks on those strata.

### Stage ablation

Cite this on the **anchored stratum** (n=162). Averaging in the 162 unanchored
cases, which are pinned at 0.000 by construction, halves every effect and makes
weighting look inert when it is not.

| stage | success@1px | median error (px) | ms |
|---|---|---|---|
| `ncc` | 0.753 | 0.454 | 32 |
| `+ weighting` | 0.833 | 0.456 | 159 |
| `+ selection` | 0.833 | 0.456 | 162 |
| `+ sub-pixel` | 0.852 | 0.042 | 207 |
| **`+ pose` (full system)** | **0.938** | **0.035** | 389 |

> **The first four rows hold pose at nominal by construction.** `ablate.py`
> resolves pose in `fast` mode (`ablate.py:70`) so that Stage 4 is isolated from
> Stage 1; re-running it after rotation estimation landed returned byte-identical
> stage figures, which is the expected result and a useful check that the
> ablation measures what it claims. The final row is the real pipeline from
> `results/full_324.csv`. Quote it as the system's accuracy; quote the others
> only as deltas between each other.

**Weighting is the entire disambiguation gain, +0.080.** Selection buys exactly
0.000: with `TIE_SIGMA = 0.0` the tolerance is always zero, `n_tied` is always
1, and `select_candidate` returns `peaks[0]` unchanged. The tie-break is
mandated and fires correctly when ties exist; on this dataset exact ties never
occur. Sub-pixel buys +0.019 and an 11x error reduction — 0.454 px is the
integer-peak quantisation floor.

> On the overall stratum the same ablation reads `ncc` 0.377 → `selected` 0.417
> → `weighted` 0.417, which invites the conclusion that weighting contributes
> nothing. That is an artefact of averaging over an unsolvable half, not a
> result.

---

## 4. PSR does not separate correct from incorrect

The peak-to-sidelobe ratio is the detection statistic gating escalation. It
carries little discriminative signal:

| architecture | mean PSR | median PSR | success@1px |
|---|---|---|---|
| DRAM | 2.628 | 2.665 | 0.444 |
| FinFET | 1.841 | 1.771 | 0.494 |

The cause is structural rather than a coding error. PSR compares the winning
peak against the surrounding surface, but on a periodic layout the sidelobe
region contains genuine lattice peaks rather than noise. The background the
statistic normalises against is itself signal, which inflates the sidelobe
standard deviation and compresses PSR toward the same value regardless of
correctness.

**Consequence:** with accept thresholds at 8.0 (fast) and 4.0 (robust),
**320 of 324 cases escalate** to the ambiguous tier — 3 accept at `robust`, 1 at
`fast`. Slow, but almost never falsely confident. The thresholds are
deliberately untuned: lowering them to buy speed would buy it by making wrong
answers confident.

> **Do not state that PSR is capped near 3.4.** That figure comes from the
> superseded 108 set. On 324 the observed maximum is **12.314**, with 2 cases
> at or above 8.0 and 4 at or above 4.0. The "everything escalates" conclusion
> survives at 98.8%, but the 3.4 ceiling is falsifiable by opening the CSV.

One measurement bug was found and fixed in an earlier pass: sidelobe statistics
excluded a **Euclidean disc** around the peak while peak extraction suppresses
over a **Chebyshev square**. The two regions disagree at the corners, so part of
the peak's own shoulder was counted as background, biasing PSR low. Now matched.

---

## 5. Confidence is uncalibrated, and the plot will be flat

R4's calibrator is at chance on this data. Mean confidence, split by whether the
answer was actually correct:

| architecture | correct | wrong | separation |
|---|---|---|---|
| DRAM | 0.0624 (n=72) | 0.0604 (n=90) | **+0.0021** |
| FinFET | 0.0632 (n=66) | 0.0548 (n=96) | **+0.0084** |

322 of 324 cases carry `low_confidence_flag = 1`. The pathology is visible in a
single row: case `dram_anchored_pose-none_0000` lands at `err_px` 0.0136 — a
correct sub-pixel answer — with confidence 0.0709.

**Caption the calibration plot honestly rather than hiding it.** A flat
reliability curve is the correct depiction of a calibrator that has no
discriminative feature to work with, and the reason it has none is section 4:
PSR is the dominant input and PSR does not separate.

Two features are dead outright and should be described as such:

- `n_tied` is **1** in all 324 cases.
- `tie_break_used` is **False** in all 324 cases.

Both follow from `TIE_SIGMA = 0.0`, which is the correct value — at the previous
1.0 a sigma width was being passed as a raw score tolerance, tying nearly all
candidates and handing the answer to the centre prior blind. The fix recovered
accuracy by close to an order of magnitude. The side effect is that neither
field carries information as a confidence feature.

---

## 6. Recall@K: the candidate list is not the bottleneck

Was truth in the candidate list at all, before disambiguation ran?
(weighted, NMS radius 8.)

> **Read this table at nominal pose.** `src/recall.py` resolves pose in `fast`
> mode (`recall.py:31`), so every figure below is measured with θ = 0 — the same
> convention as the stage ablation, and deliberately, so that candidate recall
> is isolated from Stage 1. It is why end-to-end anchored accuracy (0.938) now
> *exceeds* r@30 (0.852): the full pipeline estimates rotation and this harness
> does not. The two are not measured on the same pipeline and must not be
> subtracted from each other.

| stratum | n | r@1 | r@5 | r@10 | r@30 |
|---|---|---|---|---|---|
| anchored | 162 | 0.833 | 0.840 | 0.846 | **0.852** |
| unanchored | 162 | 0.000 | 0.000 | 0.000 | 0.012 |

> **Reporting hazard.** `rank = -1` means the true peak never entered the top
> 30. Those rows are misses. Filtering them out of the denominator — easy to do
> by accident when hand-rolling analysis from the CSV — yields a spurious
> `r@30 = 1.000`, a 2.3x overstatement. `src/recall.py` divides by the full case
> count and is correct; the hazard is in downstream re-analysis.

Two conclusions:

1. **Going from K=1 to K=30 buys +0.019 on anchored.** When the true peak is
   findable at all, it is essentially always rank 1. Deepening the candidate
   list is not where accuracy is hiding, and `--max-k 30` is generous already.
2. **At nominal pose, recall@30 on anchored (0.852) equalled end-to-end
   accuracy at nominal pose (0.852).** Disambiguation was losing nothing that
   peak extraction found, and the 24 remaining anchored failures were cases
   where the true peak was never a candidate — a Stage 1/3 problem, not a
   Stage 4 problem. Rotation estimation has since addressed most of exactly
   that: end-to-end anchored accuracy is now 0.938. Re-running this harness
   with pose live would be the clean way to confirm the same identity holds at
   the new operating point; it has not been done.

### NMS radius does not bite

`DEFAULT_NMS_RADIUS_PX = 8` sits inside the FinFET fin pitch range of 7.2–10.8
search px, which predicts that NMS could suppress genuine adjacent lattice
peaks on roughly 22% of FinFET pairs. **Measured, it does not:**

| comparison | cases differing in rank |
|---|---|
| radius 8 vs 2 | 0 / 324 |
| radius 8 vs 4 | 0 / 324 |
| radius 8 vs 16 | 1 / 324 |

Radii 2 through 8 are bit-identical. The parameter is plumbed through — 16 moves
one case — it simply has almost no effect, consistent with
`DEFAULT_NMS_RADIUS_PX` being a documented fallback superseded at runtime by a
radius derived from Stage 1's spectral pitch estimate. The predicted PSR
inflation is also absent: FinFET PSR is *lower* than DRAM, the opposite of what
suppression would produce.

This tests that the constant does not bite. It does **not** test that the
runtime override is correct; that is a separate, unmeasured claim.

`PSR_EXCLUSION_RADIUS_PX` and `DEFAULT_NMS_RADIUS_PX` are nonetheless both 8 by
coincidence and are set by different physics — correlation main-lobe width
versus lattice pitch. They must not be recoupled now that R1 randomises pitch.

---

## 7. Secondary observations

**Pose is the live failure axis.** none 0.481 → small 0.463 → large 0.333. That
is the degradation curve, and it is monotone.

**Noise does not matter, and this is now measured rather than inferred.** The
strata comparison is weak evidence on its own — high 0.444, low 0.417, medium
0.417, with "high" nominally *best*, a spread of about three cases. The
noise-free control settles it properly.

`dataset_control` is generated with `--noise-levels none --seeds-per-cell 27`,
which yields 324 pairs sharing **every** geometry field with the shipped set
pair-for-pair: `ground_truth`, `layout_params`, `crop_centre_nm`,
`search_centre_nm`, `anchors_gt`, `anchors_in_reference` and `seed` are
identical across all 324. Only `strata.noise_level` and
`physics_params.noise_level` differ. That permits a per-pair difference — *this
exact scene, clean versus noisy* — rather than an aggregate over different
scenes.

| | count |
|---|---|
| correct in both | 143 |
| correct only without noise | 5 |
| correct only with noise | 9 |
| wrong in both | 167 |

**310 of 324 pairs return the identical verdict.** Fourteen are discordant, and
they lean the wrong way — removing noise costs 4 net cases (anchored 0.914 clean
against 0.938 noisy), which is not a causal effect but coin-flip variation among
cases sitting on the 1 px threshold. The discordant count roughly doubled when
rotation estimation landed, which is what you would expect: a more accurate
system puts more pairs close enough to the threshold for a coin flip to matter.

The correct claim is therefore: **at these strata, noise is not a driver of
failure.** The unstated part, which must accompany it, is that this bounds the
strata R1 generated, not SEM noise in general — if the strata are mild, this
result says the pipeline is insensitive to mild noise and nothing more.

**Sub-pixel refinement** resolves 290 of 324 cases by phase cross-correlation
and falls back to surface upsampling on 34.

**No case reports a failure mode.** `failure_mode` is `none` in all 324 rows;
every failure here is a wrong answer delivered normally, not a crash or a
detected breakdown.

---

## 8. How to reproduce

```bash
uv run python -m src.evaluate     --data dataset --gt dataset/ground_truth.jsonl --out results/full_324.csv
uv run python -m src.baseline_ncc --data dataset --gt dataset/ground_truth.jsonl --out results/baseline_324.csv
uv run python -m src.ablate       --data dataset --gt dataset/ground_truth.jsonl --out results/ablation_324.csv
uv run python -m src.recall       --data dataset --gt dataset/ground_truth.jsonl --out results/recall_324.csv --max-k 30
```

Pass `--gt` explicitly. A silently wrong ground-truth path poisons every number
in this document.

Per-case rows — including `psr`, `n_tied`, `tie_break_used`, `uniqueness_score`,
`confidence` and `failure_mode` — are in the CSVs, which are tracked as evidence.
Dataset integrity is verifiable against `dataset/dataset_manifest.json`.

Generation is byte-reproducible across platforms, verified 2026-08-12 against
pre-registered hashes on Windows and Mac, so the dataset can be regenerated from
seed rather than transferred:

```bash
uv run python -m src.generate_dataset --output-dir dataset --seed 20260807
```
