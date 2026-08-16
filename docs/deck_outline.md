# Deck — Drift-Sense, PS-02

**Seven slides, in the official i4C template's shape.** Rebuilt 14 August after
R1's submission brief: the template caps the deck at 6–7 slides including the
title, mandates its own file, and is submitted as **PDF** named
`TeamName_PS02`. An earlier 12-slide outline is superseded; its content is
folded into the slots below.

The template lists nine content headings but allows only six or seven slides,
so four are merged. The merges are chosen so that every heading carrying marks
still appears: the scoring split is 50% accuracy and runtime, 30% synthetic-data
realism and citations, 10% failure analysis, plus the RGB bonus.

Every number is traceable to a tracked file, named on the slide. The provenance
table at the end maps slide to source.

---

## Rules for anyone editing this deck

Six ways to be wrong in front of a judge. Five have already happened once in
this repo, and the sixth nearly reached this deck.

1. **Never quote a 108-set number.** That set was physics-free and is
   superseded. Anchored 77.8%, 105 ms and 0.01 px sub-pixel are all from it.
2. **Never quote all-pairs accuracy without the anchored figure.** 0.469
   averages a stratum we solve to 0.035 px with one that is unsolvable by
   construction. Lead with anchored 0.938 and explain the ceiling.
3. **Never quote a latency without its machine.** 389 ms is the Mac in
   `results/full_324.meta.json`. A Windows machine read ~1.7x slower, but that
   comparison predates pose estimation and has not been re-measured.
4. **Never say PSR is capped near 3.4.** 324-set maximum is 12.314.
5. **Never claim our dimensions match a production node.** They are relaxed
   1.5–5x, for the contrast reason on slide 4. Say it before you are asked.
6. **Never claim we have zero confident wrong answers.** True on the set we
   developed against, false on the holdout, where both accepted answers were
   wrong. Slide 6 states it correctly; do not "simplify" it back.

---

## Slide 1 — Title and team

Template slot 1. Team name, members, college, leader contact and email.

> **Blocked:** nobody has supplied names, academic years, college, or the
> leader's phone and email. This is the only slide with no content at all.

## Slide 2 — Problem statement, and why it is hard

Template slots 2 and 3 merged (Problem Statement + Idea Description).

A wafer-inspection tool revisits the same microscopic site. The stage lands
roughly a micrometre off, and **because every die is a printed copy of the same
design, the wrong location looks identical to the right one.**

Given a zoomed-in `reference` (1 nm/px) and a zoomed-out `search` (10 nm/px),
return the reference's centre in search pixels.

The hard part is not finding a match, it is choosing between identical ones:
correlating a periodic template against a periodic field produces a **lattice of
near-identical peaks**. Recall is easy; picking the right peak is the problem.

> Visual: one correlation surface showing the lattice of maxima. This single
> image earns the rest of the deck.

## Slide 3 — Proposed solution

Template slot 4 (Methodology, technologies, implementation).

Six stages, escalating rather than spending compute up front:

```
0 normalise  1 pose  2 template  3 ZNCC surface (FFT matched filter)
3b peaks + NMS  4 disambiguate  5 sub-pixel  6 confidence + flag
```

**The idea that does the work is Stage 4a, uniqueness weighting.** Not all of
the reference is equally informative — periodic regions match everywhere,
aperiodic ones match in one place — so the matched filter is weighted toward
the parts that can localise. The score is a tile's autocorrelation sidelobe: a
signal's autocorrelation *is* its ambiguity function under matched filtering
(Woodward, 1953), so this measures positional ambiguity directly.

Ablation on the anchored stratum, n=162:

| stage | accuracy | median error |
|---|---|---|
| ZNCC only | 0.753 | 0.454 px |
| + uniqueness weighting | 0.833 | 0.456 px |
| + candidate selection | 0.833 | 0.456 px |
| + sub-pixel | 0.852 | 0.042 px |
| **+ pose estimation** *(full system)* | **0.938** | **0.035 px** |

Two stages carry the system. **Uniqueness weighting is the entire
disambiguation gain, +0.080**, and **pose estimation adds +0.086** — the largest
single contribution. Sub-pixel buys an 11x error reduction; candidate selection
buys exactly 0.000, because `TIE_SIGMA = 0.0` means exact ties never occur on
this dataset, so the mandated tie-break is correct and never fires.

> The first four rows are measured with pose held at nominal, by construction
> (`ablate.py` resolves pose in `fast` mode), so that Stage 4 is isolated from
> Stage 1. The final row is the real pipeline from `results/full_324.csv`.
> Quote it as the system's accuracy; quote the others only as stage deltas.

## Slide 4 — Synthetic data: realism, diversity, reproducibility

Template slot 4 continued, and the **30% block**. If a slide gets cut, not this
one.

- **324 stratified pairs**: 2 architectures x 2 anchor states x 3 pose
  conditions x 3 noise strata x 9 seeds.
- **Real SEM physics**: edge brightening, PSF blur, Poisson shot noise, read
  noise, scan artifacts, applied independently per capture.
- **Every parameter is tagged** in `docs/citations.md` — cited mechanism *and*
  magnitude, cited mechanism with our number, or declared assumption. Most of
  the SEM chain is the latter two, and the document says so.
- **Byte-reproducible across platforms.** The holdout's tree hashes were
  pre-registered before generation and a Mac run reproduced them exactly
  against the Windows values.

**Say this before a judge asks it:** our dimensions are relaxed 1.5–5x from
production silicon. Not because tighter is hard, but because at real ground
rules the lattice reaches the search image at 45% contrast, and at leading-edge
fin pitch at 20% — at or below the shot-noise floor. The task would stop being
hard and start being ill-posed. Nyquist is *not* the binding constraint;
contrast is.

## Slide 5 — Results, and the honest reading

Template slots 5 and 6 merged (Innovation & Uniqueness + Impact & Benefits),
and the **50% block**.

| tolerance | plain NCC (all / anchored) | full pipeline (all / anchored) |
|---|---|---|
| ≤ 5 px | 0.423 / 0.846 | 0.472 / **0.944** |
| ≤ 4 px | 0.417 / 0.833 | 0.472 / **0.944** |
| ≤ 2 px | 0.398 / 0.796 | 0.472 / **0.944** |
| ≤ 1 px | 0.377 / 0.753 | 0.469 / **0.938** |

Median error 0.035 px anchored. Latency 389 ms median, 422 ms p95, on the Mac
named in the metadata sidecar.

**The unanchored zero is a ceiling, not a failure.** The generator asserts that
an unanchored layout carries an empty anchor list, so the stratum is built to
contain no distinguishing feature. The ceiling is 0.500 and we are at 0.938 of
what is achievable. The true peak is absent from the top 30 candidates in 160
of 162 unanchored cases — not a near miss, the information is not there. All
162 escalate to the ambiguous tier, answer with the highest-scoring lattice
cell, and carry a low-confidence flag.

> Speaker note: if a judge challenges the 0.469, this is the answer. Do not
> apologise for the zero; explain why a correct system must produce it.

> Speaker note: do **not** say these come back as the centre prior — an earlier
> draft did. The centre tie-break decides between candidates that tie, and with
> `TIE_SIGMA = 0.0` nothing ties: `n_tied` is 1 and `tie_break_used` is `False`
> in all 324 rows. The centre-of-image answer is the failure degradation, and
> `failure_mode` is `none` everywhere. If asked where the unanchored answers
> land: a median 386 px from the centre, i.e. scattered across the lattice —
> which is exactly what "the evidence does not identify a position" looks like.

## Slide 6 — Failure analysis and explainability

Template slot 6 continued, the **10% block**, plus the RGB bonus.

### We held out 324 pairs and tested ourselves

A second dataset, seed 389722107, disjoint from the one we developed against,
with its tree hashes **registered before it was generated**. Scored **once**,
after development stopped.

| anchored, n=162 | developed on | **never seen** |
|---|---|---|
| success @ 1 px | 0.938 | **0.951** |
| success @ 4 px | 0.944 | **0.975** |
| median error | 0.035 px | **0.031 px** |

**Accuracy generalised. Our confidence measure did not** — and that is the
result worth your attention.

On the development set the system accepted one answer without a flag and it was
correct. On the holdout it accepted two, and **both were wrong**:

| case | error | PSR | accept threshold |
|---|---|---|---|
| `finfet_anchored_pose-small_0201` | 1.49 px | 11.83 | 8.0 |
| `finfet_anchored_pose-large_0236` | 1.66 px | 14.97 | 8.0 |

Both are FinFET, whose coarse gate lattice produces genuine strong sidelobes
that PSR reads as a clean isolated peak. **PSR is a detection statistic, and on
a periodic layout it cannot tell a correct peak from a confident one.** That is
why our thresholds stay untuned, and why every answer ships with a flag rather
than a bare number.

Two honest footnotes, both of which we volunteer:

- Both errors are under 1.7 px, so at the **2 px** tolerance they score as
  passes and this count reads zero. The failure is real at 1 px and vanishes at
  2 px. We report it at both rather than pick the flattering one.
- An earlier revision of this deck claimed *"zero confident wrong answers."* It
  was true on the data we tuned against. The holdout is what stopped us saying
  it on stage.

Where the remaining error lives, anchored: **not pose, and not noise.** Pose is
now nearly flat (0.944 / 0.944 / 0.926 across none / small / large) and the
*high*-noise stratum is the most accurate at 0.963. The largest split left is
architecture — **DRAM 0.889 against FinFET 0.988** — an inversion of the
pre-pose ordering, because FinFET's regular gate lattice gives the spectral
estimator a strong rotation peak and DRAM's finer pitch does not.

**RGB bonus — a measured limit.** Visible light cannot resolve this layout:
incoherent imaging passes nothing above `2·NA/λ`, so at NA 0.95 the finest
surviving period is 237 nm (blue) to 337 nm (red), against a 72–108 nm fin
pitch. Measured, the best channel retains ~6% of the geometry's contrast. This
is *why* the problem is posed with an SEM.

> Speaker note: lead with the holdout, not the accuracy. Most teams will show a
> number from the data they built against; we can show what happened when the
> system met data it had never seen, including the part that failed. If asked
> why we volunteered a failure, the answer is that the graders have their own
> test data and we would rather find this ourselves.
>
> Have ready, in case it is asked: the previous confident error
> (`finfet_anchored_pose-large_0226`, 2.28 px at PSR 8.13) was fixed by pose
> estimation — the same pair now lands at 0.098 px with PSR 1.81. So the
> mechanism is understood; it is the *statistic* that does not generalise, not
> the pipeline.

## Slide 7 — Technology, repository, video and references

Template slots 7, 8 and 9 merged (Technology & Feasibility + GitHub & Video +
Research & References).

- **Stack:** Python 3.12, NumPy, SciPy, OpenCV, scikit-image, Pillow. No deep
  learning, no pretrained weights, no network access at runtime.
- **Reproducibility:** `uv.lock` and `.python-version` committed; 597 tests;
  ruff, ruff-format and mypy in CI; a clean-room check that unpacks
  `git archive` and runs the suite, the linters and the CLI against exactly
  what a grader unzips.
- **GitHub:** repository link.
- **Video:** prototype/simulation link — **required by the template and nobody
  has started it.**
- **References:** at least three. Lewis 1995 (fast normalised
  cross-correlation), Guizar-Sicairos et al. 2008 (sub-pixel registration),
  Reddy & Chatterji 1996 (Fourier–Mellin pose), Woodward 1953 (ambiguity
  function), Seiler 1983 (SE yield), Jan et al. IEDM 2012 (22 nm ground rules).
  Full list in `docs/citations.md` and `docs/citations_layout.md`.

---

## Open items this deck cannot close

| item | owner | status |
|---|---|---|
| Team details for slide 1 | anyone | **missing** |
| Prototype/simulation video | unowned | **not started** |
| Deadline confirmation on the i4C portal | anyone | **unconfirmed** |
| Physics citations for the 30% block | R2 | not written |
| `generate_dataset.py` / `localize.py` at top level | R4 | not done |
| Scale range 9:1–11:1 (spec) vs 9.5:1–10.5:1 (ours) | R1 | optional |

## Number provenance

| slide | claim | source |
|---|---|---|
| 3 | stage ablation | `results/ablation_324.csv` |
| 4 | strata, physics, reproducibility | `dataset/dataset_manifest.json`, `docs/citations.md` |
| 4 | contrast and relaxation | `docs/assumptions.md` §4 |
| 5 | pass rates, latency | `results/full_324.csv`, `results/baseline_324.csv`, `results/full_324.meta.json` |
| 5 | 160 of 162 top-30 misses | `results/recall_324.csv` |
| 6 | the confident error | `results/full_324.csv`, row `finfet_anchored_pose-large_0226` |
| 6 | RGB limit | `src/optical.py`, `docs/rgb_optical_bonus.md` |
| 7 | test count, tooling | CI, `scripts/clean_room_check.sh` |
