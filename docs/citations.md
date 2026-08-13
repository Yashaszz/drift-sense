# Citations and sources

Every parameter in the dataset generator and every algorithm in the pipeline,
with the source it comes from — and, where there is no source, an explicit
statement that the number is an engineering assumption.

The distinction matters more than the bibliography does. Drift-Sense trains and
evaluates on synthetic data, so the honest question is not "is this realistic?"
but "which parts of it are grounded, and what does the ungrounded remainder
buy or cost?" A citation that dresses up a guessed constant is worse than no
citation, because it moves an assumption out of sight.

## How to read this document

Every row carries a tier:

| tier | meaning |
|---|---|
| **[C]** | Mechanism *and* magnitude come from the cited source. |
| **[M]** | The **mechanism** is cited; the **number** is ours, chosen for the simulation. |
| **[A]** | **Assumption.** No source. Stated here so it can be argued with. |

There are more `[M]` and `[A]` rows than `[C]` rows, and that is the accurate
picture: `src/sem_physics.py` is a simulation baseline whose form is taken from
the SEM literature and whose constants were chosen to produce two visibly
different captures, not measured on a tool.

Numbered sources are listed in §7.

---

## 1. Layout geometry

`src/generate_dataset.py:142-178`, rendered by `src/layouts.py` and
`src/render.py`.

| parameter | value | tier | source / note |
|---|---|---|---|
| DRAM line pitch | 180 nm nominal | **[M]** | Periodic word-line/bit-line arrays on a fixed pitch are the defining feature of DRAM array regions [8, 9]. The magnitude is ours — see the deviation note below. |
| DRAM line width | 40 nm nominal | **[A]** | Chosen for a line:space ratio that survives the 10 nm/px search sampling. |
| DRAM via diameter | 60 nm nominal | **[A]** | As above. |
| FinFET fin pitch | 90 nm nominal | **[M]** | Fins on a uniform sub-100 nm pitch, crossed by gates on a coarser pitch, is the FinFET standard-cell topology [10]. Intel's 22 nm tri-gate is on a 60 nm fin pitch and a 90 nm contacted gate pitch [10, 11]. |
| FinFET fin width | 24 nm nominal | **[A]** | Real fins are far narrower — Auth et al. report an 8 nm fin width at 22 nm [10]. Widened here so a fin is more than a single pixel after decimation. |
| FinFET gate width | 13 nm nominal | **[A]** | |
| FinFET gate pitch | 420 nm nominal | **[A]** | Set so that a 1000 nm reference crop contains more than one gate; an earlier revision put two gates across the whole 12 µm layout and the crop saw at most one (`src/layouts.py:221`). |
| Pitch randomisation | ±20% | **[A]** | From the work-split document, pinned by `tests/test_geometry.py:339`. |
| Linewidth randomisation | ±15% | **[A]** | As above. |

**The deviation, stated plainly.** These dimensions are relaxed from real
silicon by **1.5× to 5×, depending on which dimension you compare**:

| ours | nearest real reference | ratio |
|---|---|---|
| fin pitch 90 nm | 60 nm, Intel 22 nm tri-gate [10, 11] | 1.5× |
| fin pitch 90 nm | 42 nm, Intel 14 nm [11] | 2.1× |
| gate pitch 420 nm | 90 nm contacted gate pitch, Intel 22 nm [10] | 4.7× |
| fin width 24 nm | 8 nm, Intel 22 nm [10] | 3.0× |
| DRAM pitch 180 nm | sub-20 nm-class M1 half-pitch, i.e. sub-40 nm pitch [8, 9] | ~4–5× |

This is deliberate, and the reason is **contrast, not sampling.** An earlier
revision of this document argued that real pitches would alias at 10 nm/px.
That was wrong, and `docs/assumptions.md` §4 is the correction: Nyquist needs
≥2 px/period, and even a 42 nm pitch clears it at 4.2 px/period, so nothing in
the dataset aliases.

What actually binds is the modulation surviving R2's search PSF (σ = 12 nm).
Through a Gaussian MTF, a 60 nm real pitch reaches the search image at **45%
contrast before any noise is added**, and Intel's 42 nm fin pitch at **20%**,
which under the `high` noise stratum is at or below the shot-noise floor. At
those dimensions the task is not hard, it is **ill-posed** — every unanchored
pair would fail for reasons that say nothing about the matcher. Our geometry
lands at 58–94% contrast against 45–74% for every confirmed real dimension.

The full derivation and the per-feature contrast table are in
`docs/assumptions.md` §4, which is the authority for this argument; this
section defers to it rather than restating it.

The consequence for the results: **the periodic-ambiguity problem is modelled
faithfully; the specific node is not.** Nothing in the pipeline is tuned to a
pitch value (`PSR_EXCLUSION_RADIUS_PX` and `DEFAULT_NMS_RADIUS_PX` are
deliberately decoupled from it, see `src/config.py`), so the scaling does not
leak into the algorithm.

**The ±20% / ±15% randomisation is not a process-tolerance figure.** Real CD
uniformity is a small-single-digit-percent 3σ. Ours is far wider on purpose, in
the spirit of domain randomisation [12]: the point is that no matcher parameter
may depend on a pitch the grader could change, not that fabs vary this much.

### 1.1 Anchors are defect classes

The aperiodic features that make a reference localisable are
`missing_via`, `oversized_via`, `shifted_via` (`src/layouts.py:414`) and
`missing_fin`, `merged_fin`, `gate_break` (`src/layouts.py:528`).

**[M]** — missing pattern, bridged/merged pattern, and open lines are the
canonical defect categories in the yield and inspection literature [13, 14].
Their *sizes and placement* are ours **[A]**, chosen so an anchor falls inside
the reference crop (`DEFAULT_ANCHOR_HALF_SPAN_NM = 340.0`).

The unanchored stratum carries an empty anchor list by construction. That is
what makes its 0.000 accuracy a ceiling rather than a failure — see
`docs/failure_analysis.md` §1.

---

## 2. SEM image formation

`src/sem_physics.py`. Applied in a fixed order: edge brightening → PSF blur →
Poisson shot noise → read noise → scan artifacts, independently per capture.

The module's own docstring says it: *"a simulation baseline, not a calibrated
SEM instrument model."* This section does not upgrade that claim. It records
which mechanism each stage is imitating.

### 2.1 Edge brightening — `edge_brightening()`

**[M]** Secondary-electron yield rises as the surface tilts away from the
beam, approximately as 1/cos θ, so edges and sidewalls emit more than flat
regions and appear bright. Seiler's review is the standard source for SE yield
versus angle and for the escape-depth argument behind it [1]; Goldstein et al.
and Reimer treat the resulting edge contrast as the dominant topographic
contrast mechanism in SE imaging [2, 3].

**[A]** The implementation is a Sobel-gradient proxy, normalised by the 99th
percentile and Gaussian-spread — not a yield model. `strength` 0.055
(reference) / 0.070 (search) and `sigma_nm` 1.5 / 15.0 are chosen values. A
real edge signal is asymmetric with respect to detector position; ours is
isotropic.

### 2.2 Probe blur — `psf_blur()`

**[M]** Finite probe size plus the lateral spread of the interaction volume
band-limits the image; modelling it as a convolution is standard [2, 3].

**[A]** An isotropic Gaussian is an approximation and a known one: the real SE
point-spread has a narrow SE1 core and a broad SE2 shoulder from backscattered
electrons, so it is heavy-tailed rather than Gaussian, as Monte Carlo
modelling shows [4]. NIST's synthetic-SEM work makes the same simplification
for the same reason — a tractable generator [5]. `sigma_nm` 1.2 (reference) /
12.0 (search) are chosen; the 10× ratio mirrors the 10× sampling ratio.

### 2.3 Poisson shot noise — `poisson_shot_noise()`

**[M]** Electron emission and detection are counting processes, so image
variance is proportional to signal — the dominant noise term in SEM at
practical dwell times [2, 3]. The Rose criterion, that a feature needs SNR of
about 5 to be reliably seen, is the classical statement of what that costs
[6].

**[A]** `white_counts` = 12 000 (reference) / 3 000 (search) are effective
counts at white, used *only* to parameterise the Poisson variance. They are not
measured detector counts and no dwell time or beam current is claimed. The
stratum multipliers — low ×2.0, medium ×1.0, high ×0.35 — are ordinal, not
calibrated: they order the strata, they do not locate them.

### 2.4 Read noise — `read_noise()`

**[M]** Signal-independent, approximately Gaussian amplifier/detector noise
added after the counting stage is the standard sensor noise model [7].

**[A]** σ = 0.004 (reference) / 0.010 (search) in normalised intensity units,
scaled ×0.5 / ×1.0 / ×2.0 across the strata.

### 2.5 Scan artifacts — `scan_artifacts()`

**[M]** Raster acquisition produces line-to-line intensity variation and
low-frequency banding, and SEM images additionally carry real geometric drift
and scan distortion. Sutton et al. measured both on a real instrument across
200×–10 000× and removed them to about ±0.02 px [15].

**[A] and an important divergence.** Our model applies **intensity-only**
artifacts: correlated row gain (σ 0.006 / 0.018), row offset (σ 0.003 / 0.009),
a 3–4 row correlation length, and a weak sinusoidal band (amplitude 0.002 /
0.006). It deliberately applies **no row displacement**, because the module's
coordinate contract forbids changing geometry — R1's pixel-centre ground truth
must stay exact.

So the dataset is *easier than a real tool in one specific respect*: real scan
distortion and drift would add a spatially varying geometric error on top of
the rigid pose error we do model. That is a known, bounded gap, and it is the
first thing to add if this ever runs against instrument data.

### 2.6 Two independent captures

**[M]** The reference and search images are separate physical acquisitions, so
they get independent noise draws and different presets — different PSF,
different noise level, different gain (`src/generate_dataset.py:183-192`). This is
why the matcher uses ZNCC and not SSD (§3.1).

---

## 3. Algorithms

| stage | choice | tier | source |
|---|---|---|---|
| 3 — correlation | Zero-mean normalised cross-correlation, FFT-accelerated | **[C]** | Lewis [16] is the standard reference for fast normalised cross-correlation; ZNCC is invariant to affine intensity change, which independent captures with different detector gain and offset require. `src/matcher.py:714`. |
| 3 — framing | Correlation as matched filtering | **[C]** | Turin [17]. The matched filter is the optimal linear detector for a known signal in additive noise, which is exactly the template-in-search problem. `src/matcher.py:8`. |
| 1 — pose | Fourier / log-polar (Fourier–Mellin) rotation and scale estimation | **[C]** | De Castro & Morandi [18]; Reddy & Chatterji [19]. `src/pose.py:11-20`. |
| 5 — sub-pixel | Upsampled-DFT phase correlation, `DEFAULT_UPSAMPLE = 100` | **[C]** | Guizar-Sicairos, Thurman & Fienup [20] — evaluate the inverse DFT directly on a fine grid near the peak by matrix multiplication instead of zero-padding the whole transform (`_upsampled_patch`, `src/matcher.py:1298`). Foroosh et al. [21] for sub-pixel phase correlation, which is the primary routine (`src/matcher.py:1221`); surface interpolation is the documented fallback. |
| 4b — detection statistic | Peak-to-sidelobe ratio | **[C]** | Kumar & Hassebrook [22] define PSR as a correlation-filter performance measure; MACE-filter work uses it as the accept/reject statistic [23]; MOSSE uses it directly as a per-frame failure detector [24]. `src/disambiguate.py:107`. |
| 4b — thresholding | Accept / escalate on a detection statistic | **[M]** | Structurally CFAR: set the threshold from the local background rather than absolutely [25, 26]. Our thresholds (8.0 / 4.0) are **[A]** and deliberately untuned — lowering them buys speed by making wrong answers confident. `src/config.py:112-115`. |
| 4a — uniqueness | Tile autocorrelation sidelobe as a positional-ambiguity score | **[M]** | A signal's autocorrelation *is* its ambiguity function under matched filtering, so sidelobe height measures positional ambiguity directly — Woodward's formulation [27]. The tiling, the σ = 4 px prefilter, the Hann taper and the absolute (never per-image) scale are ours **[A]**; the reasoning is in `src/uniqueness.py`. |
| 3b — candidates | Non-maximum suppression at a fixed radius | **[A]** | Standard practice; radius 8 px is set by lattice pitch, not by literature. Recall is flat across radii 2–16 (`results/recall_324.csv`), so the choice is not load-bearing. |
| 6 — confidence | Logistic calibration of a diagnostic vector | **[C]** | Platt scaling [28]; Niculescu-Mizil & Caruana [29] on why raw scores are not probabilities. `src/confidence.py:218`. |
| overall | Two-stage coarse-to-fine registration | **[C]** | Brown [30] and Zitová & Flusser [31], the standard registration surveys. |

---

## 4. What is not cited, in one place

If you read nothing else in this document:

1. **No number in `src/sem_physics.py` is calibrated against a real
   instrument.** Every preset constant is `[A]`. Absolute noise levels are not
   a claim; only the *ordering* of the strata is.
2. **Layout dimensions are 2–5× relaxed from production**, for the sampling
   reason in §1. The ambiguity structure is faithful; the node is not, and no
   figure here should be presented as matching one.
3. **The randomisation tolerances are not process tolerances.**
4. **No geometric scan distortion or drift is modelled** (§2.5), by design, to
   keep ground truth exact. Real SEM data would be harder here.
5. **PSR thresholds are untuned**, and on this dataset 320 of 324 cases
   escalate. That is reported as a finding, not hidden as a default.

Two of these are directly falsifiable from tracked evidence rather than taken
on trust: `results/control_324.csv` is a paired noise-free control — the same
324 scenes, same seeds, `noise_level` alone changed — which is what lets §2's
noise model be argued about per-pair instead of in aggregate
(`docs/failure_analysis.md` §7).

---

## 5. Sources

**SEM physics**

1. H. Seiler, "Secondary electron emission in the scanning electron
   microscope," *Journal of Applied Physics* **54**(11), R1–R18, 1983.
   doi:10.1063/1.332840
2. J. Goldstein et al., *Scanning Electron Microscopy and X-Ray
   Microanalysis*, 4th ed., Springer, 2018.
3. L. Reimer, *Scanning Electron Microscopy: Physics of Image Formation and
   Microanalysis*, 2nd ed., Springer, 1998.
4. D. C. Joy, *Monte Carlo Modeling for Electron Microscopy and
   Microanalysis*, Oxford University Press, 1995.
5. P. Cizmar, A. E. Vladár, B. Ming, M. T. Postek, "Simulated SEM images for
   resolution measurement," *Scanning* **30**(5), 381–391, 2008.
   doi:10.1002/sca.20120
6. A. Rose, "The sensitivity performance of the human eye on an absolute
   scale," *Journal of the Optical Society of America* **38**(2), 196–208,
   1948.
7. J. R. Janesick, *Photon Transfer: DN → λ*, SPIE Press, 2007.
15. M. A. Sutton et al., "Scanning electron microscopy for quantitative small
    and large deformation measurements part I: SEM imaging at magnifications
    from 200 to 10,000," *Experimental Mechanics* **47**(6), 775–787, 2007.
    doi:10.1007/s11340-007-9042-z

**Devices, layout and yield**

8. IEEE, *International Roadmap for Devices and Systems (IRDS) 2022 — More
   Moore*. https://irds.ieee.org/images/files/pdf/2022/2022IRDS_MM.pdf
9. IEEE, *IRDS 2022 — Lithography*.
   https://irds.ieee.org/images/files/pdf/2022/2022IRDS_Litho.pdf
10. C. Auth et al., "A 22nm high performance and low-power CMOS technology
    featuring fully-depleted tri-gate transistors, self-aligned contacts and
    high density MIM capacitors," *Symposium on VLSI Technology*, 131–132,
    2012.
11. WikiChip, "22 nm lithography process."
    https://en.wikichip.org/wiki/22_nm_lithography_process
12. J. Tobin et al., "Domain randomization for transferring deep neural
    networks from simulation to the real world," *IROS*, 2017.
13. I. Koren, Z. Koren, "Defect tolerance in VLSI circuits: techniques and
    yield analysis," *Proceedings of the IEEE* **86**(9), 1819–1838, 1998.
14. IEEE, *IRDS — Yield Enhancement* chapter. https://irds.ieee.org/editions

**Registration, detection and calibration**

16. J. P. Lewis, "Fast normalized cross-correlation," *Vision Interface*,
    120–123, 1995.
17. G. L. Turin, "An introduction to matched filters," *IRE Transactions on
    Information Theory* **6**(3), 311–329, 1960.
18. E. De Castro, C. Morandi, "Registration of translated and rotated images
    using finite Fourier transforms," *IEEE TPAMI* **9**(5), 700–703, 1987.
19. B. S. Reddy, B. N. Chatterji, "An FFT-based technique for translation,
    rotation and scale-invariant image registration," *IEEE Transactions on
    Image Processing* **5**(8), 1266–1271, 1996.
20. M. Guizar-Sicairos, S. T. Thurman, J. R. Fienup, "Efficient subpixel image
    registration algorithms," *Optics Letters* **33**(2), 156–158, 2008.
21. H. Foroosh, J. B. Zerubia, M. Berthod, "Extension of phase correlation to
    subpixel registration," *IEEE Transactions on Image Processing* **11**(3),
    188–200, 2002.
22. B. V. K. Vijaya Kumar, L. Hassebrook, "Performance measures for
    correlation filters," *Applied Optics* **29**(20), 2997–3006, 1990.
23. A. Mahalanobis, B. V. K. Vijaya Kumar, D. Casasent, "Minimum average
    correlation energy filters," *Applied Optics* **26**(17), 3633–3640, 1987.
24. D. S. Bolme, J. R. Beveridge, B. A. Draper, Y. M. Lui, "Visual object
    tracking using adaptive correlation filters," *CVPR*, 2010.
25. H. Rohling, "Radar CFAR thresholding in clutter and multiple target
    situations," *IEEE Transactions on Aerospace and Electronic Systems*
    **19**(4), 608–621, 1983.
26. M. A. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed.,
    McGraw-Hill, 2014 — CFAR detection.
27. P. M. Woodward, *Probability and Information Theory, with Applications to
    Radar*, Pergamon Press, 1953 — the ambiguity function.
28. J. Platt, "Probabilistic outputs for support vector machines and
    comparisons to regularized likelihood methods," *Advances in Large Margin
    Classifiers*, 1999.
29. A. Niculescu-Mizil, R. Caruana, "Predicting good probabilities with
    supervised learning," *ICML*, 2005.
30. L. G. Brown, "A survey of image registration techniques," *ACM Computing
    Surveys* **24**(4), 325–376, 1992.
31. B. Zitová, J. Flusser, "Image registration methods: a survey," *Image and
    Vision Computing* **21**(11), 977–1000, 2003.

---

## 6. Verification status

Checked against the publisher record on **2026-08-13**: [1], [5], [8], [9],
[10], [11], [15].

The remainder are standard works in their fields, written from the literature
rather than re-checked against a publisher record on that date. Volume, issue
and page numbers on those entries should be confirmed before the submission
zip if the graders weight bibliographic precision. Nothing in §1–§4 depends on
a page number.
