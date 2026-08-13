# Citations — layout and geometry (R1)

**Scope: layout and sampling geometry only.** Written by R1 under work-split
rule 5 (whoever picks a number writes its citation). **Not yet audited** — R3
compiles and audits the project-wide `citations.md`, and this file is intended
to be folded in as its layout section once both have landed. It is deliberately
named `citations_layout.md` to avoid a whole-file collision with that add.

Physics parameters are R2's and are **not** yet cited anywhere:
`src/sem_physics.py` states at line 6 that its presets "must be tuned against
the evaluation strata and replaced (or supported) by literature citations before
final submission". That work is outstanding, not filed elsewhere.

**Status of this file:** the references below were located via web search and
are recorded with working URLs. **Each still needs to be opened and the quoted
figure confirmed** before submission — a citation that does not support its
number fails the audit more damagingly than a missing one, because it casts
doubt on the numbers that are correct.

---

## How our dimensions relate to these sources

Our layout dimensions are **not** claimed to replicate any of the processes
below. They are ~2–4x relaxed, deliberately, because the problem statement fixes
the search capture at 10 nm/px and true leading-edge pitches do not survive that
sampling with usable contrast. See `docs/assumptions.md` §4 for the derivation
and the contrast table.

These sources are therefore cited as **the reference points we relaxed away
from**, not as the origin of our values. That distinction must survive into the
deck: it is the difference between an honest engineering decision and a
misattributed number.

---

## Process dimension references

**[1] Intel 14 nm process — fin pitch 42 nm**
Mark Bohr, *14 nm Process Technology: Opening New Horizons*, Intel Developer
Forum, 2014.
https://www.intel.com/content/dam/www/public/us/en/documents/pdf/foundry/mark-bohr-2014-idf-presentation.pdf
- [ ] Confirm 42 nm fin pitch on the slide and note the page number

**[2] Intel 22FFL — fin pitch 45 nm, gate pitch ~108 nm**
IEDM 2017 disclosure, reported by WikiChip Fuse.
https://fuse.wikichip.org/news/567/iedm-2017-intel-details-22ffl-a-relaxed-14nm-process-for-foundry-customers-targets-mobile-and-rf-apps/
- [ ] Confirm both figures
- [ ] Prefer the IEDM paper itself if reachable; WikiChip is secondary

**[3] Intel 22 nm SoC platform — CONFIRMED from the primary PDF**
C.-H. Jan et al., *A 22nm SoC Platform Technology Featuring 3-D Tri-Gate and
High-k/Metal Gate, Optimized for Ultra Low Power, High Performance and High
Density SoC Applications*, IEDM 2012.
https://people.eecs.berkeley.edu/~pister/147fa14/Resources/Intel22nm.pdf

Figures extracted from the paper's own text and tables:

| Parameter | Value | Quoted as |
|---|---|---|
| Gate pitch | **90 nm** | "on a 90nm pitch with a 30nm and 34nm gate length" |
| Gate pitch, relaxed | **108 nm** | "increase the gate pitch from 90nm to 108nm" |
| Fin pitch | **60 nm** | table row `Fin 60` |
| M1 pitch | **90 nm** | table row `M1 90 SAV ULK CDO` |
| Gate length | **30 / 34 / 40 nm** | table row `Lgate (nm) 30 34 34 40` |

- [x] **Confirmed.** Prose figures are quoted directly; the fin, M1 and Lgate
      values come from table rows, so re-read the table visually to be certain
      of column alignment before the deck.

**[4] 22 nm lithography process — consolidated node figures**
WikiChip.
https://en.wikichip.org/wiki/22_nm_lithography_process
- [ ] Secondary source; use only to cross-check [2] and [3]

**[5] IRDS More Moore — roadmap tables for fin pitch, CPP, gate length**
IEEE International Roadmap for Devices and Systems, 2023 and 2024 editions.
https://irds.ieee.org/images/files/pdf/2023/2023IRDS_MM.pdf
https://irds.ieee.org/images/files/pdf/2024/2024IRDS_MM.pdf
- [ ] Locate the More Moore table and record the specific table number and year
      column used. **Note:** the PDF resisted automated text extraction; open it
      manually.

**[5a] IRDS DRAM ground rules — CONFIRMED from the primary PDF**
IEEE IRDS 2023, *More Moore*, §5.1 "DRAM", page 28.
https://irds.ieee.org/images/files/pdf/2023/2023IRDS_MM.pdf

> "…as the capacitor of DRAMs having the ground rules between **48nm and 30nm
> half-pitch**."

and page 29:

> "lines still require **~20nm half-pitch** that is only achievable by 193i
> lithography with double patterning."

So current DRAM sits at **30–48 nm half-pitch**, i.e. a **60–96 nm pitch**.
Ours is 144–216 nm pitch (72–108 nm half-pitch), roughly **2.3–3.6x relaxed**.
Page 28 also defines the cell size factor `a = [DRAM cell size] / [DRAM half
pitch]^2`, with 6F2 current and 4F2 the practical limit.

- [x] **Confirmed** by text extraction from the PDF. This is the primary DRAM
      source; the secondary items below are now redundant and kept only as
      cross-checks.

**[6] DRAM node definition — node = half the Metal 1 pitch**
Tokyo Electron, *Telescope Magazine*, "What Exactly Does the 14 nm Dimension
Correspond to?"
https://www.tel.com/museum/magazine/material/150227_report04_01/
> "In a dynamic random access memory (DRAM), half the pitch between metal lines
> in the lowermost interconnect layer (called Metal 1 or M1 layer) is used as a
> scaling indicator, also known as technology node."
- [x] **Fetched and confirmed.** Gives the definition, not a dimension table.

**[7] DRAM dimensions — secondary, figures NOT confirmed**
- SK Hynix 30-nm class: wordline pitch measured at **88 nm** in the bitline
  direction. https://www.eetimes.com/hynix-dram-layout-process-integration-adapt-to-change/
  - [ ] **Fetch timed out; figure comes from a search summary.** Open manually
        and confirm before use.
- Node shorthand ≈ active-area half-pitch, 1X ≈ 18 nm, 1Y ≈ 17 nm (so ~36 nm
  pitch at 1X).
  - [ ] Attribution unclear from search; find the source or drop the figure.
- https://blog.entegris.com/dram-device-fabrication (30–40 nm half-pitch class)
  - [ ] Secondary, vendor blog. Cross-check only.

> **Honest status:** no primary DRAM process paper secured after several
> attempts. What *is* solid is the definition [6], which is enough to support
> the relaxation argument qualitatively — our 180 nm pitch is several times any
> recent DRAM node however the node is defined. If a primary source cannot be
> found, state the DRAM comparison qualitatively and cite [6] for the
> definition. Do not quote 88 nm or 18 nm until [7] is opened and confirmed.

---

## Numbers that need no external citation

Recorded here so the audit finds an answer rather than a gap. Full reasoning in
`docs/assumptions.md`.

| Number | Class | Justification |
|---|---|---|
| `REF_PX_NM` = 1.0, `SEARCH_PX_NM` = 10.0 | Spec | Given in the problem statement |
| `OUT_SIZE` = 1000, `EXPECTED_IMAGE_SHAPE` | Spec | Given |
| Coordinate convention (origin, axes, pixel centre) | Spec | Image convention; pixel centre at integer |
| `EXTENT_NM` = 12 000 | Derived | Must exceed the 10 µm search field; 2 µm margin |
| `ANCHOR_SPAN_FRACTION` = 0.34 | Derived | Fraction of reference FOV so it tracks `out_size` |
| `DEFAULT_ANCHOR_HALF_SPAN_NM` = 340 | Derived | 1000 px × 1 nm/px × 0.34 |
| `DEFAULT_SUPERSAMPLE` = 4 | Engineering | Anti-aliasing vs render cost |
| `PITCH_TOLERANCE` ±20%, `WIDTH_TOLERANCE` ±15% | Team spec | Work-split document, Phase 2 |
| `POSE_RANGES` small ±5°/±3% | Team spec | Work-split document, Phase 2 |
| `POSE_RANGES` large ±8°/±5% | Engineering | Deliberate stress beyond the tuned range |
| `OVERLAY_BIAS_TOLERANCE_PX` = 0.2 | Measured | Noise floor ~0.10 px; planted 0.5 px reads −0.46 |
| `DEFAULT_SEED` = 20260807 | Engineering | Frozen; pair *i* uses `base_seed + i` |
| Held-out seed 389722107 | Engineering | Drawn with `secrets`; disjoint from tuning range |

---

## Audit checklist for R3

- [ ] Every `SOURCE NEEDED` in `assumptions.md` is resolved or explicitly
      converted to an engineering rationale
- [ ] Every URL above opens and contains the figure attributed to it
- [ ] No citation is attached to a number it does not support — specifically,
      no process reference is presented as the origin of our layout dimensions
- [ ] The relaxation argument (§4 of `assumptions.md`) appears in the deck, not
      just in this repository
