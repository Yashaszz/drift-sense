# Citations — layout and geometry (R1)

Compiled and audited by R3 against the code (work-split, rule 5: the person who
picks a number writes its citation). Physics citations are R2's and live beside
their presets in `src/sem_physics.py`.

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

**[3] Intel 22 nm SoC platform — tri-gate, fin dimensions**
*A 22nm SoC Platform Technology Featuring 3-D Tri-Gate and High-k/Metal Gate*,
IEDM 2012.
https://people.eecs.berkeley.edu/~pister/147fa14/Resources/Intel22nm.pdf
- [ ] Confirm fin pitch and gate pitch

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
