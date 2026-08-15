# RGB optical bonus — what visible light can and cannot see

Owner: R1. Implementation in `src/optical.py`, tests in `tests/test_optical.py`.

## The short version

The bonus asks for a three-channel dataset with diffraction-limited blur. Those
two requirements are in tension, and working the physics out first changed what
the deliverable should be:

> **Visible light cannot resolve this layout.** Not "resolves it poorly" —
> transmits it at exactly zero contrast. Roughly 94% of the structural signal is
> gone, and what survives is the coarse envelope rather than the device lattice.

That is the finding. It is not a modelling shortcut, and it is the reason the
problem is posed with an SEM in the first place.

## Why

Incoherent imaging through a circular aperture passes no spatial frequency above

```
f_c = 2 NA / lambda        finest surviving period = lambda / (2 NA)
```

At NA 0.95 — about the ceiling for a dry objective — that limit is:

| Channel | Wavelength | Finest period resolved |
|---|---|---|
| Blue | 450 nm | **237 nm** |
| Green | 550 nm | **290 nm** |
| Red | 640 nm | **337 nm** |

Against the dataset's own dimensions:

| Feature | Period | Survives? |
|---|---|---|
| FinFET fin pitch | 72–108 nm | **No** — at any wavelength, at any NA |
| DRAM pitch | 144–216 nm | **No** at NA 0.95; marginal at NA 1.35 in blue |
| FinFET gate pitch | 336–504 nm | Blue and green yes, red marginal |

Oil immersion does not rescue it: NA 1.35 only pulls the blue limit to 167 nm,
still coarser than the fin pitch.

## Measured, not asserted

Sinusoidal gratings through `render_rgb`, Michelson contrast per channel:

| Period | R | G | B |
|---|---|---|---|
| 73 nm | 0.0000 | 0.0000 | 0.0000 |
| 205 nm | 0.0000 | 0.0000 | 0.0000 |
| 256 nm | 0.0000 | 0.0000 | 0.0243 |
| 341 nm | 0.0018 | 0.0694 | 0.1937 |
| 512 nm | 0.2275 | 0.3206 | 0.4328 |

The zeros are exact. At 256 nm only blue passes, because 256 sits between the
blue cutoff (237) and the green one (290) — the channels separate exactly where
the physics says they should. The green figure at 512 nm matches the analytic
OTF (0.32058) to four decimals.

On actual rendered layouts rather than gratings:

| Architecture | Geometry std | R | G | B | Best channel retains |
|---|---|---|---|---|---|
| DRAM | 0.452 | 0.013 | 0.019 | 0.026 | **5.7%** |
| FinFET | 0.448 | 0.016 | 0.022 | 0.030 | **6.8%** |

## What this means for the dataset

An optical DRAM field is close to uniform grey. Generating optical pairs and
reporting an accuracy number on them would be reporting on noise — the answer is
not recoverable because the information is not present, which is a different
statement from "the matcher failed".

So the bonus is delivered as **a measured limit with a working renderer**, not as
an accuracy table. Three things are worth stating in the deck:

1. **Optical alignment cannot address this problem at these dimensions.** The
   numbers above are checkable from the shipped code.
2. **The wavelength dependence is real and measurable.** Cutoff scales as
   `1/lambda`, so blue carries strictly more structure than red — visible in
   both the grating sweep and the rendered layouts. Three channels that behaved
   identically would make the dataset greyscale in disguise.
3. **This is why the SEM chain exists.** At 10 nm/px with a 12 nm PSF the search
   capture retains 45–94% contrast (see `assumptions.md` §4). The same layout
   through a visible optic retains ~6%.

## Deliberately not a Gaussian

A Gaussian blur has infinite support in frequency: it attenuates fine structure
without ever removing it. A matcher could then lock onto a lattice at 0.1%
contrast that no real objective would deliver, and the bonus would produce an
accuracy figure for something physically unobservable. The circular-aperture OTF
is exactly zero above cutoff, which is what makes "cannot resolve" a claim rather
than a turn of phrase. It costs one `arccos`.

## Free parameters, stated

- **NA = 0.95.** The one genuinely free choice. The conclusion is insensitive to
  it — see the NA 1.35 note above.
- **Channel centres 640 / 550 / 450 nm.** Band centres, not measured filter
  responses. The bonus rests on the *ordering*, which is insensitive to the
  exact values. Anyone modelling a specific camera should substitute its curves.

Neither is a literature citation and neither is presented as one.

## Contract

`render_rgb` changes intensities only — no resampling, translation, rotation or
crop — the same contract as `apply_sem_chain`. So one `ground_truth.jsonl`
scores both modalities, and `src.evaluate.load_image` already collapses three
channels to luminance, so nothing downstream needs a separate code path.

## The samples

`docs/rgb_optical_bonus.png` is the composite figure, and it is
**contrast-stretched per panel** so the surviving structure is visible at all.
That stretch is honest about the physics but flattering to the image, so the
unstretched files are here too:

```bash
python scripts/render_rgb_samples.py
```

Twelve PNGs in `docs/rgb_samples/`, three per case across DRAM and FinFET,
anchored and unanchored:

| file | what it is |
|---|---|
| `<case>_geometry.png` | the layout as rendered, for reference |
| `<case>_optical.png` | the same field through the optic, **true RGB, unstretched** |
| `<case>_optical_x8.png` | the same, contrast boosted 8x about mid grey |

Most of the `_optical` files look like flat grey rectangles. **That is the
result, not a rendering failure.** The fraction of the geometry's contrast that
survives the diffraction limit, per channel:

| case | R (640 nm) | G (550 nm) | B (450 nm) |
|---|---|---|---|
| DRAM anchored | 2.9% | 4.1% | 5.7% |
| DRAM unanchored | 3.1% | 4.4% | 6.1% |
| FinFET anchored | 3.5% | 5.0% | 6.8% |
| FinFET unanchored | 1.4% | 2.0% | 2.7% |

Blue retains most and red least in every case, which is the ordering the physics
predicts and the reason the bonus rests on it rather than on the absolute
numbers.

The `_x8` files exist because "flat grey" and "broken renderer" look identical
on a slide, and a judge is entitled to ask which one they are looking at. The
boost is in the filename precisely so it can never be mistaken for the real
thing.

The renderer is deterministic — fixed seeds, no time dependence — and
reproduces all twelve files byte-identically.
