# The payload claim doesn't survive its own arithmetic

The dry lab script currently says:

> "DigiActivate packs a lot into one cell: a guide protein, a transcriptional effector, and an E3 ligase binder for every switch. **Natural E3 ligases are simply too big to fit alongside everything else.**"

I priced it out before animating it. The bolded sentence is false for the E3 module you're actually using.

```
DigiActivate payload, 3 switches, VP64 effector

  E3 module           total aa   coding kb    AAV   Lenti   obligate partners
  ------------------------------------------------------------------------------
  de novo binder         2,078        6.24     NO     yes   -
  BIR3                   2,012        6.04     NO     yes   -
  CRBN-midi              2,822        8.47     NO      NO   -
  VHL+EloBC              3,011        9.04     NO      NO   EloB, EloC
  CRBN+DDB1              6,428       19.29     NO      NO   DDB1 (1140 aa)
  cIAP1 (full length)    3,536       10.61     NO      NO   -
```

**The de novo binder construct is 0.20 kb larger than the isolated-BIR3 construct** — 132 aa versus 110 aa per switch, three switches, 3.3% bigger overall. cIAP1 BIR3 folds and binds SMAC mimetics on its own; it doesn't need the other 508 residues of BIRC2. So "we made it smaller" is not a claim you can make against the obvious alternative, and 6W74 — the structure your whole cIAP1 pipeline is built on — *is* that alternative.

A second thing fell out of the same table: **dCas9 alone is 4.10 kb, and AAV's usable coding room is about 3.7 kb.** AAV was never available to this construct, with any E3 module, before a single switch was added. If AAV appears anywhere in the wiki or the presentation as a delivery route, it needs to come out.

Reproduce with `python3 payload_budget.py --switches 3 --effector VP64`. Component lengths and their justifications are in the `COMPONENTS` dict; change them and rerun if you disagree with any.

---

## What does hold up

Three arguments survive. The first two are the real ones.

**1. Orthogonality — the architectural argument.** DigiActivate's premise is an *array* of independently addressable switches: "PROTAC A alone recruits one effector, PROTAC B alone recruits two, A+B recruits three." That needs N mutually orthogonal PROTAC–binder pairs. Nature offers very few. cIAP1 BIR3 and XIAP BIR3 are homologous and both bind SMAC mimetics — your own Phase 1 notes cite 6EXW achieving only 23–32× selectivity between them, which is a preference, not a channel. CRBN needs DDB1 or the 8RQ9 engineering. VHL needs EloB/EloC. Natural domains get you one clean channel, maybe two. De novo design gets you one per warhead you can make. **This is the argument the project is actually built on, and it's the one the animation should carry.**

**2. No endogenous partners — and this one is close to disqualifying for CAR-T.** Your own CRBN background doc says it: cIAP1/2 "regulate both the canonical and the non-canonical NF-κB signaling pathways." Fuse BIR3 to dCas9 and express it in a T cell and you have built a binding sink for SMAC/DIABLO and endogenous IAP antagonists, sitting in the middle of the NF-κB signalling that CAR-T activation and persistence run through. You would be perturbing the pathway your therapeutic depends on, in a construct whose entire selling point is *not* perturbing things you didn't intend to. A de novo binder has no endogenous partner by construction.

**3. Size, but only against the obligate complexes.** CRBN+DDB1 is 19.3 kb. That's past every viral vector there is. VHL+EloBC is 9.0 kb, past lentivirus. So "too big" is true for two of the three natural recruiters — just not for the one in your pipeline.

---

## Suggested script rewrite

Replace:

> "DigiActivate packs a lot into one cell: a guide protein, a transcriptional effector, and an E3 ligase binder for every switch. Natural E3 ligases are simply too big to fit alongside everything else."

With something along these lines:

> "Every switch needs its own binder, and every binder has to answer to a different drug. Nature only gives us a handful — and they cross-react with each other. Worse, they're already busy: bolt cIAP1's pocket onto dCas9 and you've built a sponge for the cell's own IAP signalling. In a T cell, that's the exact pathway your CAR depends on.
>
> So our dry lab team designs new binders from scratch — one for every PROTAC, and none of them the cell has ever seen before."

If you want to keep a size beat, the honest version is a separate line about the other two recruiters: *"Cereblon drags DDB1 along for the ride — that alone blows the cassette past every viral vector we have."* The cassette SVG makes that visible in one frame.

---

## What this does to the shrink animation

Less than you'd think. The visual is "keep the pocket, throw away the protein," which fits the crosstalk argument *better* than it fits the size argument — you're discarding precisely the parts the cell recognises and keeping only the chemistry that touches the drug.

`prep_shrink.py` now takes `--framing`:

- `--framing partners` **(default)** — counters read `cIAP1 · 618 aa · endogenous IAP / NF-κB regulator` → `de novo binder · 132 aa · no endogenous partners`
- `--framing size` — the original counters, if you decide to keep the size claim anyway

The residue counts are true under both. What changes is which claim the shot is making.

---

## Files

| File | Role |
|---|---|
| `payload_budget.py` | the arithmetic; edit `COMPONENTS` and rerun |
| `output/payload_cassette.svg` | construct diagram to scale, vector limits marked |
| `output/payload_budget.json` | machine-readable, for the wiki |

The SVG is 1600 px wide, white background, drawn with the same palette as the molecular shot. It works as the `[payload visual with all the system components]` placeholder directly, or as a wiki figure. Drop it into Resolve as a still and pan across it, or hand it to whoever's doing the wiki.

## Numbers you may want to change

`--effector VP64` assumes the smallest common activator (56 aa). That's the *generous* assumption — it makes the natural-domain construct look as good as possible, which is the right way round for an argument you're trying to stress-test. VPR is 522 aa per switch and pushes every scenario past lentivirus.

`CRBN_midi` at 380 aa is my estimate for the 8RQ9 construct (DDB1-binding bundle deleted plus 12 stabilising mutations); I didn't have the exact length. If you pull it from the paper, update the dict — it's the one number in the table I'd want checked before it goes on a wiki.
