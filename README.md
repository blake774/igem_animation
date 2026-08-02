# DigiActivate — E3 ligase shrink shot

Scripts for the `[some mini-binder generation animation of a large E3 ligase shrinking into a smaller one]` beat in the dry lab section of the promo video.

Output is a ~11 s 1080p sequence in five shots: full-length cIAP1 turning → push into the BIR3 pocket → the ligase disintegrates around the retained motif → a de novo binder condenses out of noise → reveal with a ghost of the original behind it.

---

> **Before you record narration, read `README_payload.md`.** The line this shot
> illustrates — "natural E3 ligases are simply too big" — does not survive the
> payload arithmetic against an isolated BIR3 domain, which is the obvious
> alternative and the one 6W74 actually is. The animation still works; the
> narration around it needs changing. `--framing partners` is the default for
> that reason.

## Decisions baked in

**The "before" structure is full-length cIAP1, not 6W74.** 6W74 is the BIR3 domain alone, ~85 modelled residues. Your binder contig is 120–140. If the animation literally shrinks 6W74, it shrinks into something *bigger*, and that is the kind of thing a judge notices. The honest comparison for "natural E3 ligases are simply too big" is full-length cIAP1 at 618 aa (UniProt Q13490) against a ~132 aa binder — roughly a 4.7× reduction. The AlphaFold model uses UniProt numbering, so residues 312–325 line up with your contig without any renumbering, and the retained strip can glow inside the full-length protein in shot 1.

**The motif is the seed, not a leftover.** In the dissolve, `A312–A325`, the PROTAC warhead and the structural zinc never move. Everything else flies apart. The binder then condenses *around* them. That is what your contig actually specifies, and it makes the "shaped to fit the exact same binding pocket" line visible rather than asserted.

**The four hotspots that ignite in shot 2 are G312 / L313 / R314 / E325** — the intersection across all five cIAP1 structures in your docking table, and the four hotspots in `6W74_target_clean.yaml`. Same set, two independent justifications.

**Condensation is a bead trace, not a cartoon.** ChimeraX needs sane backbone geometry to assign secondary structure; during the noisy frames there isn't any, and the ribbon degenerates into spikes. Beads swarming into a fold is both robust at every timestep and the better-looking option.

---

## Prerequisites

- Python 3.9+ with numpy (nothing else — no Biopython, no ProDy)
- ChimeraX 1.5+
- Network access on first run, or the input files placed manually

---

## Quick start

```bash
# No design file yet? Build a placeholder and see the whole thing run:
python3 make_testdata.py
python3 prep_shrink.py --design input/synthetic_design.cif \
    --receptor input/6w74.pdb --rfd-traj input/synthetic_rfd_traj.pdb

# With the real one:
python3 prep_shrink.py --design ciap1_run1_chunk0_ciap1_binder_0_model_0.cif

python3 validate.py           # geometric checks on the trajectories
python3 lint_cxc.py           # catches script errors before a slow render
python3 test_fallbacks.py     # 43 checks over the error paths

chimerax --offscreen --script render_shrink.cxc      # portable renderer
python3 render_blender.py --quality preview          # path-traced renderer

python3 payload_budget.py     # the setup shot, and a claim worth checking
```

Both renderers read the same `output/` and use the same palette, so they cut
against each other. ChimeraX is the one that runs anywhere; Blender is the one
that looks like film. See **Two renderers** below.

`prep_shrink.py` fetches 6W74 and the AlphaFold model, cleans the receptor down to the cIAP1 BIR3 chain + Zn + warhead (renamed to chain A to match your yaml), locates the retained motif in your design, superposes it into the crystal frame, and writes everything into `output/`.

Fully offline:

```bash
python3 prep_shrink.py --design design.cif \
    --receptor 6w74.pdb --fulllength AF-Q13490-F1-model_v4.pdb
```

---

## Use your own diffusion trajectory

This is the single biggest upgrade available and it costs a few GPU-minutes.

Rerun one design through RFdiffusionAA with

```
inference.write_trajectory=True
```

which writes `*_pX0_traj.pdb`, then

```bash
python3 prep_shrink.py --design design.cif --rfd-traj ciap1_binder_0_pX0_traj.pdb
```

The script detects it, reverses it (trajectories are written T→0), resamples to the frame count, and shot 4 becomes your own reverse-diffusion path instead of an effect. `metrics.json` will read `"source": "rfdiffusion"`, and the console stops printing the synthetic-fallback warning.

Without it you get a variance-preserving synthetic schedule: cosine noise decay, spatially correlated noise blurred along the chain so it reads as a polymer rather than a gas, hyperspherical drift coefficients so the cloud resolves instead of pulsing, and a temporal low-pass so nothing strobes. It looks convincing. It is still not your data, and if a judge asks what shot 4 shows, the honest answer with the synthetic version is "a schematic of reverse diffusion" — which is a weaker sentence than "our trajectory."

---

## Files

| File | Role |
|---|---|
| `prep_shrink.py` | builds all structures, trajectories and metrics |
| `structlib.py` | PDB/mmCIF I/O, Kabsch, contacts, ANM — numpy only |
| `00_setup.cxc` | lighting, silhouettes, base representations |
| `10_shots.cxc` | the choreography — edit this to retime |
| `render_shrink.cxc` | master: opens models, records, encodes |
| `render_blender.py` | path-traced renderer, runs headless via `pip install bpy` |
| `lint_cxc.py` | static check on the cxc scripts |
| `validate.py` | geometric checks on the trajectories |
| `test_fallbacks.py` | exercises the fallback code paths |
| `make_testdata.py` | placeholder binder + trajectory, for offline runs |
| `payload_budget.py` | construct size arithmetic + cassette SVG |
| `README_payload.md` | why the "too big" claim fails, and what replaces it |

`output/00_open.cxc` is generated too, and it carries the open order. The
model numbers in `generated.cxc` are positional, so when there is no
full-length AlphaFold model `#1` falls back to the BIR3 domain rather than
being skipped — skipping it would renumber everything below and every
selection would quietly point one model off.

`output/generated.cxc` is written by prep and carries the palette, named selections, playback aliases with frame counts baked in, and the counter text with your real measured numbers. Don't hand-edit it; rerun prep.

---

## Two renderers

**ChimeraX** (`render_shrink.cxc`) is the portable one. `lighting soft` is
real ambient occlusion, silhouettes give it a drawn edge, and `coordset`
plays the trajectories directly. Roughly 1–5 s/frame at 1080p supersample 3,
so about half an hour for the ~315 frames.

**Blender/Cycles** (`render_blender.py`) is the one that looks like film:
path-traced global illumination, depth of field, translucent surfaces with
real transmission. It needs no Blender install and no display —

```bash
pip install bpy                 # ~370 MB wheel, CPython 3.11
python3 render_blender.py --quality preview
```

`--quality thumb|preview|final` trades resolution and samples. On four CPU
cores, `thumb` (480×270) runs about 2 s/frame — roughly eleven minutes for
the whole shot, which makes it usable for checking timing. `final`
(1920×1080, 220 samples) is much slower on CPU and wants a GPU box or a
cluster node; set `scene.cycles.device` accordingly.

Three implementation notes, because each cost a debugging session:

- **The scene is built in nanometres, not Ångströms.** Blender's lights are
  physical and its depth of field is a real thin-lens model. At 1 unit = 1 Å
  a protein is 40 metres wide, lights 50 m away render black, and no f-stop
  produces visible bokeh.
- **Cameras are framed by `fit_distance()`**, which solves the vertical field
  of view for the subject's actual bounding radius. Framing by a multiple of
  the radius of gyration — the obvious thing — pushes the camera inside
  anything whose mass is centrally concentrated, which all of these are.
  It also means the choreography survives swapping the 87-residue BIR3 for
  the 618-residue full-length model without retuning.
- **Geometry is rebuilt per frame, not keyframed.** A trajectory is already
  an explicit coordinate list, so a keyframe layer adds fragile state and
  buys nothing. A frame is a pure function of its index, which is also why
  `--start/--end` resume works.

---

## Tuning

**Timing.** Frame counts live at the top of `10_shots.cxc` and in the prep flags:

```bash
python3 prep_shrink.py --design d.cif --fps 30 \
    --dissolve-sec 1.5 --condense-sec 3.0 --breathe-sec 2.5
```

Prep and the render scripts stay in sync automatically because the aliases in `generated.cxc` carry the counts. If the edit needs to lose time, take it from shots 1 and 5 — shot 4 is what earns the section.

**Render speed.** `supersample 3` in `render_shrink.cxc` is 9× the pixels per frame and the biggest quality lever; drop it to 1 while blocking timing. `lighting multiShadow 64` → 16 while iterating. A final 1080p pass at supersample 3 runs roughly 1–5 s per frame, so budget under an hour for ~340 frames.

**Compositing.** mp4 can't carry alpha. If you want the shot over the animated sequence rather than cut to it, the commented block at the bottom of `render_shrink.cxc` writes a PNG sequence with a transparent background instead.

**Palette.** Sampled from your animation swatches so the molecular shot doesn't colour-jump against the cartoon frames — steel blue ligase, coral motif, teal PROTAC, amber binder. Change them in the `colordef` block that prep writes.

---

## Before you show it to anyone

Read `output/metrics.json` and check:

- `design_placement.rmsd` — motif Cα RMSD after superposition. Under ~1.5 Å means the design really is 6W74-conditioned. If it's high, prep printed a warning and you're probably pointing at the wrong design file.
- `core_contacts_retained` — should be `[312, 313, 314, 325]`. If any are missing, the motif in your design isn't the one in the yaml.
- `binder_ligand_contacts` vs `bir3_ligand_contacts` — the claim on screen is "shaped to fit the exact same binding pocket." If the binder makes far fewer ligand contacts than BIR3 does, that line is overselling and should be softened, or a better design picked from `filter_report.csv`.
- `trajectories.condense.source` — `rfdiffusion` or `synthetic`.

`prep_shrink.py` prints the on-screen counter values at the end so the video team can pull them without opening the JSON.

---

## Troubleshooting

**"Cannot continue without 6W74."** Network blocked. Download `https://files.rcsb.org/download/6W74.pdb` and pass `--receptor`.

**"WARNING: no full-length model."** AlphaFold DB unreachable. Get
`AF-Q13490-F1-model_v4.pdb` and pass `--fulllength`. Without it the opening
shot falls back to BIR3 and the size claim gets much weaker — worth fixing
rather than shipping around. Two things to expect once you do supply it:
AlphaFold's full-length cIAP1 has long low-pLDDT linkers between the BIR
domains that will read as spaghetti in shot 1, so consider trimming below
pLDDT 50 or leaning on the surface rather than the ribbon; and the metaball
surface in `render_blender.py` gets roughly seven times the elements, which
it compensates for by coarsening its resolution automatically.

**"WARNING: best chain X matched only n/4 fingerprint residues."** The chain auto-detection is looking for GLY312 / LEU313 / ARG314 / GLU325. If it can't find them you may have a different 6W74 file or a renumbered one.

**"WARNING: motif RMSD is high."** The sliding-window search couldn't find the motif. Check you passed a cIAP1 design and not a CRBN or VHL one.

**Nothing appears in the render.** Model numbering drifted — `generated.cxc` hard-codes `#1`–`#7` in the order `render_shrink.cxc` opens them. If you added an `open`, rerun `lint_cxc.py`, which checks this.

**Flicker in shot 4.** Prep prints the largest single-frame atom motion. Under ~3 Å is smooth. If it's larger, lower `sigma_max` in `make_condense()` or raise `--condense-sec`.
