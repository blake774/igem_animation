# Handoff — retrieving the inputs for the E3-shrink animation

**To whoever picks this up with AF3 / NOTS knowledge.** The animation pipeline
is finished and runs today against a placeholder binder. What it needs from
you is **two structure files off the cluster**. This document specifies them
exactly, gives you a script that verifies a candidate before you copy it, and
lists the things I could not determine from here.

Repo: `blake774/igem_animation`, branch `claude/e3-ligase-animation-a0jgqk`.

---

## TL;DR

| # | File | Where | Status |
|---|---|---|---|
| 1 | `input/6w74.pdb` | RCSB | **already in the repo**, nothing to do |
| 2 | `input/AF-Q13490-F1.pdb` | AlphaFold DB | `bash fetch_inputs.sh` on any machine with internet |
| 3 | **the AF3 design `.cif`** | **NOTS** | **needed — this is the main ask** |
| 4 | **`*_pX0_traj.pdb`** | **NOTS (requires a rerun)** | **needed — see §4** |

Files 3 and 4 must be **the same design**. That is the single easiest thing to
get wrong here, and §4 explains what breaks if they aren't.

Everything degrades gracefully — the pipeline never errors on a missing input,
it substitutes a labelled fallback. So the failure mode is not a crash, it's a
video that quietly shows the wrong thing. Hence the verification steps.

---

## Context: what the animation does

Five shots, ~10.5 s, showing a natural E3 ligase being replaced by a designed
mini-binder:

1. cIAP1 turns (needs **file 2**)
2. push into the PROTAC pocket, hotspot residues ignite (file 1)
3. the ligase disintegrates; residues **A312–325**, the PROTAC warhead and the
   structural Zn stay put (file 1)
4. a binder condenses out of noise around them (needs **file 4**)
5. reveal, with a ghost of the original behind it (needs **files 2 + 3**)

The retained motif is `A312-325` from 6W74, hotspots `G312 / L313 / R314 /
E325`. Everything is superposed into the 6W74 crystal frame.

---

## File 2 — full-length cIAP1

- **UniProt Q13490** (BIRC2 / cIAP1), 618 aa, AlphaFold DB entry `AF-Q13490-F1`
- `.pdb` or `.cif`, either works
- Must be **full-length in UniProt numbering** — shot 1 highlights residues
  312–325 *inside* the full-length protein, so the numbering has to line up
  with 6W74's. AFDB's file already does; a renumbered or trimmed copy will not.
- Save to `input/AF-Q13490-F1.pdb`

`bash fetch_inputs.sh` does this and verifies it. Direct URL:
`https://alphafold.ebi.ac.uk/files/AF-Q13490-F1-model_v4.pdb`

**Why it matters:** without it, shot 1 falls back to the 87-residue BIR3 domain
— which is *smaller* than the 132-residue binder, so the shrink the whole
sequence is about becomes invisible. This is the highest-value single file.

**One caveat for you to judge:** AlphaFold's full-length cIAP1 has long
low-pLDDT linkers between the BIR domains. In a ribbon these read as spaghetti.
The renderer handles the geometry (tubes split at chain breaks, cameras frame
by projection so the linkers don't blow out the shot), but you may still prefer
to trim. pLDDT is in the B-factor column:

```bash
awk '$1=="ATOM" && $11+0 >= 50' AF-Q13490-F1.pdb > AF-Q13490-trimmed.pdb
```

Your call — I'd try untrimmed first and look at shot 1.

---

## File 3 — the AF3 design (the main ask)

### What it must contain

- The **de novo binder** as the longest protein chain. `prep_shrink.py` takes
  `max(chain lengths)` as the binder, so if a target/context chain is longer,
  it picks wrong. Expected binder length ~132 aa.
- The **retained motif A312–325 geometry**, anywhere in the file. It can be:
  - numbered 312–325 in some chain, or
  - a separate 14-residue context chain, or
  - fused anywhere inside the binder chain at any numbering
  — all three are found automatically by sliding-window superposition.
- Format: **mmCIF or PDB**. `auth_*` columns are used when present, `label_*`
  otherwise, so AF3 / Boltz / Chai output all read fine.

### What it does NOT need

- **The PROTAC ligand and the Zn are not needed.** They are always taken from
  the 6W74 crystal, so a protein-only prediction is fine.
- Hydrogens, multiple models, alternate locations — all handled/stripped.

### The one hard requirement

The design must be **6W74-conditioned** — generated with a contig that retained
`A312-325`. The acceptance threshold is **motif Cα RMSD < 1.5 Å** after
superposition onto 6W74. A CRBN or VHL design, or a cIAP1 design from a
different crystal, will not pass.

### Verify before you copy

`check_design.py` needs only `numpy` + `structlib.py`, so it runs on a NOTS
login node with no environment. Copy three files up:

```bash
scp structlib.py check_design.py input/6w74.pdb YOUR_NETID@nots.rice.edu:~/dcheck/
```

Then sweep every candidate at once:

```bash
cd ~/dcheck
python3 check_design.py '/path/to/af3_out/*/*model*.cif' --receptor 6w74.pdb
```

A good file looks like:

```
  protein chains {'B': 132}
  [ok  ] motif A312-325 found on chain B at residue 60, Ca RMSD 0.31 A
  [ok  ] binder chain B: 132 aa
```

It rejects: files with no motif, fragments under 60 aa, and — a mistake worth
guarding — the cleaned target structure handed over by accident, which passes
every other check but is 100% superposable onto the receptor.

Exit status 0 if at least one candidate passes, so it scripts.

### Finding candidates

I don't know your AF3 output layout. These catch the common ones:

```bash
find "$HOME" /scratch/$USER -maxdepth 8 \
     \( -name '*binder*model*.cif' -o -name '*model_0.cif' -o -name 'model.cif' \
        -o -name 'ranked_0.cif' -o -name '*_model.cif' \) \
     -newermt '2024-01-01' -printf '%TY-%Tm-%Td %10s  %p\n' 2>/dev/null | sort -r | head -40
```

The name from the original project notes is
`ciap1_run1_chunk0_ciap1_binder_0_model_0.cif`. If there's a
`filter_report.csv` from the design pipeline, its best-scoring row is the one
to prefer — but run `check_design.py` on it regardless.

> `/scratch` is purged on a schedule on most clusters. If the run is old,
> check `/scratch` first — it's the copy most likely to vanish.

---

## File 4 — the RFdiffusion trajectory (requires a rerun)

This is the payoff shot. Shot 4 becomes the actual reverse-diffusion path that
produced the binder, instead of a synthetic schedule. A few GPU-minutes.

### What to run

Rerun **one** design with trajectory output on. The only change to the original
command:

```
inference.write_trajectory=True
```

**Use the same contig, same checkpoint, and same seed as file 3.** If the
trajectory came from a different design, shot 4 condenses into one protein and
shot 5 reveals a different one. `prep_shrink.py` detects this — it superposes
the trajectory's folded end onto the placed binder and warns above 3 Å RMSD —
but it cannot fix it.

If reproducing the exact seed is impractical, the cleaner option is to **run
the new design through AF3 as well** and hand over *that* pair as files 3+4.
A matched pair from a fresh run beats a mismatched pair from the original.

### Which file

Trajectories land in a `traj/` subdirectory beside the output prefix:

```
<outdir>/traj/<prefix>_0_pX0_traj.pdb     <- use this
<outdir>/traj/<prefix>_0_Xt-1_traj.pdb    <- not this
```

**pX0** is the running prediction of the final structure — it resolves from
noise into a fold, which is the shot. **Xt-1** is the noised input at each
step and looks like static throughout.

### Requirements on the file

- Multi-`MODEL` PDB, **≥5 models**, **identical atom count in every model**
  (ragged files are detected and rejected with a message)
- Cα-only or all-atom, both fine
- May include context/motif residues; the atom counts not matching the binder
  is handled by sliding-window superposition
- **Frame direction does not matter.** Don't try to reverse it. The pipeline
  measures backbone regularity at both ends and orients it itself — that
  convention has changed between RFdiffusion releases and differs between the
  pX0 and Xt files, so it is deliberately not trusted.

### SLURM sketch — please correct the guesses

I could not verify partition names, module names, or whether `$SCRATCH` is
defined for this allocation. Check first:

```bash
echo "$SCRATCH"      # may be empty; then use /scratch/$USER
sinfo -s             # real partition names
module spider cuda   # real module names
```

```bash
#!/bin/bash
#SBATCH --job-name=rfd_traj
#SBATCH --partition=gpu          # <-- verify with sinfo -s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=rfd_traj_%j.log

module purge
module load GCC/12.2.0 CUDA/12.0.0    # <-- verify with module spider
source ~/miniconda3/etc/profile.d/conda.sh
conda activate SE3nv                   # <-- your RFdiffusion env

cd ~/RFdiffusion
python3 scripts/run_inference.py \
    inference.output_prefix=${SCRATCH:-/scratch/$USER}/traj_rerun/ciap1_binder \
    inference.input_pdb=/path/to/6W74_target_clean.pdb \
    'contigmap.contigs=[ ...ORIGINAL CONTIG... ]' \
    inference.num_designs=1 \
    inference.write_trajectory=True
```

**Optional but ideal:** if the design pipeline is cheap to rerun end-to-end,
regenerating one design *with* `write_trajectory=True` and then folding it with
AF3 gives a guaranteed-matched files 3+4 and removes the whole seed-matching
problem.

---

## Copying down

Run on the laptop, not the cluster:

```bash
cd /path/to/igem_animation
scp YOUR_NETID@nots.rice.edu:'/verified/path/*model_0.cif'      input/
scp YOUR_NETID@nots.rice.edu:'/verified/path/*_pX0_traj.pdb'    input/
```

If NOTS 2FAs every connection, multiplex so you authenticate once:

```bash
ssh -fNM -o ControlPath=~/.ssh/cm-%r@%h:%p YOUR_NETID@nots.rice.edu
scp -o ControlPath=~/.ssh/cm-%r@%h:%p YOUR_NETID@nots.rice.edu:'...' input/
ssh -O exit -o ControlPath=~/.ssh/cm-%r@%h:%p YOUR_NETID@nots.rice.edu
```

Trajectory PDBs are large and very compressible — `rsync -avzP` for a whole
output directory.

---

## Acceptance test

With all four files in place:

```bash
python3 prep_shrink.py \
    --design    input/YOUR_DESIGN.cif \
    --receptor  input/6w74.pdb \
    --fulllength input/AF-Q13490-F1.pdb \
    --rfd-traj  input/YOUR_pX0_traj.pdb

python3 validate.py        # expect 20/20
python3 lint_cxc.py        # expect 0 errors
```

Then check these in `output/metrics.json`:

| Key | Expected | If wrong |
|---|---|---|
| `design_placement.rmsd` | **< 1.5 Å** | design isn't 6W74-conditioned |
| `core_contacts_retained` | `[312, 313, 314, 325]` | wrong motif in the design |
| `full_length_residues` | `618` | file 2 is trimmed or renumbered |
| `binder_residues` | ~132 | wrong chain picked as binder |
| `shrink_factor` | ~4.7× | follows from the two above |
| `trajectories.condense.source` | `"rfdiffusion"` | file 4 rejected — read the console |
| `trajectories.condense.superposition_rmsd` | **< 3 Å** | files 3 and 4 are different designs |
| `trajectories.condense.max_frame_jump_A` | **< 3 Å** | raise `--condense-sec` |
| `binder_ligand_contacts` vs `bir3_ligand_contacts` | comparable | see below |

That last one is a claim check, not a bug check. The on-screen line is "shaped
to fit the exact same binding pocket." If the binder makes far fewer warhead
contacts than BIR3's 11, that line is overselling and either the line softens
or a better design gets picked.

Then render:

```bash
python3 render_blender.py --quality preview      # path-traced; pip install bpy
chimerax --offscreen --script render_shrink.cxc  # portable alternative
```

---

## Things I could not determine — please correct rather than work around

1. **AF3 output layout and naming** on your setup, and whether it was AF3
   proper, Boltz, or Chai. The reader handles all three dialects; I just don't
   know where the files live.
2. **NOTS specifics** — I have `nots.rice.edu` as the login node and everything
   else in the SLURM script is a placeholder. Partition, modules, `$SCRATCH`,
   and the RFdiffusion conda env name are all guesses.
3. **The original contig string.** It should be in the `.trb` beside the
   original RFdiffusion output, or in the design pipeline config.
4. **Whether the seed for the original design is recoverable.** If not, see the
   "regenerate a matched pair" option in §4.
5. **ChimeraX scripts are unverified.** I had no ChimeraX in my environment, so
   `00_setup.cxc` / `10_shots.cxc` / `render_shrink.cxc` are lint-clean but
   never executed. `render_blender.py` I ran end to end — that one is verified.

---

## What is placeholder right now

`make_testdata.py` builds a stand-in binder: a coarse-grained Cα chain relaxed
under bond, α-helical, excluded-volume and ligand terms, with the real 312–325
motif pinned to its crystal coordinates. It is the right size and shape and
**nothing else** — not folded, not scored, not validated. It exists so the
pipeline runs and the timing can be checked. Everything else in the current
render is genuine 6W74.

Replacing it is just `--design`. No code changes.

---

## If you get stuck

Every fallback is deliberate and labelled, so a partial delivery is still
useful:

- **File 2 only** → correct opening shot, placeholder binder. Still worth it.
- **File 3 only** → real binder, synthetic condensation, weak opening shot.
- **Files 2+3** → everything real except shot 4, which is honestly describable
  as "a schematic of reverse diffusion."
- **All four** → shot 4 is your actual trajectory.

`python3 test_fallbacks.py` runs 54 checks over these paths if you change
anything and want to know you didn't break them.
