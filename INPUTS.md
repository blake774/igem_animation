# Getting the input files

Three inputs. Two are public, one is yours.

| File | Where from | Needed? |
|---|---|---|
| `input/6w74.pdb` | RCSB | **required** |
| `input/AF-Q13490-F1.pdb` | AlphaFold DB | strongly recommended |
| your design `.cif` | your cluster | strongly recommended |
| your `*_pX0_traj.pdb` | your cluster (rerun) | optional, best-looking |

The pipeline runs with only the first one. Every missing input degrades to a
clearly-labelled fallback rather than an error — but read *What each one
changes* at the bottom before deciding to skip one.

---

## 1 & 2 — the public structures

```bash
bash fetch_inputs.sh
```

Downloads both and verifies them. The check matters: the common failure is
not a network error but a `200 OK` carrying an HTML error page, which lands
on disk as a "PDB file" that every tool downstream reads as zero atoms.

If your network blocks either host, download in a browser:

- **6W74** — <https://www.rcsb.org/structure/6W74> → *Download Files* →
  *PDB Format*. Save as `input/6w74.pdb`.
- **Full-length cIAP1** — <https://alphafold.ebi.ac.uk/entry/Q13490> →
  *Download* → *PDB file*. Save as `input/AF-Q13490-F1.pdb`.

Direct URLs, if you prefer `wget` on another machine:

```
https://files.rcsb.org/download/6W74.pdb
https://alphafold.ebi.ac.uk/files/AF-Q13490-F1-model_v4.pdb
```

Both `.pdb` and `.cif` work — `prep_shrink.py` reads either.

---

## 3 — your design, off the Rice cluster

Rice's shared HPC cluster is **NOTS**, login node `nots.rice.edu`. If your
group runs somewhere else, substitute the hostname — everything below is
otherwise cluster-agnostic. Two things worth confirming rather than assuming,
since they vary between clusters and between allocations on the same one:

```bash
echo "$SCRATCH"      # may be empty; then use /scratch/$USER
sinfo -s             # the real GPU partition name for the SLURM script below
```

### Find the files

SSH in and search rather than guessing paths — RFdiffusion and AF3 output
locations depend entirely on how the job was configured.

```bash
ssh YOUR_NETID@nots.rice.edu

# Designs. AF3/Boltz/Chai all write mmCIF; this catches every naming scheme.
find "$HOME" /scratch/$USER -maxdepth 8 \
     \( -name '*binder*model*.cif' -o -name '*model_0.cif' -o -name 'model.cif' \) \
     -newermt '2024-01-01' -printf '%TY-%Tm-%Td %10s  %p\n' 2>/dev/null | sort -r | head -40

# RFdiffusion backbones and their .trb metadata (the .trb records the contig,
# which is how you confirm a design really is 6W74-conditioned)
find "$HOME" /scratch/$USER -maxdepth 8 -name '*.trb' \
     -printf '%TY-%Tm-%Td  %p\n' 2>/dev/null | sort -r | head -20

# Existing trajectories, if any run already had write_trajectory on
find "$HOME" /scratch/$USER -maxdepth 8 -name '*_traj.pdb' \
     -printf '%10s  %p\n' 2>/dev/null | sort -rn | head -20
```

The design you want matches the name in the original plan —
`ciap1_run1_chunk0_ciap1_binder_0_model_0.cif` — or whatever your best
scoring row in `filter_report.csv` points at.

> On most clusters `/scratch` is purged on a schedule. If the run is old,
> check `/scratch` **first**; it is the copy most likely to disappear.

### Copy them down

Run this **on your laptop**, not on the cluster:

```bash
cd /path/to/igem_animation

scp YOUR_NETID@nots.rice.edu:'/path/from/find/above/*model_0.cif' input/
scp YOUR_NETID@nots.rice.edu:'/path/from/find/above/*_pX0_traj.pdb' input/
```

For a whole output directory, `rsync` is better — it resumes, and `-z` helps
because trajectory PDBs are large and very compressible:

```bash
rsync -avzP YOUR_NETID@nots.rice.edu:/scratch/YOUR_NETID/rfdiff_out/ ./input/rfdiff_out/
```

If NOTS requires two-factor on every connection, open one master connection
and let the copies reuse it:

```bash
ssh -fNM -o ControlPath=~/.ssh/cm-%r@%h:%p YOUR_NETID@nots.rice.edu
scp -o ControlPath=~/.ssh/cm-%r@%h:%p YOUR_NETID@nots.rice.edu:'...' input/
ssh -O exit -o ControlPath=~/.ssh/cm-%r@%h:%p YOUR_NETID@nots.rice.edu
```

---

## 4 — the real diffusion trajectory (optional, and the biggest upgrade)

Without it, shot 4 is a synthetic reverse-diffusion schedule that looks
convincing but is not your data. With it, the shot is literally your binder
being designed. It costs a few GPU-minutes: rerun **one** design with
trajectory output on.

The only change to your existing command is the last line:

```bash
python3 scripts/run_inference.py \
    inference.output_prefix=traj_rerun/ciap1_binder \
    inference.input_pdb=6W74_target_clean.pdb \
    'contigmap.contigs=[...your original contig...]' \
    inference.num_designs=1 \
    inference.write_trajectory=True
```

Use the **same contig, same checkpoint, same seed** as the design you are
handing to `--design`, or the trajectory will condense into a different
protein than the one the shot reveals. `prep_shrink.py` will notice — it
superposes the trajectory's folded end onto the placed binder and warns above
3 Å RMSD — but it cannot fix it.

A SLURM wrapper, adjust partition and account to your allocation:

```bash
#!/bin/bash
#SBATCH --job-name=rfd_traj
#SBATCH --partition=gpu           # check `sinfo -s` for the real name
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=rfd_traj_%j.log

module purge
module load GCC/12.2.0 CUDA/12.0.0      # `module spider cuda` to confirm
source ~/miniconda3/etc/profile.d/conda.sh
conda activate SE3nv                     # your RFdiffusion env

cd ~/RFdiffusion
python3 scripts/run_inference.py \
    inference.output_prefix=${SCRATCH:-/scratch/$USER}/traj_rerun/ciap1_binder \
    ... \
    inference.write_trajectory=True
```

Submit with `sbatch rfd_traj.sh`, watch with `squeue -u $USER`.

Trajectories land in a `traj/` subdirectory beside the output prefix:

```
<output_prefix dir>/traj/ciap1_binder_0_pX0_traj.pdb     <- use this one
<output_prefix dir>/traj/ciap1_binder_0_Xt-1_traj.pdb
```

Use the **pX0** file. `Xt-1` is the noised input at each step and looks like
static rather than a fold resolving.

Don't worry about which direction it is written in — `prep_shrink.py`
measures backbone regularity at both ends and orients it itself.

---

## Running it once you have them

```bash
python3 prep_shrink.py \
    --design input/ciap1_run1_chunk0_ciap1_binder_0_model_0.cif \
    --receptor input/6w74.pdb \
    --fulllength input/AF-Q13490-F1.pdb \
    --rfd-traj input/ciap1_binder_0_pX0_traj.pdb

python3 validate.py        # 20 geometric checks
python3 lint_cxc.py        # script consistency

python3 render_blender.py --quality preview      # path-traced
chimerax --offscreen --script render_shrink.cxc  # or ChimeraX
```

Then read `output/metrics.json` — `design_placement.rmsd` under ~1.5 Å means
the design really is 6W74-conditioned, and `core_contacts_retained` should be
`[312, 313, 314, 325]`.

---

## What each one changes

**No `--fulllength`.** Shot 1 becomes the 87-residue BIR3 domain instead of
the 618-residue protein. The counters stay truthful, but the visual shrink
mostly disappears — BIR3 is *smaller* than the 132-residue binder. This is
the one worth chasing.

Note that AlphaFold's full-length cIAP1 has long low-confidence linkers
between the BIR domains that will read as spaghetti in a ribbon. Consider
leaning on the surface representation, or trimming below pLDDT 50 —
AlphaFold writes pLDDT into the B-factor column, so
`awk '$1=="ATOM" && $11+0 >= 50' AF-Q13490-F1.pdb > trimmed.pdb` is enough.

**No `--design`.** `make_testdata.py` builds a placeholder: a coarse-grained
chain relaxed under bond, α-helical, excluded-volume and ligand terms, with
the real 312–325 motif pinned to its crystal coordinates. It is the right
size and shape and nothing else. Fine for checking timing, not for showing
anyone.

**No `--rfd-traj`.** Shot 4 falls back to a synthetic variance-preserving
schedule. It looks convincing, and `metrics.json` will say
`"source": "synthetic"`. If a judge asks what shot 4 shows, the honest answer
is then "a schematic of reverse diffusion" rather than "our trajectory."
