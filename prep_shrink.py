#!/usr/bin/env python3
"""
prep_shrink.py -- build every structural asset for the DigiActivate
"E3 ligase shrinks into a de novo mini-binder" shot.

What it does
------------
 1. Gets full-length cIAP1 (AlphaFold DB, UniProt Q13490) and the crystal
    receptor 6W74, or uses local copies.
 2. Cleans 6W74 down to the cIAP1 BIR3 chain + structural Zn + the PROTAC
    warhead, renames it to chain A so it matches 6W74_target_clean.yaml.
 3. Loads the AF3 design, works out where the retained motif (A312-A325) is,
    and superposes the design onto the crystal frame.
 4. Measures the things that go on screen: residue counts, motif RMSD,
    ligand-contact residues, radius of gyration.
 5. Writes three coordinate trajectories:
      - dissolve : BIR3 disintegrates, motif + ligand survive
      - condense : binder condenses out of noise around the surviving motif
                   (real RFdiffusion trajectory if you have one, otherwise a
                    variance-preserving synthetic schedule)
      - breathe  : Ca-ANM oscillation so the final complex isn't dead-still
 6. Emits metrics.json and selections.cxc for the ChimeraX render scripts.

Usage
-----
  python3 prep_shrink.py --design ciap1_run1_chunk0_ciap1_binder_0_model_0.cif

  # with a real RFdiffusion trajectory (much better than the synthetic one):
  python3 prep_shrink.py --design design.cif --rfd-traj ciap1_binder_0_pX0_traj.pdb

  # fully offline
  python3 prep_shrink.py --design design.cif \
      --receptor 6w74.pdb --fulllength AF-Q13490-F1-v4.pdb
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace

import numpy as np

import structlib as S

# --------------------------------------------------------------------------
# project constants -- these come straight from the dry lab pipeline doc
# --------------------------------------------------------------------------

UNIPROT_CIAP1 = "Q13490"           # BIRC2 / cIAP1, 618 aa
RECEPTOR_PDB = "6W74"              # cIAP1 BIR3 + compound 15, binary
MOTIF_RANGE = (312, 325)           # retained strip, 6W74_target_clean.yaml
HOTSPOTS = {312: "CA", 313: "CG", 314: "CD", 325: "CB"}
MOTIF_FINGERPRINT = {312: "GLY", 313: "LEU", 314: "ARG", 325: "GLU"}
CORE_CONTACTS = [312, 313, 314, 325]   # intersection across all cIAP1 PDBs

AA3 = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "MSE", "SEC", "PYL", "HSD", "HSE", "HSP",
}


def log(msg: str) -> None:
    print(f"  {msg}", flush=True)


def head(msg: str) -> None:
    print(f"\n[{msg}]", flush=True)


# --------------------------------------------------------------------------
# step 1 -- acquire
# --------------------------------------------------------------------------

def acquire(args) -> tuple[str | None, str | None]:
    head("acquiring structures")
    os.makedirs(args.input_dir, exist_ok=True)

    recep = args.receptor
    if recep and os.path.exists(recep):
        log(f"receptor (local): {recep}")
    else:
        recep = S.fetch_rcsb(RECEPTOR_PDB, args.input_dir)
        log(f"receptor 6W74: {recep or 'FETCH FAILED'}")

    full = args.fulllength
    if full and os.path.exists(full):
        log(f"full-length (local): {full}")
    else:
        full = S.fetch_alphafold(UNIPROT_CIAP1, args.input_dir)
        log(f"full-length cIAP1 (AF-{UNIPROT_CIAP1}): {full or 'FETCH FAILED'}")

    if not recep:
        sys.exit("\nCannot continue without 6W74. Download it manually from\n"
                 "https://files.rcsb.org/download/6W74.pdb and pass --receptor.")
    if not full:
        log("WARNING: no full-length model. The opening 'before' shot will fall")
        log("         back to the BIR3 domain, which weakens the size claim.")
    return recep, full


# --------------------------------------------------------------------------
# step 2 -- clean the receptor
# --------------------------------------------------------------------------

def find_ciap1_chain(st: S.Structure) -> str:
    """Pick the chain carrying the BIR3 motif fingerprint GLY312/LEU313/ARG314/GLU325."""
    best, best_score = None, -1
    for ch in st.chains():
        score = 0
        for num, expect in MOTIF_FINGERPRINT.items():
            for a in st.atoms:
                if a.chain == ch and a.resseq == num and a.record == "ATOM":
                    if a.resname == expect:
                        score += 1
                    break
        if score > best_score:
            best, best_score = ch, score
    if best_score < len(MOTIF_FINGERPRINT):
        log(f"WARNING: best chain {best} matched only {best_score}/"
            f"{len(MOTIF_FINGERPRINT)} fingerprint residues")
    return best


def find_warhead(st: S.Structure, chain: str) -> str | None:
    """The PROTAC warhead = biggest non-junk het group sitting on the chain."""
    prot = [a for a in st.atoms if a.chain == chain and a.record == "ATOM"]
    best, best_n = None, 0
    for (c, seq, ic, name) in st.het_residues():
        if name == "ZN":
            continue
        grp = [a for a in st.atoms
               if a.chain == c and a.resseq == seq and a.resname == name]
        if len(grp) < 10:
            continue
        if S.min_distance(grp, prot) > 5.0:
            continue
        if len(grp) > best_n:
            best, best_n = name, len(grp)
    return best


def clean_receptor(path: str, out_pdb: str) -> tuple[S.Structure, dict]:
    head("cleaning 6W74 -> BIR3 docking receptor")
    raw = S.read_pdb(path)
    log(f"raw: {len(raw)} atoms, chains {''.join(raw.chains())}")

    chain = find_ciap1_chain(raw)
    log(f"cIAP1 BIR3 chain: {chain}")

    warhead = find_warhead(raw, chain)
    log(f"PROTAC warhead het code: {warhead or 'none found'}")

    keep_het = {"ZN"} | ({warhead} if warhead else set())

    def wanted(a: S.Atom) -> bool:
        if a.is_hydrogen() or a.altloc not in ("", "A"):
            return False
        if a.resname in S.WATER:
            return False
        if a.record == "ATOM":
            return a.chain == chain
        return a.resname in keep_het

    st = raw.select(wanted)

    # Ligand / Zn may be logged on a neighbouring chain -- drop copies that
    # belong to a different protein chain than the one we kept.
    prot = [a for a in st.atoms if a.record == "ATOM"]
    drop = set()
    for (c, seq, ic, name) in st.het_residues(skip_junk=False):
        grp = [a for a in st.atoms
               if a.chain == c and a.resseq == seq and a.resname == name]
        if S.min_distance(grp, prot) > 5.0:
            drop.add((c, seq, name))
    if drop:
        log(f"dropping {len(drop)} het group(s) belonging to another copy")
        st = st.select(lambda a: (a.chain, a.resseq, a.resname) not in drop)

    for a in st.atoms:
        a.chain = "A"          # matches 6W74_target_clean.yaml contig "A312-325"
    st.renumber_serials()

    info = {
        "source_chain": chain,
        "warhead": warhead,
        "residues": st.n_protein_residues(),
        "atoms": len(st),
        "has_zinc": any(a.resname == "ZN" for a in st.atoms),
    }
    log(f"clean: {info['residues']} residues, {info['atoms']} atoms, "
        f"Zn={'yes' if info['has_zinc'] else 'no'}")

    S.write_pdb(st, out_pdb, remarks=[
        f"6W74 cIAP1 BIR3 receptor, source chain {chain} renamed to A",
        f"kept: protein + ZN + {warhead or 'no ligand'}; waters/H/altlocs stripped",
    ])
    log(f"wrote {out_pdb}")
    return st, info


# --------------------------------------------------------------------------
# step 3 -- design placement
# --------------------------------------------------------------------------

def split_design(st: S.Structure) -> tuple[list[str], list[tuple]]:
    """Return (protein chain ids, het residue tuples) for a design file."""
    prot_chains, het = [], []
    for ch in st.chains():
        res = [r for r in st.residues() if r[0] == ch]
        aa = sum(1 for r in res if r[3] in AA3)
        if aa >= 5:
            prot_chains.append(ch)
        else:
            het.extend(r for r in res if r[3] not in S.WATER)
    return prot_chains, het


def locate_motif(design: S.Structure, ref_motif_ca: np.ndarray,
                 ref_keys: list) -> dict:
    """
    Work out how the design maps onto the crystal frame.

    Strategy, in order of preference:
      1. a design chain whose residue numbers are literally 312-325
      2. a design chain with exactly len(motif) residues -> match in order
      3. sliding structural window over every protein chain
    """
    n = len(ref_motif_ca)
    prot_chains, _ = split_design(design)

    # 1 -- numbering preserved
    for ch in prot_chains:
        nums = sorted({a.resseq for a in design.atoms
                       if a.chain == ch and a.record == "ATOM"})
        if set(range(MOTIF_RANGE[0], MOTIF_RANGE[1] + 1)).issubset(set(nums)):
            ca = np.array([[a.x, a.y, a.z] for a in design.atoms
                           if a.chain == ch and a.name == "CA"
                           and MOTIF_RANGE[0] <= a.resseq <= MOTIF_RANGE[1]])
            if len(ca) == n:
                R, t, rms = S.kabsch(ca, ref_motif_ca)
                return {"method": "resnum", "chain": ch, "start": MOTIF_RANGE[0],
                        "R": R, "t": t, "rmsd": rms}

    # 2 -- a dedicated context chain of the right length
    for ch in prot_chains:
        ca, keys = design.ca_trace(ch)
        if len(ca) == n:
            R, t, rms = S.kabsch(ca, ref_motif_ca)
            if rms < 3.0:
                return {"method": "context_chain", "chain": ch,
                        "start": keys[0][1], "R": R, "t": t, "rmsd": rms}

    # 3 -- sliding window
    best = None
    for ch in prot_chains:
        ca, keys = design.ca_trace(ch)
        for i in range(0, len(ca) - n + 1):
            R, t, rms = S.kabsch(ca[i:i + n], ref_motif_ca)
            if best is None or rms < best["rmsd"]:
                best = {"method": "window", "chain": ch, "start": keys[i][1],
                        "R": R, "t": t, "rmsd": rms}
    if best is None:
        raise RuntimeError("no protein chain in the design long enough to hold the motif")
    return best


def place_design(design_path: str, ref: S.Structure, out_binder: str,
                 out_seed: str) -> tuple[S.Structure, S.Structure, dict]:
    head("placing the de novo design in the crystal frame")
    design = S.read_structure(design_path)
    log(f"design: {len(design)} atoms, chains {''.join(design.chains())}")

    design = design.select(lambda a: not a.is_hydrogen() and a.altloc in ("", "A"))

    ref_motif = [a for a in ref.atoms
                 if a.record == "ATOM"
                 and MOTIF_RANGE[0] <= a.resseq <= MOTIF_RANGE[1]]
    ref_motif_ca = np.array([[a.x, a.y, a.z] for a in ref_motif if a.name == "CA"])
    ref_keys = [a.reskey for a in ref_motif if a.name == "CA"]
    if len(ref_motif_ca) < 5:
        raise RuntimeError(f"receptor is missing motif residues "
                           f"{MOTIF_RANGE[0]}-{MOTIF_RANGE[1]}")
    log(f"reference motif: {len(ref_motif_ca)} residues "
        f"({MOTIF_RANGE[0]}-{MOTIF_RANGE[1]})")

    fit = locate_motif(design, ref_motif_ca, ref_keys)
    log(f"motif located by '{fit['method']}' on chain {fit['chain']} "
        f"at {fit['start']}, Ca RMSD {fit['rmsd']:.2f} A")
    if fit["rmsd"] > 2.5:
        log("WARNING: motif RMSD is high. Check that --design really is a "
            "6W74-conditioned design, or pass --rfd-trb.")

    S.apply_transform(design, fit["R"], fit["t"])

    # Which chain is the actual de novo binder?
    prot_chains, het = split_design(design)
    lengths = {ch: len({a.reskey for a in design.atoms
                        if a.chain == ch and a.record == "ATOM"})
               for ch in prot_chains}
    log(f"design protein chains: {lengths}")
    binder_chain = max(lengths, key=lambda c: lengths[c])
    motif_in_binder = (fit["chain"] == binder_chain)
    log(f"binder chain: {binder_chain} ({lengths[binder_chain]} aa); "
        f"motif is {'fused into it' if motif_in_binder else 'a separate context chain'}")

    binder = design.select(lambda a: a.chain == binder_chain and a.record == "ATOM")
    binder.rename_chain(binder_chain, "B")
    binder.renumber_serials()
    S.write_pdb(binder, out_binder, remarks=[
        "de novo cIAP1-pocket mini-binder, superposed onto 6W74 frame",
        f"placement: {fit['method']}, motif Ca RMSD {fit['rmsd']:.2f} A",
    ])
    log(f"wrote {out_binder}")

    # The "seed" that survives the dissolve: retained motif + warhead + Zn,
    # always taken from the crystal so it stays in one consistent frame.
    seed = ref.select(
        lambda a: (a.record == "HETATM")
        or (MOTIF_RANGE[0] <= a.resseq <= MOTIF_RANGE[1]))
    seed.renumber_serials()
    S.write_pdb(seed, out_seed, remarks=[
        f"retained motif A{MOTIF_RANGE[0]}-{MOTIF_RANGE[1]} + PROTAC warhead + Zn",
        "this is what survives the dissolve and seeds the condensation",
    ])
    log(f"wrote {out_seed} ({seed.n_protein_residues()} motif residues + het)")

    fit_out = {k: v for k, v in fit.items() if k not in ("R", "t")}
    fit_out["binder_chain_source"] = binder_chain
    fit_out["binder_residues"] = lengths[binder_chain]
    fit_out["motif_fused_in_binder"] = motif_in_binder
    return binder, seed, fit_out


# --------------------------------------------------------------------------
# step 4 -- trajectories
# --------------------------------------------------------------------------

def ease(t: np.ndarray | float, power: float = 3.0):
    """Smooth 0->1 ease-in-out."""
    t = np.clip(t, 0.0, 1.0)
    return np.where(t < 0.5,
                    0.5 * (2 * t) ** power,
                    1 - 0.5 * (2 * (1 - t)) ** power)


def smooth_noise_field(n_atoms: int, n_frames: int, seed: int,
                       n_basis: int = 5, chain_smoothing: int = 5) -> np.ndarray:
    """
    Temporally smooth, spatially correlated noise, shape (n_frames, n_atoms, 3).

    White per-frame noise strobes badly on screen. Instead we build a handful of
    fixed random fields and drift slowly through their span, then blur each field
    along the atom index so neighbouring atoms move together and the thing reads
    as a chain rather than a gas.

    Two details that matter for the render: the drift frequencies stay well under
    one cycle across the whole shot, and the basis coefficients are scaled by a
    constant rather than renormalised per frame -- per-frame renormalisation
    swings the field direction wildly whenever a coefficient crosses zero, which
    is exactly the flicker we are trying to avoid.
    """
    rng = np.random.default_rng(seed)
    basis = rng.normal(size=(n_basis, n_atoms, 3))

    if chain_smoothing > 1:
        k = np.ones(chain_smoothing) / chain_smoothing
        for b in range(n_basis):
            for d in range(3):
                basis[b, :, d] = np.convolve(basis[b, :, d], k, mode="same")
        basis /= (basis.std() + 1e-9)

    # Hyperspherical coefficients: the vector drifts smoothly across the basis
    # while its norm stays exactly 1. Constant magnitude matters because the
    # collapse should read as noise *resolving*, not as the cloud pulsing in
    # and out on its way down.
    t = np.linspace(0.0, 1.0, n_frames)
    freqs = rng.uniform(0.20, 0.70, size=n_basis - 1)   # < 1 cycle over the shot
    phases = rng.uniform(0, 2 * np.pi, size=n_basis - 1)
    angles = np.pi * (0.5 + 0.45 * np.sin(
        2 * np.pi * freqs[None, :] * t[:, None] + phases[None, :]))

    coeff = np.ones((n_frames, n_basis))
    running = np.ones(n_frames)
    for b in range(n_basis - 1):
        coeff[:, b] = running * np.cos(angles[:, b])
        running = running * np.sin(angles[:, b])
    coeff[:, -1] = running
    return np.einsum("fb,bad->fad", coeff, basis)


def temporal_smooth(frames: list[np.ndarray], window: int = 3,
                    pin_last: bool = True) -> list[np.ndarray]:
    """
    Low-pass the trajectory in time. Belt and braces against strobing: whatever
    the schedule does, adjacent frames end up within a fraction of an Angstrom
    of each other's motion. The final frame is pinned so the shot still lands
    exactly on the deposited design.
    """
    if window < 2 or len(frames) < window + 2:
        return frames
    arr = np.stack(frames)
    pad = window // 2
    padded = np.concatenate(
        [np.repeat(arr[:1], pad, axis=0), arr, np.repeat(arr[-1:], pad, axis=0)])
    kernel = np.ones(window) / window
    out = np.empty_like(arr)
    for d in range(3):
        out[..., d] = np.apply_along_axis(
            lambda v: np.convolve(v, kernel, mode="valid"), 0, padded[..., d])
    if pin_last:
        out[-1] = arr[-1]
        out[0] = arr[0]
    return [out[i] for i in range(len(out))]


def chain_disorder(xyz: np.ndarray) -> float:
    """
    Mean absolute deviation of consecutive-atom spacing from the ideal Ca-Ca
    3.81 A. Near zero for a folded backbone, large for a noised one. Used to
    work out which end of a diffusion trajectory is the noise, rather than
    trusting a convention that varies by release.

    Trajectory files are Ca-only in practice; on an all-atom file the value is
    still monotonic in noise level, which is all the comparison needs.
    """
    if len(xyz) < 3:
        return 0.0
    d = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    return float(np.abs(d - 3.81).mean())


def resample_frames(frames: list[np.ndarray], n_out: int) -> list[np.ndarray]:
    """
    Retime a trajectory to exactly n_out frames by linear interpolation.

    Picking the nearest source frame instead -- which is the obvious thing to
    write -- stutters badly whenever n_out is not a multiple of the source
    length: some output frames are duplicates (no motion at all) and their
    neighbours carry a full double step. On a 48 -> 90 upsample that roughly
    doubles the largest per-frame jump, which is exactly the strobing the
    render is trying to avoid.

    Interpolating between diffusion steps is a retime, not a fabrication: the
    source frames are all still hit, and nothing between them is claimed to be
    a sampled state. The remark written into the PDB says so.
    """
    n_src = len(frames)
    if n_src == 0:
        return []
    if n_src == 1:
        return [frames[0].copy() for _ in range(n_out)]
    out = []
    for i in range(n_out):
        pos = i * (n_src - 1) / max(n_out - 1, 1)
        lo = int(np.floor(pos))
        hi = min(lo + 1, n_src - 1)
        w = pos - lo
        out.append(frames[lo] * (1.0 - w) + frames[hi] * w)
    return out


def superpose_trajectory(frames: list[np.ndarray], template: S.Structure,
                         binder: S.Structure):
    """
    Find the rigid transform putting the folded end of a diffusion trajectory
    onto the already-placed binder.

    Returns (R, t, rmsd, description) or None.

    Three cases, because trajectory files vary: the trajectory may be Ca-only
    or all-atom, and it may or may not carry the motif/context residues that
    the design file also has. Equal Ca counts is the easy case; otherwise the
    shorter trace is slid along the longer one and the best window wins, the
    same trick locate_motif() uses.
    """
    folded = frames[-1]

    tmpl_ca = [i for i, a in enumerate(template.atoms) if a.name.strip() == "CA"]
    if not tmpl_ca:
        # Some trajectory writers emit a bare backbone trace with no atom
        # names at all; then every point is a trace point.
        tmpl_ca = list(range(len(template.atoms)))
    traj_ca = folded[tmpl_ca]

    binder_ca, _ = binder.ca_trace()
    if len(binder_ca) < 3 or len(traj_ca) < 3:
        return None

    if len(traj_ca) == len(binder_ca):
        R, t, rms = S.kabsch(traj_ca, binder_ca)
        return R, t, rms, "1:1 Ca match"

    short, long_, flipped = ((traj_ca, binder_ca, False)
                             if len(traj_ca) < len(binder_ca)
                             else (binder_ca, traj_ca, True))
    n = len(short)
    best = None
    for i in range(len(long_) - n + 1):
        window = long_[i:i + n]
        P, Q = (window, short) if flipped else (short, window)
        R, t, rms = S.kabsch(P, Q)
        if best is None or rms < best[2]:
            best = (R, t, rms, f"best {n}-residue window at offset {i} "
                               f"({len(traj_ca)} traj Ca vs {len(binder_ca)} binder Ca)")
    return best


def make_dissolve(receptor: S.Structure, n_frames: int, out_path: str,
                  seed: int = 7) -> dict:
    """
    BIR3 disintegrates. Motif + ligand + Zn hold still; everything else flies
    outward on a radial + tangential path with an ease-in so it accelerates.
    """
    head("building dissolve trajectory")
    xyz0 = receptor.coords()
    fixed = np.array([
        (a.record == "HETATM") or (MOTIF_RANGE[0] <= a.resseq <= MOTIF_RANGE[1])
        for a in receptor.atoms])
    log(f"{int(fixed.sum())} atoms held (motif + het), "
        f"{int((~fixed).sum())} atoms dissolving")

    pocket = xyz0[fixed].mean(0) if fixed.any() else xyz0.mean(0)
    rng = np.random.default_rng(seed)

    radial = xyz0 - pocket
    norm = np.linalg.norm(radial, axis=1, keepdims=True)
    radial = radial / np.clip(norm, 1e-6, None)

    up = np.array([0.0, 0.0, 1.0])
    tangent = np.cross(radial, up)
    tn = np.linalg.norm(tangent, axis=1, keepdims=True)
    tangent = np.where(tn > 1e-6, tangent / np.clip(tn, 1e-6, None), 0.0)

    jitter = rng.normal(scale=0.35, size=xyz0.shape)
    direction = radial + 0.45 * tangent + jitter
    direction /= np.clip(np.linalg.norm(direction, axis=1, keepdims=True), 1e-6, None)

    rg = receptor.radius_of_gyration()
    speed = rng.uniform(0.8, 1.6, size=(len(xyz0), 1)) * rg * 2.6
    stagger = np.clip((norm / np.clip(norm.max(), 1e-6, None)) * 0.35, 0, 0.35)

    frames = []
    for i in range(n_frames):
        t = i / max(n_frames - 1, 1)
        local = np.clip((t - stagger) / np.clip(1.0 - stagger, 1e-6, None), 0, 1)
        amt = (local ** 2.4) * speed
        xyz = xyz0 + direction * amt
        xyz[fixed] = xyz0[fixed]
        frames.append(xyz)

    S.write_trajectory(frames, receptor, out_path, remarks=[
        "dissolve: cIAP1 BIR3 disintegrates around the retained motif",
        "frame 1 = intact crystal receptor",
    ])
    log(f"wrote {out_path} ({n_frames} frames)")
    return {"frames": n_frames, "held_atoms": int(fixed.sum()),
            "dissolving_atoms": int((~fixed).sum())}


def make_condense(binder: S.Structure, seed_struct: S.Structure, n_frames: int,
                  out_path: str, rfd_traj: str | None = None,
                  rng_seed: int = 11) -> dict:
    """
    Binder condenses out of noise. If a real RFdiffusion *_pX0_traj.pdb is
    supplied we play that back (reversed, because trajectories are written
    T -> 0). Otherwise we synthesise a variance-preserving schedule:

        x_t = c + a_t (x_0 - c) + s_t * eps

    with a cosine schedule, temporally smooth correlated noise, and a slight
    per-residue stagger so it settles rather than snapping.
    """
    head("building condensation trajectory")

    if rfd_traj and os.path.exists(rfd_traj):
        models = S.read_pdb_models(rfd_traj)
        log(f"real RFdiffusion trajectory: {len(models)} models from "
            f"{os.path.basename(rfd_traj)}")
        if len(models) >= 5:
            counts = {len(m) for m in models}
            if len(counts) == 1:
                frames = [m.coords() for m in models]

                # Which end is the noise? Do not assume. RFdiffusion's own
                # frame order has changed between releases and differs
                # between the pX0 and Xt trajectory files, and getting it
                # backwards plays the hero shot as a protein melting -- which
                # looks deliberate enough that nobody catches it in review.
                # Backbone regularity is the giveaway: a folded chain has
                # Ca-Ca ~3.8 A throughout, a noised one does not.
                first, last = chain_disorder(frames[0]), chain_disorder(frames[-1])
                log(f"chain disorder: first model {first:.2f} A, "
                    f"last model {last:.2f} A")
                if first < last:
                    frames = frames[::-1]
                    order = "reversed (file ran folded -> noise)"
                else:
                    order = "as written (file already ran noise -> folded)"
                log(f"playback order: {order}")

                # A diffusion trajectory lives in the diffusion frame, which
                # has nothing to do with the crystal frame everything else in
                # this shot is in. Superpose the folded end onto the placed
                # binder and carry that transform back through every frame,
                # or the binder condenses somewhere off-screen.
                fit = superpose_trajectory(frames, models[0], binder)
                if fit is None:
                    log("could not superpose the trajectory onto the binder "
                        "-- falling back to synthetic")
                else:
                    R, t, rms, how = fit
                    log(f"superposed onto the placed binder by {how}: "
                        f"Ca RMSD {rms:.2f} A")
                    if rms > 3.0:
                        log("WARNING: high trajectory-to-binder RMSD. The "
                            "trajectory and --design may be different designs.")
                    frames = [f @ R.T + t for f in frames]

                    n_src = len(frames)
                    frames = resample_frames(frames, n_frames)
                    jump = max(float(np.abs(frames[i + 1] - frames[i]).max())
                               for i in range(len(frames) - 1))
                    log(f"resampled {n_src} -> {n_frames} frames "
                        f"(linear interpolation)")
                    log(f"largest single-frame atom motion: {jump:.2f} A "
                        f"({'smooth' if jump < 3.0 else 'CHECK FOR FLICKER'})")
                    if jump >= 3.0:
                        log("     a real diffusion path is genuinely jumpy at "
                            "high noise; raise --condense-sec to spread it over "
                            "more frames")
                    S.write_trajectory(frames, models[0], out_path, remarks=[
                        "condense: real RFdiffusionAA reverse-diffusion trajectory",
                        f"source: {os.path.basename(rfd_traj)}, {order}",
                        f"superposed onto the placed binder ({how}, "
                        f"RMSD {rms:.2f} A)",
                        f"resampled {n_src} -> {n_frames} frames by interpolation",
                    ])
                    log(f"wrote {out_path} ({n_frames} frames, REAL trajectory)")
                    return {"frames": n_frames, "source": "rfdiffusion",
                            "file": os.path.basename(rfd_traj),
                            "source_frames": n_src,
                            "playback_order": order,
                            "superposition": how,
                            "superposition_rmsd": round(rms, 3),
                            "max_frame_jump_A": round(jump, 2),
                            "disorder_first": round(first, 3),
                            "disorder_last": round(last, 3)}
            log("models have inconsistent atom counts -- falling back to synthetic")
        else:
            log("too few models -- falling back to synthetic")

    log("synthesising a variance-preserving condensation")
    xyz0 = binder.coords()
    n = len(xyz0)
    centre = seed_struct.coords().mean(0) if len(seed_struct) else xyz0.mean(0)
    rg = binder.radius_of_gyration()
    # ~1.2 Rg gives a cloud whose extent matches the folded protein, which is
    # what high-t reverse diffusion actually looks like. Bigger reads as an
    # explosion rather than a fold.
    sigma_max = 1.45 * rg
    log(f"binder Rg {rg:.1f} A -> noise sigma {sigma_max:.1f} A")

    noise = smooth_noise_field(n, n_frames, rng_seed)

    # residues settle over a short window, N-term first
    rng = np.random.default_rng(rng_seed + 1)
    res_ids = {}
    order = []
    for a in binder.atoms:
        if a.reskey not in res_ids:
            res_ids[a.reskey] = len(res_ids)
            order.append(a.reskey)
    n_res = max(len(res_ids), 1)
    per_atom_res = np.array([res_ids[a.reskey] for a in binder.atoms])
    stagger = (per_atom_res / n_res) * 0.18
    stagger = stagger + rng.normal(scale=0.02, size=n)
    stagger = np.clip(stagger, 0.0, 0.30)[:, None]

    frames = []
    for i in range(n_frames):
        t = i / max(n_frames - 1, 1)
        local = np.clip((t - stagger) / np.clip(1.0 - stagger, 1e-6, None), 0, 1)
        a_t = np.sin(local * np.pi / 2) ** 2          # cosine schedule, 0 -> 1
        s_t = np.cos(local * np.pi / 2) ** 1.6        # noise amplitude, 1 -> 0
        xyz = centre + a_t * (xyz0 - centre) + s_t * sigma_max * noise[i]
        frames.append(xyz)
    frames[-1] = xyz0.copy()                          # land exactly on the design
    frames = temporal_smooth(frames, window=3)

    jump = max(float(np.abs(frames[i + 1] - frames[i]).max())
               for i in range(len(frames) - 1))
    log(f"largest single-frame atom motion: {jump:.2f} A "
        f"({'smooth' if jump < 3.0 else 'CHECK FOR FLICKER'})")

    S.write_trajectory(frames, binder, out_path, remarks=[
        "condense: synthetic variance-preserving reverse-diffusion schedule",
        "for the real thing rerun RFdiffusionAA with inference.write_trajectory=True",
    ])
    log(f"wrote {out_path} ({n_frames} frames, SYNTHETIC)")
    return {"frames": n_frames, "source": "synthetic",
            "sigma_max": round(float(sigma_max), 2),
            "max_frame_jump_A": round(jump, 2)}


def make_breathe(binder: S.Structure, seed_struct: S.Structure, n_frames: int,
                 out_path: str, amplitude: float = 1.3) -> dict:
    """
    Ca-ANM oscillation of the final complex. Cheap, physically motivated, and
    stops the hero frame looking like a dead crystal.
    """
    head("building ANM breathing trajectory")
    complex_st = S.Structure(
        [replace(a) for a in seed_struct.atoms] + [replace(a) for a in binder.atoms],
        name="complex")
    complex_st.renumber_serials()

    ca, ca_keys = complex_st.ca_trace()
    log(f"complex: {len(complex_st)} atoms, {len(ca)} Ca")
    if len(ca) < 8:
        log("too few Ca for ANM -- writing a static single frame")
        S.write_pdb(complex_st, out_path)
        return {"frames": 1, "modes": 0}

    vals, vecs = S.anm_modes(ca, cutoff=15.0, n_modes=3)
    log(f"ANM eigenvalues (modes 7-9): "
        f"{', '.join(f'{v:.3f}' for v in vals)}")

    mode = vecs[0] + 0.45 * vecs[1]
    mode /= np.sqrt((mode ** 2).sum(1).mean())        # unit per-atom RMS

    ca_index = {k: i for i, k in enumerate(ca_keys)}
    per_atom = np.zeros((len(complex_st), 3))
    for j, a in enumerate(complex_st.atoms):
        i = ca_index.get(a.reskey)
        if i is not None:
            per_atom[j] = mode[i]

    # het groups ride with their nearest Ca so the ligand doesn't detach
    het_idx = [j for j, a in enumerate(complex_st.atoms) if a.record == "HETATM"]
    if het_idx and len(ca):
        hx = complex_st.coords()[het_idx]
        d = np.sqrt(((hx[:, None, :] - ca[None, :, :]) ** 2).sum(-1))
        near = d.argmin(1)
        for k, j in enumerate(het_idx):
            per_atom[j] = mode[near[k]]

    xyz0 = complex_st.coords()
    frames = [xyz0 + amplitude * np.sin(2 * np.pi * i / n_frames) * per_atom
              for i in range(n_frames)]

    S.write_trajectory(frames, complex_st, out_path, remarks=[
        "breathe: Ca-ANM modes 7+8, one full loopable oscillation",
        f"peak displacement ~{amplitude:.1f} A RMS",
    ])
    log(f"wrote {out_path} ({n_frames} frames, loopable)")
    return {"frames": n_frames, "modes": 3, "amplitude_A": amplitude,
            "eigenvalues": [round(float(v), 4) for v in vals]}


# --------------------------------------------------------------------------
# step 5 -- metrics + ChimeraX selection names
# --------------------------------------------------------------------------

def measure(full: S.Structure | None, receptor: S.Structure,
            binder: S.Structure, seed_struct: S.Structure,
            warhead: str | None) -> dict:
    head("measuring")
    lig = [a for a in seed_struct.atoms
           if a.record == "HETATM" and a.resname == warhead] if warhead else []

    m = {
        "full_length_residues": full.n_protein_residues() if full else None,
        "bir3_residues": receptor.n_protein_residues(),
        "binder_residues": binder.n_protein_residues(),
        "motif_residues": seed_struct.n_protein_residues(),
        "binder_rg_A": round(binder.radius_of_gyration(), 2),
        "bir3_rg_A": round(receptor.radius_of_gyration(), 2),
        "warhead": warhead,
    }
    if full:
        m["shrink_factor"] = round(m["full_length_residues"] / max(m["binder_residues"], 1), 2)
        m["residues_removed"] = m["full_length_residues"] - m["binder_residues"]

    if lig:
        bc = S.contacts(binder.atoms, lig, cutoff=4.0)
        rc = S.contacts([a for a in receptor.atoms if a.record == "ATOM"], lig, 4.0)
        m["binder_ligand_contacts"] = len(bc)
        m["bir3_ligand_contacts"] = len(rc)
        m["binder_min_ligand_dist_A"] = round(S.min_distance(binder.atoms, lig), 2)
        m["bir3_contact_residues"] = [f"{r[2]}{r[1]}" for r in rc]
        m["binder_contact_residues"] = [f"{r[2]}{r[1]}" for r in bc]
        log(f"ligand {warhead}: BIR3 makes {len(rc)} contacts, "
            f"binder makes {len(bc)}")
        log(f"BIR3 contact residues: {', '.join(m['bir3_contact_residues'])}")

    core_present = [r for r in CORE_CONTACTS
                    if any(a.resseq == r for a in seed_struct.atoms
                           if a.record == "ATOM")]
    m["core_contacts_retained"] = core_present
    m["core_contacts_expected"] = CORE_CONTACTS
    log(f"core contacts retained in motif: {core_present} "
        f"of {CORE_CONTACTS}")

    for k in ("full_length_residues", "bir3_residues", "binder_residues"):
        log(f"{k}: {m[k]}")
    if "shrink_factor" in m:
        log(f"shrink factor: {m['shrink_factor']}x")
    return m


GEN_TEMPLATE = """\
# generated.cxc -- written by prep_shrink.py. Do not edit; rerun prep instead.
#
# Source this AFTER opening the models, in this order:
#   #1 01_ciap1_full.pdb      #2 02_bir3_ref.pdb     #3 03_binder_final.pdb
#   #4 04_motif_seed.pdb      #5 05_dissolve.pdb     #6 06_condense.pdb
#   #7 07_complex_breathe.pdb

# ---------------------------------------------------------------- palette
# Sampled from the DigiActivate animation swatches so the molecular shot
# cuts against the cartoon sequence without a colour jump.
colordef cE3       #2E6FA8
colordef cE3dark   #1B4C78
colordef cMotif    #E8663C
colordef cHotspot  #FF8C42
colordef cLigand   #14B8A6
colordef cBinder   #F2B441
colordef cBinderHi #FFD98A
colordef cGhost    #C8CCD2
colordef cZinc     #9AA7B4

# ------------------------------------------------------------ selections
name motifFull      #1/A:{m0}-{m1}
name bulkFull       #1/A & protein & ~#1/A:{m0}-{m1}
name motifRef       #2/A:{m0}-{m1}
name motifSeed      #4/A:{m0}-{m1}
name bulkRef        #2/A & protein & ~#2/A:{m0}-{m1}
name hotspotsRef    #2/A:{hs}
name hotspotsSeed   #4/A:{hs}
name ligandRef      {lig_ref}
name ligandSeed     {lig_seed}
name zincSeed       #4:ZN
name binder         #3/B
name dissolveBulk   #5/A & protein & ~#5/A:{m0}-{m1}
name dissolveKeep   #5/A:{m0}-{m1}
name dissolveHet    #5 & ~protein
name condenseBinder #6
name breatheAll     #7

# ------------------------------------------------- playback (frame counts
# baked in so the render scripts never drift out of sync with prep)
alias playDissolve  coordset #5 1,{n_dissolve}
alias playCondense  coordset #6 1,{n_condense}
alias playBreathe   coordset #7 1,{n_breathe}
alias fadeDissolve  perframe "transparency dissolveBulk $1 target acs" range 0,100 frames {n_dissolve}
alias fadeCondense  perframe "transparency #6 $1 target ab" range 70,0 frames {n_condense}

# ---------------------------------------------------- on-screen counters
alias labelBefore   2dlabels create counter text "{lbl_before}" xpos .06 ypos .09 size 30 color black bold true
alias labelAfter    2dlabels change counter text "{lbl_after}"
alias labelMotif    2dlabels create sub text "{lbl_motif}" xpos .06 ypos .04 size 20 color #555555
alias labelStats    2dlabels change sub text "{lbl_stats}"
alias labelsOff     2dlabels delete all
"""


OPEN_TEMPLATE = """\
# 00_open.cxc -- written by prep_shrink.py. Do not edit; rerun prep instead.
#
# generated.cxc hard-codes model numbers #1-#7, so the open order here is
# load-bearing. When there is no full-length AlphaFold model, #1 falls back to
# the BIR3 domain rather than being skipped -- skipping it would renumber
# every model below it and every selection in generated.cxc would silently
# point at the wrong thing.
#
# Trajectories MUST open with `coordset true`. Without it ChimeraX loads a
# multi-MODEL PDB as N sibling models instead of one model with N coordsets,
# and `coordset #5 1,{n_dissolve}` has nothing to play.

open {f_full}                       # 1 {full_note}
open {out}/02_bir3_ref.pdb          # 2  crystal receptor, chain A + Zn + warhead
open {out}/03_binder_final.pdb      # 3  de novo binder, chain B
open {out}/04_motif_seed.pdb        # 4  retained motif + warhead + Zn
open {out}/05_dissolve.pdb coordset true    # 5  {n_dissolve} frames
open {out}/06_condense.pdb coordset true    # 6  {n_condense} frames
open {out}/07_complex_breathe.pdb coordset true  # 7  {n_breathe} frames
"""


def write_open(path: str, out_dir: str, has_full: bool, traj: dict) -> None:
    full = (os.path.join(out_dir, "01_ciap1_full.pdb") if has_full
            else os.path.join(out_dir, "02_bir3_ref.pdb"))
    note = ("full-length cIAP1" if has_full
            else "NO full-length model -- BIR3 stands in (see README)")
    with open(path, "w") as fh:
        fh.write(OPEN_TEMPLATE.format(
            f_full=full, full_note=note, out=out_dir,
            n_dissolve=traj["dissolve"]["frames"],
            n_condense=traj["condense"]["frames"],
            n_breathe=traj["breathe"]["frames"]))
    log(f"wrote {path}")


def write_generated(path: str, warhead: str | None, traj: dict,
                    metrics: dict, fit: dict, framing: str = "partners") -> None:
    lig_ref = f"#2:{warhead}" if warhead else "#2 & ~protein & ~#2:ZN"
    lig_seed = f"#4:{warhead}" if warhead else "#4 & ~protein & ~#4:ZN"

    before_n = metrics.get("full_length_residues") or metrics["bir3_residues"]
    before_name = "cIAP1" if metrics.get("full_length_residues") else "cIAP1 BIR3"

    # Two framings for the counters. See README_payload.md -- the size claim
    # does not survive against an isolated BIR3 domain, so "partners" is the
    # default. The residue counts are true either way; what changes is which
    # claim the shot is making.
    if framing == "size":
        lbl_before = f"{before_name}   {before_n} aa"
        lbl_after = f"de novo binder   {metrics['binder_residues']} aa"
    else:
        lbl_before = (f"{before_name}   {before_n} aa"
                      f"   |   endogenous IAP / NF-kB regulator")
        lbl_after = (f"de novo binder   {metrics['binder_residues']} aa"
                     f"   |   no endogenous partners")
    stats = f"motif RMSD {fit['rmsd']:.2f} A"
    if metrics.get("binder_ligand_contacts") is not None:
        stats += f"   |   {metrics['binder_ligand_contacts']} ligand contacts"

    with open(path, "w") as fh:
        fh.write(GEN_TEMPLATE.format(
            m0=MOTIF_RANGE[0], m1=MOTIF_RANGE[1],
            hs=",".join(str(h) for h in sorted(HOTSPOTS)),
            lig_ref=lig_ref, lig_seed=lig_seed,
            n_dissolve=traj["dissolve"]["frames"],
            n_condense=traj["condense"]["frames"],
            n_breathe=traj["breathe"]["frames"],
            lbl_before=lbl_before,
            lbl_after=lbl_after,
            lbl_motif=f"retained motif {MOTIF_RANGE[0]}-{MOTIF_RANGE[1]}"
                      f"   |   {metrics['motif_residues']} aa",
            lbl_stats=stats,
        ))
    log(f"wrote {path}")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description="Prepare structures for the DigiActivate E3-shrink shot.")
    p.add_argument("--design", required=True,
                   help="AF3 design .cif/.pdb (e.g. ciap1_run1_chunk0_ciap1_binder_0_model_0.cif)")
    p.add_argument("--receptor", help="local 6W74 pdb (otherwise fetched)")
    p.add_argument("--fulllength", help="local AlphaFold cIAP1 pdb (otherwise fetched)")
    p.add_argument("--rfd-traj", dest="rfd_traj",
                   help="real RFdiffusionAA *_pX0_traj.pdb -- strongly preferred")
    p.add_argument("--input-dir", default="input")
    p.add_argument("--out-dir", default="output")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--dissolve-sec", type=float, default=1.5)
    p.add_argument("--condense-sec", type=float, default=3.0)
    p.add_argument("--breathe-sec", type=float, default=2.5)
    p.add_argument("--framing", default="partners", choices=["partners", "size"],
                   help="which claim the on-screen counters make; see "
                        "README_payload.md before choosing 'size'")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    o = lambda name: os.path.join(args.out_dir, name)

    recep_path, full_path = acquire(args)

    full = None
    if full_path:
        full = S.read_pdb(full_path)
        full = full.select(lambda a: not a.is_hydrogen())
        full.rename_chain(full.chains()[0], "A")
        full.renumber_serials()
        S.write_pdb(full, o("01_ciap1_full.pdb"), remarks=[
            f"full-length cIAP1 / BIRC2, AlphaFold DB {UNIPROT_CIAP1}",
            f"motif {MOTIF_RANGE[0]}-{MOTIF_RANGE[1]} keeps UniProt numbering",
        ])
        log(f"wrote {o('01_ciap1_full.pdb')} "
            f"({full.n_protein_residues()} residues)")

    receptor, rinfo = clean_receptor(recep_path, o("02_bir3_ref.pdb"))
    binder, seed_struct, fit = place_design(
        args.design, receptor, o("03_binder_final.pdb"), o("04_motif_seed.pdb"))

    nd = max(int(args.fps * args.dissolve_sec), 2)
    nc = max(int(args.fps * args.condense_sec), 2)
    nb = max(int(args.fps * args.breathe_sec), 2)

    traj = {
        "dissolve": make_dissolve(receptor, nd, o("05_dissolve.pdb")),
        "condense": make_condense(binder, seed_struct, nc, o("06_condense.pdb"),
                                  rfd_traj=args.rfd_traj),
        "breathe": make_breathe(binder, seed_struct, nb, o("07_complex_breathe.pdb")),
    }

    metrics = measure(full, receptor, binder, seed_struct, rinfo["warhead"])
    metrics.update({
        "receptor_info": rinfo,
        "design_placement": fit,
        "trajectories": traj,
        "fps": args.fps,
        "motif_range": list(MOTIF_RANGE),
        "hotspots": HOTSPOTS,
        "design_file": os.path.basename(args.design),
    })

    head("writing metadata")
    with open(o("metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    log(f"wrote {o('metrics.json')}")
    write_generated(o("generated.cxc"), rinfo["warhead"], traj, metrics, fit,
                    framing=args.framing)
    write_open(o("00_open.cxc"), args.out_dir, full is not None, traj)

    head("on-screen counters")
    if metrics.get("full_length_residues"):
        print(f"    cIAP1 (full length)      {metrics['full_length_residues']} aa")
    print(f"    cIAP1 BIR3 (crystal)     {metrics['bir3_residues']} aa")
    print(f"    de novo binder           {metrics['binder_residues']} aa")
    print(f"    retained motif           {metrics['motif_residues']} aa")
    if metrics.get("shrink_factor"):
        print(f"    shrink factor            {metrics['shrink_factor']}x")
    print(f"    motif Ca RMSD            {fit['rmsd']:.2f} A")
    if traj["condense"]["source"] == "synthetic":
        print("\n    NOTE: condensation is synthetic. Rerun RFdiffusionAA with")
        print("          inference.write_trajectory=True and pass --rfd-traj")
        print("          to use your own reverse-diffusion path instead.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
