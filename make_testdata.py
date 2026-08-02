#!/usr/bin/env python3
"""
make_testdata.py -- synthesise the inputs prep_shrink.py expects, so the whole
pipeline can be exercised without the real design files.

Produces:
  input/synthetic_design.cif   a de novo-style mini-binder with the 312-325
                               motif grafted in, in AF3-ish mmCIF
  input/synthetic_rfd_traj.pdb a multi-MODEL reverse-diffusion-style trajectory
                               in RFdiffusion's T->0 order

WHAT THIS IS NOT
----------------
This is not a designed binder. Nothing was folded, scored, or validated -- it
is a shape with the right size, the right secondary-structure content and the
real motif in the real place, built so the animation code has something to
chew on. Every frame it appears in should be treated as a placeholder until
the actual AF3 model replaces it:

    python3 prep_shrink.py --design your_real_design.cif

The one thing it does honestly: the retained motif A312-325 carries its true
6W74 coordinates, so the superposition step in prep_shrink.py is solving a
real problem and `design_placement.rmsd` in metrics.json means something.

HOW THE SHAPE IS BUILT
----------------------
Not by hand-placing helices -- that always looks like hand-placed helices.
Instead a coarse-grained Ca chain is relaxed under a small energy function:
ideal Ca-Ca bonds, alpha-helical i/i+2/i+3/i+4 distance restraints on the
helical segments, excluded volume against itself and against the PROTAC
warhead, a weak compaction term, and hard positional pins on the 14 grafted
motif residues. Gradient descent from a self-avoiding random walk. The result
is compact, clash-free, and its helices close on themselves the way a real
bundle does.

Backbone N/C/O/CB are then rebuilt from the Ca trace using local-frame
coefficients fitted to 6W74 itself, so the geometry that comes out is
calibrated against a real crystal structure rather than remembered constants.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

import structlib as S

MOTIF_RANGE = (312, 325)

# Ideal alpha-helix Ca-Ca distances (Angstrom) for |i-j| = 1..4.
HELIX_D = {1: 3.81, 2: 5.43, 3: 5.05, 4: 6.20}
BOND_CA = 3.81

# Deliberately boring de novo alphabet: helices are the usual Ala/Leu/Glu/Lys
# designable set, loops get the turn-formers. Designed sequences really do
# look like this.
HELIX_AA = ["ALA", "LEU", "GLU", "LYS", "ALA", "ARG", "LEU", "GLU",
            "ALA", "ILE", "LYS", "ALA", "LEU", "GLN", "GLU", "ALA"]
LOOP_AA = ["GLY", "SER", "PRO", "ASN", "ASP", "GLY", "THR", "SER"]


def log(msg: str) -> None:
    print(f"  {msg}", flush=True)


def head(msg: str) -> None:
    print(f"\n[{msg}]", flush=True)


# ---------------------------------------------------------------------------
# backbone reconstruction, calibrated against a real structure
# ---------------------------------------------------------------------------

def _local_frame(prev: np.ndarray, cur: np.ndarray, nxt: np.ndarray):
    """Orthonormal frame from three consecutive Ca. Returns (origin, 3x3)."""
    e1 = nxt - prev
    n1 = np.linalg.norm(e1)
    if n1 < 1e-6:
        return None
    e1 = e1 / n1
    b = cur - 0.5 * (prev + nxt)
    b = b - np.dot(b, e1) * e1
    n2 = np.linalg.norm(b)
    if n2 < 1e-6:
        return None
    e2 = b / n2
    e3 = np.cross(e1, e2)
    return cur, np.stack([e1, e2, e3])


def fit_backbone_coeffs(path: str) -> dict:
    """
    Learn where N, C, O and CB sit relative to the local Ca frame, from a real
    structure. Binned by the Ca(i-1)-Ca(i+1) distance, which is what actually
    distinguishes helix (~5.4 A) from extended (~6.7 A) geometry.
    """
    st = S.read_pdb(path)
    by_res: dict[tuple, dict] = {}
    order: list[tuple] = []
    for a in st.atoms:
        if a.record != "ATOM" or a.resname not in S.AMINO3:
            continue
        k = a.reskey
        if k not in by_res:
            by_res[k] = {}
            order.append(k)
        by_res[k][a.name.strip()] = a.xyz

    bins = {"helix": [], "extended": []}
    for i in range(1, len(order) - 1):
        kp, kc, kn = order[i - 1], order[i], order[i + 1]
        # only consecutive residues form a valid frame
        if not (kp[1] + 1 == kc[1] and kc[1] + 1 == kn[1] and kp[0] == kc[0] == kn[0]):
            continue
        try:
            prev, cur, nxt = by_res[kp]["CA"], by_res[kc]["CA"], by_res[kn]["CA"]
        except KeyError:
            continue
        fr = _local_frame(prev, cur, nxt)
        if fr is None:
            continue
        origin, M = fr
        d13 = float(np.linalg.norm(nxt - prev))
        rec = {}
        for nm in ("N", "C", "O", "CB"):
            if nm in by_res[kc]:
                rec[nm] = M @ (by_res[kc][nm] - origin)
        if not rec:
            continue
        rec["_d13"] = d13
        bins["helix" if d13 < 6.0 else "extended"].append(rec)

    coeffs = {}
    for label, recs in bins.items():
        if not recs:
            continue
        c = {}
        for nm in ("N", "C", "O", "CB"):
            vs = [r[nm] for r in recs if nm in r]
            if vs:
                c[nm] = np.mean(vs, axis=0)
        coeffs[label] = c
    log(f"backbone coefficients fitted from {os.path.basename(path)}: "
        f"{len(bins['helix'])} helical, {len(bins['extended'])} extended frames")
    return coeffs


def rebuild_backbone(ca: np.ndarray, resnames: list[str],
                     coeffs: dict) -> list[tuple[str, np.ndarray]]:
    """Ca trace -> list of (atom_name, xyz) in N, CA, C, O, CB order."""
    out: list[tuple[str, np.ndarray]] = []
    n = len(ca)
    for i in range(n):
        prev = ca[i - 1] if i > 0 else ca[i] - (ca[i + 1] - ca[i])
        nxt = ca[i + 1] if i < n - 1 else ca[i] + (ca[i] - ca[i - 1])
        fr = _local_frame(prev, ca[i], nxt)
        if fr is None:
            out.append(("CA", ca[i]))
            continue
        origin, M = fr
        d13 = float(np.linalg.norm(nxt - prev))
        c = coeffs["helix" if d13 < 6.0 else "extended"]
        rec = []
        for nm in ("N", "CA", "C", "O", "CB"):
            if nm == "CA":
                rec.append(("CA", ca[i]))
                continue
            if nm == "CB" and resnames[i] == "GLY":
                continue
            if nm not in c:
                continue
            rec.append((nm, origin + M.T @ c[nm]))
        out.append(rec)
    flat: list[tuple[str, np.ndarray]] = []
    for r in out:
        if isinstance(r, tuple):
            flat.append([r])
        else:
            flat.append(r)
    return flat


# ---------------------------------------------------------------------------
# the coarse-grained relaxation
# ---------------------------------------------------------------------------

def build_layout(n_total: int, n_motif: int) -> tuple[list[tuple], int]:
    """
    Segment plan for the chain. Motif sits in the middle, flanked by two
    helix-loop-helix units -- the usual shape RFdiffusion returns when you
    hand it an interior contig.

    Returns ([(kind, length), ...], motif_start_index).
    """
    body = n_total - n_motif
    half = body // 2
    # helix, loop, helix per half; loops absorb the remainder
    h = (half - 9) // 2
    l1 = 5
    l2 = half - 2 * h - l1
    left = [("H", h), ("L", l1), ("H", h), ("L", l2)]
    right_len = body - half
    h2 = (right_len - 9) // 2
    r1 = 4
    r2 = right_len - 2 * h2 - r1
    right = [("L", r1), ("H", h2), ("L", r2), ("H", h2)]
    segs = left + [("M", n_motif)] + right
    total = sum(s[1] for s in segs)
    if total != n_total:                       # push any slop into the last loop
        for i in range(len(segs) - 1, -1, -1):
            if segs[i][0] == "L":
                segs[i] = ("L", segs[i][1] + (n_total - total))
                break
    start = sum(s[1] for s in segs[:len(left)])
    return segs, start


def ideal_helix(n: int) -> np.ndarray:
    """n Ca on an ideal alpha-helix, axis along +z, first residue at angle 0.

    r=2.3 A, rise=1.5 A/residue, 100 deg/residue -- the standard parameters,
    and the ones that reproduce HELIX_D exactly."""
    k = np.arange(n)
    ang = np.deg2rad(100.0) * k
    return np.stack([2.3 * np.cos(ang), 2.3 * np.sin(ang), 1.5 * k], axis=1)


def _align_to(v: np.ndarray) -> np.ndarray:
    """Rotation taking +z onto the unit vector v."""
    v = v / np.linalg.norm(v)
    z = np.array([0.0, 0.0, 1.0])
    c = float(np.dot(z, v))
    if c > 1 - 1e-9:
        return np.eye(3)
    if c < -1 + 1e-9:
        return np.diag([1.0, -1.0, -1.0])
    ax = np.cross(z, v)
    s = np.linalg.norm(ax)
    ax = ax / s
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + s * K + (1 - c) * (K @ K)


def init_walk(n: int, centre: np.ndarray, radius: float,
              pins: dict[int, np.ndarray], rng,
              segs: list[tuple] | None = None) -> np.ndarray:
    """
    Starting geometry for the relaxation.

    Helical segments are laid down as ideal coils rather than left for gradient
    descent to discover -- descent from a random walk reliably gets stuck with
    two or three helices half-formed, because straightening a tangled segment
    means climbing over the excluded-volume term. Loops are random walks; those
    the relaxation handles fine.

    The chain is grown outward in both directions from the pinned motif, since
    that is the one block whose coordinates are not ours to choose.
    """
    pts = np.zeros((n, 3))
    idx = sorted(pins)
    for i in idx:
        pts[i] = pins[i]
    lo, hi = idx[0], idx[-1]

    kinds = ["L"] * n
    if segs:
        pos = 0
        for kind, length in segs:
            for i in range(pos, pos + length):
                kinds[i] = kind
            pos += length

    def inward(frm: np.ndarray) -> np.ndarray:
        """A unit vector that leans back towards the binder centre, so
        segments wrap the core instead of radiating off it."""
        d = centre - frm
        nd = np.linalg.norm(d)
        bias = d / nd if nd > 1e-6 else np.zeros(3)
        w = np.clip(nd / max(radius, 1e-6), 0.0, 1.5)
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        out = v + (0.9 * w) * bias
        return out / np.linalg.norm(out)

    def run_segments(start: int, stop: int, step: int) -> None:
        """Fill [start, stop) walking by `step` (+1 forward, -1 backward)."""
        i = start
        while (i < stop) if step > 0 else (i > stop):
            kind = kinds[i]
            # how many consecutive residues of this kind lie ahead
            j = i
            while ((j + step < stop) if step > 0 else (j + step > stop)) \
                    and kinds[j + step] == kind:
                j += step
            length = abs(j - i) + 1
            anchor = pts[i - step]
            if kind == "H" and length >= 6:
                coil = ideal_helix(length)
                R = _align_to(inward(anchor))
                coil = coil @ R.T
                if step < 0:                      # growing backwards: reverse it
                    coil = coil[::-1]
                coil = coil - coil[0] + anchor + BOND_CA * inward(anchor)
                for k in range(length):
                    pts[i + step * k] = coil[k]
            else:
                prev = anchor
                for k in range(length):
                    prev = prev + BOND_CA * inward(prev)
                    pts[i + step * k] = prev
            i = j + step

    run_segments(hi + 1, n, +1)
    run_segments(lo - 1, -1, -1)
    return pts


def relax(pts: np.ndarray, segs: list[tuple], motif_start: int,
          pins: dict[int, np.ndarray], lig: np.ndarray,
          centre: np.ndarray, n_steps: int = 4000,
          seed: int = 3) -> np.ndarray:
    """
    Gradient descent on a small coarse-grained energy.

    Terms, in rough order of stiffness:
      pin      motif Ca nailed to their crystal positions
      bond     Ca-Ca = 3.81 A
      helix    i/i+2, i/i+3, i/i+4 restraints inside helical segments
      clash    soft excluded volume, chain-chain and chain-ligand
      compact  weak pull to the binder centre so it folds rather than sprawls
    """
    n = len(pts)
    x = pts.copy()

    helix_pairs = []
    pos = 0
    for kind, length in segs:
        if kind == "H":
            for i in range(pos, pos + length):
                for off, d0 in HELIX_D.items():
                    j = i + off
                    if j < pos + length:
                        helix_pairs.append((i, j, d0))
        pos += length
    hp_i = np.array([p[0] for p in helix_pairs], dtype=int)
    hp_j = np.array([p[1] for p in helix_pairs], dtype=int)
    hp_d = np.array([p[2] for p in helix_pairs], dtype=float)

    pin_idx = np.array(sorted(pins), dtype=int)
    pin_xyz = np.array([pins[i] for i in sorted(pins)], dtype=float)

    seq = np.arange(n)
    k_bond, k_helix, k_clash, k_comp, k_lig = 20.0, 2.5, 14.0, 0.25, 10.0
    r_clash, r_lig = 4.3, 4.4
    # Flat-bottomed compaction. A plain harmonic pull to the centre squeezes
    # every bond a few percent short, because nothing balances it; only acting
    # outside the envelope leaves the interior free to keep ideal geometry.
    r_envelope = 2.05 * (len(x) ** (1.0 / 3.0)) + 6.0
    lr = 0.010

    for step in range(n_steps):
        g = np.zeros_like(x)

        # bonds. E = k/2 (L-L0)^2 with d = x[i+1]-x[i], so the gradient is
        # +f on the far atom and -f on the near one. Getting this backwards
        # turns the restraint into a repulsion and the whole chain explodes.
        d = x[1:] - x[:-1]
        L = np.linalg.norm(d, axis=1, keepdims=True)
        u = d / np.clip(L, 1e-9, None)
        f = k_bond * (L - BOND_CA) * u
        g[1:] += f
        g[:-1] -= f

        # helical shape
        if len(hp_i):
            d2 = x[hp_j] - x[hp_i]
            L2 = np.linalg.norm(d2, axis=1, keepdims=True)
            u2 = d2 / np.clip(L2, 1e-9, None)
            f2 = k_helix * (L2 - hp_d[:, None]) * u2
            np.add.at(g, hp_j, f2)
            np.add.at(g, hp_i, -f2)

        # self excluded volume, ignoring i..i+2
        diff = x[:, None, :] - x[None, :, :]
        dist = np.sqrt((diff ** 2).sum(-1)) + np.eye(n) * 1e9
        mask = (np.abs(seq[:, None] - seq[None, :]) > 2) & (dist < r_clash)
        if mask.any():
            ii, jj = np.nonzero(mask)
            dv = diff[ii, jj]
            dd = dist[ii, jj][:, None]
            f3 = k_clash * (dd - r_clash) * dv / np.clip(dd, 1e-9, None)
            np.add.at(g, ii, f3)

        # keep out of the warhead
        if len(lig):
            dl = x[:, None, :] - lig[None, :, :]
            distl = np.sqrt((dl ** 2).sum(-1))
            m2 = distl < r_lig
            if m2.any():
                ii, jj = np.nonzero(m2)
                dv = dl[ii, jj]
                dd = distl[ii, jj][:, None]
                f4 = k_lig * (dd - r_lig) * dv / np.clip(dd, 1e-9, None)
                np.add.at(g, ii, f4)

        # compaction, but only on atoms outside the envelope
        rvec = x - centre
        rlen = np.linalg.norm(rvec, axis=1, keepdims=True)
        out = rlen > r_envelope
        if out.any():
            g += np.where(out, k_comp * (rlen - r_envelope)
                          * rvec / np.clip(rlen, 1e-9, None), 0.0)

        # The random-walk start can drop two Ca almost on top of each other,
        # and the excluded-volume gradient there is enormous. Clip per-atom
        # step length rather than lowering lr for the other 4000 steps.
        gn = np.linalg.norm(g, axis=1, keepdims=True)
        g = np.where(gn > 50.0, g * (50.0 / np.clip(gn, 1e-9, None)), g)

        x -= lr * g
        x[pin_idx] = pin_xyz                    # hard pin, re-applied every step

        if step == n_steps // 2:
            lr *= 0.5

    return x


def energy_report(x: np.ndarray, segs, lig: np.ndarray) -> dict:
    d = np.linalg.norm(x[1:] - x[:-1], axis=1)
    n = len(x)
    seq = np.arange(n)
    diff = x[:, None, :] - x[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1)) + np.eye(n) * 1e9
    far = np.abs(seq[:, None] - seq[None, :]) > 2
    worst = float(dist[far].min())
    ligmin = float("inf")
    if len(lig):
        ligmin = float(np.sqrt((((x[:, None, :] - lig[None, :, :]) ** 2).sum(-1))).min())
    return {
        "bond_mean": round(float(d.mean()), 3),
        "bond_min": round(float(d.min()), 3),
        "bond_max": round(float(d.max()), 3),
        "worst_nonlocal_contact": round(worst, 2),
        "min_ligand_dist": round(ligmin, 2),
        "rg": round(float(np.sqrt(((x - x.mean(0)) ** 2).sum(1).mean())), 2),
    }


# ---------------------------------------------------------------------------
# mmCIF writing
# ---------------------------------------------------------------------------

CIF_HEADER = """\
data_synthetic_ciap1_binder
#
_entry.id   synthetic_ciap1_binder
#
_struct.title  'SYNTHETIC PLACEHOLDER - not a designed binder, see make_testdata.py'
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.auth_seq_id
_atom_site.auth_asym_id
_atom_site.auth_comp_id
_atom_site.auth_atom_id
_atom_site.pdbx_PDB_model_num
"""


def write_cif(atoms: list[S.Atom], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as fh:
        fh.write(CIF_HEADER)
        for i, a in enumerate(atoms, start=1):
            fh.write(
                f"{a.record} {i} {a.element} {a.name} . {a.resname} {a.chain} 1 "
                f"{a.resseq} ? {a.x:.3f} {a.y:.3f} {a.z:.3f} "
                f"{a.occupancy:.2f} {a.bfactor:.2f} {a.resseq} {a.chain} "
                f"{a.resname} {a.name} 1\n")
        fh.write("#\n")


def random_rigid(rng, scale: float = 40.0):
    """A random rotation+translation, so prep_shrink's superposition has to
    actually recover something rather than being handed the answer."""
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, xq, yq, zq = q
    R = np.array([
        [1 - 2 * (yq * yq + zq * zq), 2 * (xq * yq - zq * w), 2 * (xq * zq + yq * w)],
        [2 * (xq * yq + zq * w), 1 - 2 * (xq * xq + zq * zq), 2 * (yq * zq - xq * w)],
        [2 * (xq * zq - yq * w), 2 * (yq * zq + xq * w), 1 - 2 * (xq * xq + yq * yq)],
    ])
    t = rng.uniform(-scale, scale, size=3)
    return R, t


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--receptor", default="input/6w74.pdb")
    p.add_argument("--out-dir", default="input")
    p.add_argument("--length", type=int, default=132,
                   help="binder length in residues (README quotes 132 aa)")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--traj-frames", type=int, default=48)
    p.add_argument("--no-transform", action="store_true",
                   help="skip the random rigid transform (debugging only)")
    args = p.parse_args()

    if not os.path.exists(args.receptor):
        sys.exit(f"receptor not found: {args.receptor}\n"
                 f"Download https://files.rcsb.org/download/6W74.pdb and pass --receptor.")

    rng = np.random.default_rng(args.seed)

    head("reading receptor")
    rec = S.read_pdb(args.receptor)
    log(f"{rec} -- {rec.n_protein_residues()} protein residues")

    motif_atoms = [a for a in rec.atoms
                   if a.record == "ATOM" and MOTIF_RANGE[0] <= a.resseq <= MOTIF_RANGE[1]]
    motif_ca = [a for a in motif_atoms if a.name.strip() == "CA"]
    motif_ca.sort(key=lambda a: a.resseq)
    if len(motif_ca) != MOTIF_RANGE[1] - MOTIF_RANGE[0] + 1:
        sys.exit(f"receptor is missing motif residues: found {len(motif_ca)}")
    motif_xyz = np.array([[a.x, a.y, a.z] for a in motif_ca])
    motif_names = [a.resname for a in motif_ca]
    log(f"motif {MOTIF_RANGE[0]}-{MOTIF_RANGE[1]}: {'-'.join(motif_names)}")

    lig = np.array([[a.x, a.y, a.z] for a in rec.atoms
                    if a.record == "HETATM" and a.resname not in S.JUNK
                    and a.resname != "ZN"])
    log(f"warhead atoms: {len(lig)}")

    coeffs = fit_backbone_coeffs(args.receptor)

    head("laying out the chain")
    n_motif = len(motif_ca)
    segs, motif_start = build_layout(args.length, n_motif)
    log("segments: " + " ".join(f"{k}{n}" for k, n in segs))
    log(f"total {sum(n for _, n in segs)} residues, motif at index "
        f"{motif_start}-{motif_start + n_motif - 1}")

    lig_c = lig.mean(0) if len(lig) else motif_xyz.mean(0)
    d_out = motif_xyz.mean(0) - lig_c
    d_out /= np.linalg.norm(d_out)
    # Binder body sits behind the motif relative to the drug, so the warhead
    # ends up in a surface groove rather than buried in the core.
    centre = lig_c + 10.5 * d_out
    log(f"binder centre {centre.round(2)}, {np.linalg.norm(centre - lig_c):.1f} A "
        f"from the warhead")

    pins = {motif_start + i: motif_xyz[i] for i in range(n_motif)}
    x0 = init_walk(args.length, centre, 15.0, pins, rng, segs)

    head("relaxing")
    log(f"{args.steps} steps of gradient descent on {args.length} Ca")
    x = relax(x0, segs, motif_start, pins, lig, centre,
              n_steps=args.steps, seed=args.seed)
    rep = energy_report(x, segs, lig)
    for k, v in rep.items():
        log(f"{k}: {v}")
    if rep["bond_max"] > 4.2 or rep["bond_min"] < 3.4:
        log("WARNING: bond lengths drifted; raise --steps")
    if rep["worst_nonlocal_contact"] < 3.6:
        log("WARNING: residual clash; raise --steps")

    # motif must not have moved
    drift = float(np.abs(x[motif_start:motif_start + n_motif] - motif_xyz).max())
    log(f"motif pin drift: {drift:.4f} A")

    head("rebuilding backbone")
    resnames = []
    pos = 0
    for kind, length in segs:
        for i in range(length):
            if kind == "M":
                resnames.append(motif_names[i])
            elif kind == "H":
                resnames.append(HELIX_AA[(pos + i) % len(HELIX_AA)])
            else:
                resnames.append(LOOP_AA[(pos + i) % len(LOOP_AA)])
        pos += length

    built = rebuild_backbone(x, resnames, coeffs)
    n_atoms = sum(len(r) for r in built)
    log(f"{n_atoms} atoms from {len(x)} Ca")

    # pLDDT-ish B-factors: confident helices, floppier loops
    plddt = np.zeros(args.length)
    pos = 0
    for kind, length in segs:
        base = {"H": 92.0, "M": 95.0, "L": 71.0}[kind]
        for i in range(length):
            plddt[pos + i] = base + rng.normal(scale=2.5)
        pos += length
    plddt = np.clip(plddt, 30, 98)

    atoms: list[S.Atom] = []
    for i, rec_atoms in enumerate(built):
        for nm, xyz in rec_atoms:
            atoms.append(S.Atom(
                record="ATOM", serial=len(atoms) + 1, name=nm,
                resname=resnames[i], chain="B", resseq=i + 1,
                x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]),
                occupancy=1.0, bfactor=round(float(plddt[i]), 2),
                element=S.guess_element(nm, resnames[i]),
            ))

    design = S.Structure(atoms, "synthetic_design")

    if not args.no_transform:
        R, t = random_rigid(rng)
        S.apply_transform(design, R, t)
        log("applied a random rigid transform -- prep_shrink.py has to "
            "recover the placement by superposition")

    head("writing")
    cif_path = os.path.join(args.out_dir, "synthetic_design.cif")
    write_cif(design.atoms, cif_path)
    log(f"wrote {cif_path} ({len(design)} atoms, {design.n_protein_residues()} residues)")

    # round-trip check: the cif reader has to give back what we wrote
    back = S.read_cif(cif_path)
    if len(back) != len(design):
        log(f"WARNING: cif round-trip lost atoms ({len(back)} vs {len(design)})")
    else:
        err = float(np.abs(back.coords() - design.coords()).max())
        log(f"cif round-trip max coordinate error: {err:.4f} A")

    # ------------------------------------------------------------------ traj
    head("writing a synthetic RFdiffusion-style trajectory")
    ca_only = S.Structure([a for a in design.atoms if a.name == "CA"], "traj")
    xyz_final = ca_only.coords()
    c = xyz_final.mean(0)
    rg = ca_only.radius_of_gyration()
    frames = []
    rs = np.random.default_rng(args.seed + 5)
    basis = rs.normal(size=(4, len(xyz_final), 3))
    k = np.ones(5) / 5
    for b in range(len(basis)):
        for dm in range(3):
            basis[b, :, dm] = np.convolve(basis[b, :, dm], k, mode="same")
    basis /= basis.std()
    for i in range(args.traj_frames):
        t = i / max(args.traj_frames - 1, 1)          # t=0 is the folded end
        a_t = math.cos(t * math.pi / 2) ** 2
        s_t = math.sin(t * math.pi / 2) ** 1.5
        ph = 2 * math.pi * t
        field = (basis[0] * math.cos(ph) + basis[1] * math.sin(ph)
                 + basis[2] * math.cos(0.5 * ph) + basis[3] * math.sin(0.5 * ph)) / 2.0
        frames.append(c + a_t * (xyz_final - c) + s_t * 1.4 * rg * field)
    # RFdiffusion writes T -> 0, i.e. noise first. Match that, because
    # prep_shrink.py reverses whatever it is handed.
    frames = frames[::-1]
    traj_path = os.path.join(args.out_dir, "synthetic_rfd_traj.pdb")
    S.write_trajectory(frames, ca_only, traj_path, remarks=[
        "SYNTHETIC stand-in for a RFdiffusionAA *_pX0_traj.pdb",
        "written T -> 0 (noise first) to match RFdiffusion's convention",
    ])
    log(f"wrote {traj_path} ({len(frames)} models, {len(ca_only)} Ca each)")

    head("done")
    print(f"    python3 prep_shrink.py --design {cif_path} \\")
    print(f"        --receptor {args.receptor} --rfd-traj {traj_path}\n")
    print("    Remember: this binder is a placeholder. Swap in the real AF3")
    print("    model before anything gets rendered for the actual video.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
