#!/usr/bin/env python3
"""
structlib.py -- the structural plumbing under the E3-shrink shot.

Deliberately dependency-light: numpy and the standard library, nothing else.
The pipeline has to run on a cluster login node, inside ChimeraX's bundled
Python, and on a laptop with no conda, so Biopython and ProDy are out.

Contents
--------
  Atom / Structure   minimal mutable structure model, chain+residue aware
  read_pdb           PDB reader (first model)
  read_pdb_models    multi-MODEL PDB reader, for RFdiffusion trajectories
  read_cif           mmCIF reader, for AF3 / Boltz / Chai design output
  read_structure     dispatch on extension
  write_pdb          PDB writer
  write_trajectory   multi-MODEL PDB writer, loadable by `open ... coordset true`
  kabsch             optimal superposition, returns (R, t, rmsd)
  contacts           residues within a cutoff of a ligand
  anm_modes          Ca anisotropic network model, the cheap way to get
                     physically-motivated breathing motion
  fetch_rcsb         network helpers; both return None rather than raising, so
  fetch_alphafold    an offline run degrades instead of dying
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from typing import Callable, Iterable, Sequence

import numpy as np

WATER = {"HOH", "WAT", "DOD", "H2O", "TIP", "TIP3", "SOL"}

# Crystallisation additives. Skipped when hunting for the PROTAC warhead so a
# well-ordered glycerol never gets promoted to "the ligand". Metals are
# deliberately NOT in here -- the structural zinc is part of the fold.
JUNK = {
    "SO4", "PO4", "GOL", "EDO", "PEG", "PG4", "PGE", "1PE", "MPD", "ACT",
    "ACY", "FMT", "DMS", "TRS", "EPE", "MES", "IMD", "BME", "CIT", "TAR",
    "NO3", "AZI", "IOD", "BR", "CL", "FLC", "SCN", "P6G", "2PE", "OLC",
} | WATER

AMINO3 = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "MSE", "SEC", "PYL", "HSD", "HSE", "HSP", "CSO", "PTR", "SEP", "TPO",
}

_ELEMENT_FROM_NAME = {
    "C": "C", "N": "N", "O": "O", "S": "S", "P": "P", "H": "H", "D": "H",
}


# ---------------------------------------------------------------------------
# atoms and structures
# ---------------------------------------------------------------------------

@dataclass
class Atom:
    record: str = "ATOM"        # ATOM | HETATM
    serial: int = 1
    name: str = "CA"
    altloc: str = ""
    resname: str = "GLY"
    chain: str = "A"
    resseq: int = 1
    icode: str = ""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    occupancy: float = 1.0
    bfactor: float = 0.0
    element: str = "C"

    @property
    def reskey(self) -> tuple:
        """(chain, resseq, icode) -- unique per residue across a Structure."""
        return (self.chain, self.resseq, self.icode)

    @property
    def restuple(self) -> tuple:
        return (self.chain, self.resseq, self.icode, self.resname)

    @property
    def xyz(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)

    def is_hydrogen(self) -> bool:
        if self.element:
            return self.element.strip().upper() in ("H", "D")
        # No element column (some minimised / hand-edited files). PDB atom
        # names put the element in cols 13-14, so a leading digit means the
        # name is shifted and the element is the next character.
        nm = self.name.strip()
        return bool(nm) and (nm[0] in "HD" or (nm[0].isdigit() and len(nm) > 1 and nm[1] == "H"))

    def is_protein(self) -> bool:
        return self.record == "ATOM" and self.resname in AMINO3


class Structure:
    """An ordered bag of atoms. Order is file order and is never shuffled."""

    def __init__(self, atoms: Iterable[Atom], name: str = "structure"):
        self.atoms: list[Atom] = list(atoms)
        self.name = name

    # -- basics ------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.atoms)

    def __repr__(self) -> str:
        return f"<Structure {self.name}: {len(self.atoms)} atoms, chains {''.join(self.chains())}>"

    def copy(self) -> "Structure":
        return Structure([replace(a) for a in self.atoms], self.name)

    def select(self, pred: Callable[[Atom], bool]) -> "Structure":
        """Copies. Callers mutate the result (chain renames, renumbering) and
        must not corrupt the parent -- `ref` is reused after `seed` is cut."""
        return Structure([replace(a) for a in self.atoms if pred(a)], self.name)

    # -- inventory ---------------------------------------------------------
    def chains(self) -> list[str]:
        seen, out = set(), []
        for a in self.atoms:
            if a.chain not in seen:
                seen.add(a.chain)
                out.append(a.chain)
        return out

    def residues(self) -> list[tuple]:
        seen, out = set(), []
        for a in self.atoms:
            k = a.restuple
            if k not in seen:
                seen.add(k)
                out.append(k)
        return out

    def het_residues(self, skip_junk: bool = True) -> list[tuple]:
        out = []
        for (c, seq, ic, name) in self.residues():
            if name in AMINO3:
                continue
            if skip_junk and name in JUNK:
                continue
            out.append((c, seq, ic, name))
        return out

    def n_protein_residues(self) -> int:
        return len({a.reskey for a in self.atoms if a.is_protein()})

    # -- geometry ----------------------------------------------------------
    def coords(self) -> np.ndarray:
        if not self.atoms:
            return np.zeros((0, 3))
        return np.array([[a.x, a.y, a.z] for a in self.atoms], dtype=float)

    def set_coords(self, xyz: np.ndarray) -> None:
        if len(xyz) != len(self.atoms):
            raise ValueError(f"coord/atom mismatch: {len(xyz)} vs {len(self.atoms)}")
        for a, p in zip(self.atoms, xyz):
            a.x, a.y, a.z = float(p[0]), float(p[1]), float(p[2])

    def centre(self) -> np.ndarray:
        xyz = self.coords()
        return xyz.mean(0) if len(xyz) else np.zeros(3)

    def radius_of_gyration(self) -> float:
        xyz = self.coords()
        if len(xyz) < 2:
            return 0.0
        return float(np.sqrt(((xyz - xyz.mean(0)) ** 2).sum(1).mean()))

    def ca_trace(self, chain: str | None = None) -> tuple[np.ndarray, list[tuple]]:
        """Ca coordinates in file order, plus their reskeys."""
        pts, keys = [], []
        for a in self.atoms:
            if a.name.strip() != "CA" or a.record != "ATOM":
                continue
            if a.resname not in AMINO3:
                continue
            if chain is not None and a.chain != chain:
                continue
            pts.append([a.x, a.y, a.z])
            keys.append(a.reskey)
        return np.array(pts, dtype=float).reshape(-1, 3), keys

    # -- edits -------------------------------------------------------------
    def rename_chain(self, old: str, new: str) -> None:
        for a in self.atoms:
            if a.chain == old:
                a.chain = new

    def renumber_serials(self, start: int = 1) -> None:
        for i, a in enumerate(self.atoms):
            a.serial = start + i


# ---------------------------------------------------------------------------
# element guessing
# ---------------------------------------------------------------------------

def guess_element(name: str, resname: str = "") -> str:
    nm = name.strip()
    if not nm:
        return "C"
    # Standard PDB: element right-justified in the atom-name field, so a name
    # starting in column 13 with two letters is a two-letter element (ZN, FE).
    if len(nm) >= 2 and nm[:2].upper() in (
            "ZN", "FE", "MG", "MN", "CA", "NA", "CL", "CU", "CO", "NI", "SE",
            "BR", "CD", "HG", "PT", "AU", "MO", "K", "LI"):
        # "CA" is ambiguous: alpha-carbon vs calcium. Protein residue -> carbon.
        if nm[:2].upper() == "CA" and resname in AMINO3:
            return "C"
        if resname.upper() == nm[:2].upper():
            # Uppercase, matching RCSB and the PDB spec. Mixed case ("Zn")
            # is legal but then string comparisons against element tables
            # silently miss, and a zinc renders with a carbon radius.
            return nm[:2].upper()
    if nm[0].isdigit():
        nm = nm[1:]
    return _ELEMENT_FROM_NAME.get(nm[0].upper(), nm[0].upper())


# ---------------------------------------------------------------------------
# PDB I/O
# ---------------------------------------------------------------------------

def _parse_pdb_atom(line: str) -> Atom | None:
    try:
        rec = line[0:6].strip()
        if rec not in ("ATOM", "HETATM"):
            return None
        name = line[12:16].strip()
        resname = line[17:20].strip()
        element = line[76:78].strip()
        try:
            serial = int(line[6:11])
        except ValueError:
            serial = 0
        try:
            resseq = int(line[22:26])
        except ValueError:
            return None
        return Atom(
            record=rec,
            serial=serial,
            name=name,
            altloc=line[16].strip(),
            resname=resname,
            chain=(line[21].strip() or "A"),
            resseq=resseq,
            icode=line[26].strip(),
            x=float(line[30:38]), y=float(line[38:46]), z=float(line[46:54]),
            occupancy=float(line[54:60]) if line[54:60].strip() else 1.0,
            bfactor=float(line[60:66]) if line[60:66].strip() else 0.0,
            element=(element or guess_element(name, resname)),
        )
    except (ValueError, IndexError):
        return None


def read_pdb(path: str, model: int = 1) -> Structure:
    """Read one model of a PDB file (default: the first)."""
    atoms, current, started = [], 0, False
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            tag = line[:6]
            if tag.startswith("MODEL"):
                started = True
                try:
                    current = int(line[10:].split()[0])
                except (ValueError, IndexError):
                    current += 1
                continue
            if tag.startswith("ENDMDL"):
                if started and current == model:
                    break
                continue
            if started and current != model:
                continue
            a = _parse_pdb_atom(line)
            if a is not None:
                atoms.append(a)
    return Structure(atoms, name=os.path.basename(path))


def read_pdb_models(path: str) -> list[Structure]:
    """Every MODEL in a multi-model PDB. A file with no MODEL records comes
    back as a one-element list, which is what RFdiffusion's single-step
    output looks like."""
    models: list[Structure] = []
    cur: list[Atom] = []
    saw_model = False
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            if line.startswith("MODEL"):
                saw_model = True
                cur = []
                continue
            if line.startswith("ENDMDL"):
                models.append(Structure(cur, f"{os.path.basename(path)}#{len(models)+1}"))
                cur = []
                continue
            a = _parse_pdb_atom(line)
            if a is not None:
                cur.append(a)
    if cur:
        models.append(Structure(cur, f"{os.path.basename(path)}#{len(models)+1}"))
    if not saw_model and len(models) == 1:
        models[0].name = os.path.basename(path)
    return models


def write_pdb(st: Structure, path: str, remarks: Sequence[str] = ()) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as fh:
        for r in remarks:
            fh.write(f"REMARK 250 {r[:68]}\n")
        _write_atom_records(fh, st.atoms)
        fh.write("END\n")


def _fmt_atom(a: Atom, serial: int) -> str:
    # Atom names: 4-char field. One- and two-letter elements start in col 14
    # so the name reads as a name, not as a shifted element.
    nm = a.name.strip()
    if len(nm) >= 4:
        name = nm[:4]
    elif len(a.element.strip()) == 2 and nm[:2].upper() == a.element.strip().upper():
        name = f"{nm:<4}"
    else:
        name = f" {nm:<3}"
    return (
        f"{a.record:<6}{serial % 100000:5d} {name}{(a.altloc or ' '):1}"
        f"{a.resname[:3]:>3} {a.chain[:1]:1}{a.resseq % 10000:4d}{(a.icode or ' '):1}   "
        f"{a.x:8.3f}{a.y:8.3f}{a.z:8.3f}"
        f"{a.occupancy:6.2f}{a.bfactor:6.2f}          {a.element.strip()[:2]:>2}\n"
    )


def _write_atom_records(fh, atoms: Sequence[Atom]) -> None:
    prev_chain, serial = None, 0
    for a in atoms:
        serial += 1
        if prev_chain is not None and a.chain != prev_chain:
            fh.write("TER\n")
        fh.write(_fmt_atom(a, serial))
        prev_chain = a.chain
    fh.write("TER\n")


def write_trajectory(frames: Sequence[np.ndarray], template: Structure,
                     path: str, remarks: Sequence[str] = ()) -> None:
    """
    Multi-MODEL PDB, one MODEL per frame, identical atom records throughout.

    ChimeraX must open this with `coordset true` to get a single model with a
    coordinate set rather than N sibling models -- see render_shrink.cxc.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    atoms = template.atoms
    with open(path, "w") as fh:
        for r in remarks:
            fh.write(f"REMARK 250 {r[:68]}\n")
        fh.write(f"REMARK 250 {len(frames)} frames, {len(atoms)} atoms per frame\n")
        for i, xyz in enumerate(frames, start=1):
            if len(xyz) != len(atoms):
                raise ValueError(
                    f"frame {i} has {len(xyz)} coords, template has {len(atoms)} atoms")
            fh.write(f"MODEL     {i:4d}\n")
            serial = 0
            prev_chain = None
            for a, p in zip(atoms, xyz):
                serial += 1
                if prev_chain is not None and a.chain != prev_chain:
                    fh.write("TER\n")
                moved = replace(a, x=float(p[0]), y=float(p[1]), z=float(p[2]))
                fh.write(_fmt_atom(moved, serial))
                prev_chain = a.chain
            fh.write("TER\nENDMDL\n")
        fh.write("END\n")


# ---------------------------------------------------------------------------
# mmCIF I/O -- AF3 / Boltz / Chai design files
# ---------------------------------------------------------------------------

def _cif_tokens(line: str) -> list[str]:
    """Split an mmCIF data line, honouring ' and " quoting."""
    out, i, n = [], 0, len(line)
    while i < n:
        c = line[i]
        if c in " \t":
            i += 1
            continue
        if c in "'\"":
            q, i = c, i + 1
            start = i
            while i < n:
                if line[i] == q and (i + 1 >= n or line[i + 1] in " \t"):
                    break
                i += 1
            out.append(line[start:i])
            i += 1
        else:
            start = i
            while i < n and line[i] not in " \t":
                i += 1
            out.append(line[start:i])
    return out


def read_cif(path: str, model: int = 1) -> Structure:
    """
    Read the _atom_site loop of an mmCIF. auth_* columns win over label_*
    where both exist, which is what keeps AF3 output in UniProt numbering.
    """
    with open(path, "r", errors="replace") as fh:
        lines = fh.read().splitlines()

    cols: list[str] = []
    rows: list[list[str]] = []
    i, n = 0, len(lines)
    while i < n:
        s = lines[i].strip()
        if s.startswith("loop_"):
            j, names = i + 1, []
            while j < n and lines[j].strip().startswith("_"):
                names.append(lines[j].strip().split()[0])
                j += 1
            if names and names[0].startswith("_atom_site."):
                cols = [nm.split(".", 1)[1] for nm in names]
                while j < n:
                    row = lines[j]
                    t = row.strip()
                    if not t or t.startswith("#") or t.startswith("loop_") or t.startswith("_"):
                        break
                    if t.startswith("data_") or t.startswith("save_"):
                        break
                    tok = _cif_tokens(row)
                    if len(tok) == len(cols):
                        rows.append(tok)
                    j += 1
                break
            i = j
            continue
        i += 1

    if not cols or not rows:
        raise RuntimeError(f"no _atom_site loop found in {path}")

    idx = {c: k for k, c in enumerate(cols)}

    def get(row, *names, default=""):
        for nm in names:
            k = idx.get(nm)
            if k is not None:
                v = row[k]
                if v not in (".", "?", ""):
                    return v
        return default

    atoms = []
    for row in rows:
        mdl = get(row, "pdbx_PDB_model_num", default="1")
        try:
            if int(float(mdl)) != model:
                continue
        except ValueError:
            pass
        try:
            x = float(get(row, "Cartn_x"))
            y = float(get(row, "Cartn_y"))
            z = float(get(row, "Cartn_z"))
        except ValueError:
            continue
        seq_raw = get(row, "auth_seq_id", "label_seq_id", default="")
        try:
            resseq = int(float(seq_raw))
        except ValueError:
            continue
        name = get(row, "auth_atom_id", "label_atom_id", default="C")
        resname = get(row, "auth_comp_id", "label_comp_id", default="UNK")
        elem = get(row, "type_symbol", default="") or guess_element(name, resname)
        try:
            occ = float(get(row, "occupancy", default="1.0"))
        except ValueError:
            occ = 1.0
        try:
            bfac = float(get(row, "B_iso_or_equiv", default="0.0"))
        except ValueError:
            bfac = 0.0
        atoms.append(Atom(
            record=get(row, "group_PDB", default="ATOM").upper() or "ATOM",
            serial=len(atoms) + 1,
            name=name,
            altloc=(get(row, "label_alt_id", default="") or ""),
            resname=resname.upper(),
            chain=get(row, "auth_asym_id", "label_asym_id", default="A"),
            resseq=resseq,
            icode=get(row, "pdbx_PDB_ins_code", default=""),
            x=x, y=y, z=z,
            occupancy=occ, bfactor=bfac,
            element=elem.upper(),
        ))
    return Structure(atoms, name=os.path.basename(path))


def read_structure(path: str) -> Structure:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".cif", ".mmcif"):
        return read_cif(path)
    return read_pdb(path)


# ---------------------------------------------------------------------------
# superposition and geometry
# ---------------------------------------------------------------------------

def kabsch(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimal rigid superposition of P onto Q.

    Returns (R, t, rmsd) with the convention  P @ R.T + t  ~=  Q,
    matching apply_transform() below. The reflection guard is the usual
    det-correction -- without it a near-planar motif can fit as its mirror
    image, which superposes beautifully and is chemically nonsense.
    """
    P = np.asarray(P, dtype=float).reshape(-1, 3)
    Q = np.asarray(Q, dtype=float).reshape(-1, 3)
    if len(P) != len(Q) or len(P) < 3:
        raise ValueError(f"kabsch needs >=3 paired points, got {len(P)} and {len(Q)}")

    pc, qc = P.mean(0), Q.mean(0)
    Pc, Qc = P - pc, Q - qc

    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T

    t = qc - pc @ R.T
    rmsd = float(np.sqrt((((Pc @ R.T) - Qc) ** 2).sum(1).mean()))
    return R, t, rmsd


def apply_transform(st: Structure, R: np.ndarray, t: np.ndarray) -> None:
    """In-place: xyz -> xyz @ R.T + t."""
    xyz = st.coords()
    if not len(xyz):
        return
    st.set_coords(xyz @ np.asarray(R).T + np.asarray(t))


def min_distance(group_a: Sequence[Atom], group_b: Sequence[Atom]) -> float:
    """Closest approach between two atom groups. inf if either is empty, so
    callers can use `> cutoff` as 'not touching' without a None check."""
    if not group_a or not group_b:
        return float("inf")
    A = np.array([[a.x, a.y, a.z] for a in group_a], dtype=float)
    B = np.array([[b.x, b.y, b.z] for b in group_b], dtype=float)
    return float(np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)).min())


def contacts(group: Sequence[Atom], ligand: Sequence[Atom],
             cutoff: float = 4.0) -> list[tuple]:
    """
    Residues in `group` with any heavy atom within `cutoff` of `ligand`.

    Returns (chain, resseq, resname) sorted by residue number -- three-tuples,
    not the four-tuples residues() yields, because callers format these as
    "GLY312" for the on-screen contact list.
    """
    if not group or not ligand:
        return []
    A = np.array([[a.x, a.y, a.z] for a in group], dtype=float)
    B = np.array([[b.x, b.y, b.z] for b in ligand], dtype=float)
    d = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)).min(1)
    hit, out = set(), []
    for a, dist in zip(group, d):
        if dist > cutoff:
            continue
        k = (a.chain, a.resseq, a.resname)
        if k not in hit:
            hit.add(k)
            out.append(k)
    return sorted(out, key=lambda k: (k[0], k[1]))


# ---------------------------------------------------------------------------
# anisotropic network model
# ---------------------------------------------------------------------------

def anm_modes(ca: np.ndarray, cutoff: float = 15.0,
              n_modes: int = 3, gamma: float = 1.0
              ) -> tuple[np.ndarray, np.ndarray]:
    """
    Ca anisotropic network model.

    Every Ca pair within `cutoff` gets a Hookean spring along its connecting
    vector; the 3N x 3N Hessian's low eigenvectors are the collective motions.
    The first six modes are rigid-body translation and rotation (eigenvalue
    ~0) and are discarded, so mode 7 is the softest real deformation -- for a
    two-lobe complex that is almost always the hinge between the lobes, which
    is exactly the "breathing" the hero frame wants.

    Returns (eigenvalues[n_modes], eigenvectors[n_modes, n_ca, 3]) with each
    eigenvector normalised to unit per-atom RMS displacement.
    """
    ca = np.asarray(ca, dtype=float).reshape(-1, 3)
    n = len(ca)
    if n < 4:
        raise ValueError(f"ANM needs >=4 Ca, got {n}")

    diff = ca[:, None, :] - ca[None, :, :]
    dist2 = (diff ** 2).sum(-1)
    np.fill_diagonal(dist2, np.inf)
    near = dist2 <= cutoff ** 2

    # A cutoff that isolates an atom leaves a zero row and an extra spurious
    # zero mode; widening for the stragglers is cheaper than failing.
    lonely = ~near.any(1)
    if lonely.any():
        for i in np.flatnonzero(lonely):
            near[i, np.argsort(dist2[i])[:4]] = True
        near |= near.T

    H = np.zeros((3 * n, 3 * n))
    for i in range(n):
        js = np.flatnonzero(near[i])
        for j in js:
            if j <= i:
                continue
            d2 = dist2[i, j]
            v = diff[i, j]
            block = -gamma * np.outer(v, v) / d2
            H[3*i:3*i+3, 3*j:3*j+3] = block
            H[3*j:3*j+3, 3*i:3*i+3] = block
            H[3*i:3*i+3, 3*i:3*i+3] -= block
            H[3*j:3*j+3, 3*j:3*j+3] -= block

    vals, vecs = np.linalg.eigh(H)
    keep = slice(6, 6 + n_modes)
    modes = vecs[:, keep].T.reshape(-1, n, 3)

    out = np.empty_like(modes)
    for k in range(len(modes)):
        rms = np.sqrt((modes[k] ** 2).sum(1).mean())
        out[k] = modes[k] / (rms if rms > 1e-12 else 1.0)
    return vals[keep].copy(), out


# ---------------------------------------------------------------------------
# network helpers -- never raise, always return a path or None
# ---------------------------------------------------------------------------

def _download(url: str, dest: str, timeout: float = 30.0) -> str | None:
    try:
        os.makedirs(os.path.dirname(os.path.abspath(dest)) or ".", exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "digiactivate-shrink/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read()
        if len(data) < 512:
            return None
        with open(dest, "wb") as fh:
            fh.write(data)
        return dest
    except (urllib.error.URLError, OSError, ValueError):
        return None


def fetch_rcsb(pdb_id: str, out_dir: str = "input") -> str | None:
    dest = os.path.join(out_dir, f"{pdb_id.lower()}.pdb")
    if os.path.exists(dest) and os.path.getsize(dest) > 512:
        return dest
    return _download(f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb", dest)


def fetch_alphafold(uniprot: str, out_dir: str = "input",
                    versions: Sequence[int] = (4, 3, 2)) -> str | None:
    dest = os.path.join(out_dir, f"AF-{uniprot}-F1.pdb")
    if os.path.exists(dest) and os.path.getsize(dest) > 512:
        return dest
    for v in versions:
        url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot}-F1-model_v{v}.pdb"
        got = _download(url, dest)
        if got:
            return got
    return None
