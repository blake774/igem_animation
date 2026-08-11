#!/usr/bin/env python3
"""
test_fallbacks.py -- exercise the paths that only run when something is wrong.

The happy path gets tested every time anyone runs the pipeline. These do not:
they fire when the network is down, when a design file is in a slightly
different dialect, when a trajectory has the wrong number of atoms. Each one
is a branch that would otherwise be discovered during a render.

    python3 test_fallbacks.py            # needs input/6w74.pdb
    python3 test_fallbacks.py -v         # show the pipeline's own output

No pytest dependency -- this runs anywhere the pipeline runs.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

import structlib as S

HERE = os.path.dirname(os.path.abspath(__file__))
RECEPTOR = os.path.join(HERE, "input", "6w74.pdb")

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


@contextlib.contextmanager
def quiet(verbose: bool):
    if verbose:
        yield
        return
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield


# ---------------------------------------------------------------------------
# structlib-level
# ---------------------------------------------------------------------------

def test_network_offline() -> None:
    """Both fetchers must return None, not raise, when the network is gone."""
    with tempfile.TemporaryDirectory() as td:
        got = S.fetch_rcsb("XXXX", td)
        check("fetch_rcsb returns None for a bad id", got is None, repr(got))
        got = S.fetch_alphafold("NOTAUNIPROT", td, versions=(4,))
        check("fetch_alphafold returns None when unreachable", got is None, repr(got))


def test_pdb_edge_cases() -> None:
    """Files that are legal PDB but not what a naive reader expects."""
    with tempfile.TemporaryDirectory() as td:
        # no element column, names shifted, insertion codes, altlocs
        p = os.path.join(td, "odd.pdb")
        with open(p, "w") as fh:
            fh.write(
                "ATOM      1  N   ALA A  10      11.104   6.134   7.822  1.00 20.00\n"
                "ATOM      2  CA  ALA A  10      12.560   6.204   7.795  1.00 20.00\n"
                "ATOM      3  CB  ALA A  10A     13.000   7.000   8.000  1.00 20.00\n"
                "ATOM      4  CA BALA A  11      13.000   6.000   9.000  0.50 20.00\n"
                "HETATM    5 ZN    ZN A 401      10.000  10.000  10.000  1.00 20.00\n"
                "END\n")
        st = S.read_pdb(p)
        check("reads a PDB with no element column", len(st) == 5, f"{len(st)} atoms")
        elems = {a.name.strip(): a.element for a in st.atoms}
        check("infers element for backbone atoms",
              elems.get("CA") == "C" and elems.get("N") == "N", str(elems))
        check("infers ZN as zinc, not carbon", elems.get("ZN") == "ZN", str(elems))
        ic = [a for a in st.atoms if a.icode == "A"]
        check("keeps insertion codes", len(ic) == 1, f"{len(ic)} atoms with icode")
        alt = [a for a in st.atoms if a.altloc == "B"]
        check("keeps altloc labels", len(alt) == 1, f"{len(alt)} altloc B")

        # a file with zero atoms should read as empty, not explode
        p2 = os.path.join(td, "empty.pdb")
        open(p2, "w").write("REMARK nothing here\nEND\n")
        st2 = S.read_pdb(p2)
        check("empty PDB reads as zero atoms", len(st2) == 0)
        check("Rg of an empty structure is 0", st2.radius_of_gyration() == 0.0)


def test_cif_dialects() -> None:
    """AF3-style mmCIF with and without auth_* columns."""
    with tempfile.TemporaryDirectory() as td:
        label_only = """\
data_x
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
ATOM 1 N N ALA A 1 1.000 2.000 3.000 1.00 90.00
ATOM 2 C CA ALA A 1 2.000 2.000 3.000 1.00 90.00
ATOM 3 C CA GLY A 2 5.000 2.000 3.000 1.00 88.00
#
"""
        p = os.path.join(td, "label.cif")
        open(p, "w").write(label_only)
        st = S.read_cif(p)
        check("mmCIF without auth_* columns falls back to label_*",
              len(st) == 3 and st.atoms[0].chain == "A", repr(st))
        ca, keys = st.ca_trace()
        check("mmCIF Ca trace found", len(ca) == 2, f"{len(ca)} Ca")

        # quoted atom names and a second model that must be ignored
        quoted = label_only.replace(
            "_atom_site.B_iso_or_equiv\n",
            "_atom_site.B_iso_or_equiv\n_atom_site.pdbx_PDB_model_num\n")
        quoted = quoted.replace(" 1.00 90.00\n", " 1.00 90.00 1\n")
        quoted = quoted.replace(" 1.00 88.00\n", " 1.00 88.00 1\n")
        quoted += "ATOM 4 C CA GLY A 3 9.000 2.000 3.000 1.00 88.00 2\n#\n"
        p2 = os.path.join(td, "models.cif")
        open(p2, "w").write(quoted)
        st2 = S.read_cif(p2)
        check("mmCIF reads model 1 only", len(st2) == 3, f"{len(st2)} atoms")


def test_kabsch() -> None:
    """Superposition has to recover a known transform, and must not mirror."""
    rng = np.random.default_rng(0)
    P = rng.normal(size=(20, 3)) * 5
    theta = 0.7
    R_true = np.array([[np.cos(theta), -np.sin(theta), 0],
                       [np.sin(theta), np.cos(theta), 0],
                       [0, 0, 1.0]])
    t_true = np.array([3.0, -2.0, 8.0])
    Q = P @ R_true.T + t_true
    R, t, rms = S.kabsch(P, Q)
    check("kabsch recovers a known rotation+translation", rms < 1e-8, f"rmsd {rms:.2e}")
    check("kabsch transform round-trips",
          np.abs((P @ R.T + t) - Q).max() < 1e-8)
    check("kabsch never returns a reflection",
          np.linalg.det(R) > 0, f"det {np.linalg.det(R):.4f}")

    # a mirrored copy must NOT superpose to zero
    Qm = P.copy()
    Qm[:, 2] *= -1
    _, _, rms_m = S.kabsch(P, Qm)
    check("kabsch refuses to fit a mirror image", rms_m > 1.0, f"rmsd {rms_m:.2f}")


def test_anm() -> None:
    ca = np.array([[i * 3.8, 0.0, 0.0] for i in range(20)])
    vals, vecs = S.anm_modes(ca, cutoff=12.0, n_modes=3)
    check("ANM returns the requested number of modes",
          len(vals) == 3 and vecs.shape == (3, 20, 3), f"{vecs.shape}")
    check("ANM eigenvalues are non-negative", (vals > -1e-8).all(), str(vals.round(4)))
    check("ANM modes are normalised to unit RMS",
          abs(np.sqrt((vecs[0] ** 2).sum(1).mean()) - 1.0) < 1e-6)
    # an atom isolated beyond the cutoff must not crash the Hessian
    ca2 = np.vstack([ca, [500.0, 0.0, 0.0]])
    vals2, _ = S.anm_modes(ca2, cutoff=12.0, n_modes=2)
    check("ANM survives an atom outside every cutoff", len(vals2) == 2)


def test_trajectory_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as td:
        atoms = [S.Atom(name="CA", resname="ALA", chain="B", resseq=i + 1,
                        x=float(i), y=0.0, z=0.0, element="C") for i in range(10)]
        st = S.Structure(atoms)
        frames = [st.coords() + i for i in range(5)]
        p = os.path.join(td, "traj.pdb")
        S.write_trajectory(frames, st, p)
        back = S.read_pdb_models(p)
        check("trajectory round-trips frame count", len(back) == 5, f"{len(back)}")
        check("trajectory round-trips atom count",
              all(len(m) == 10 for m in back))
        err = max(float(np.abs(b.coords() - f).max()) for b, f in zip(back, frames))
        check("trajectory round-trips coordinates", err < 5e-4, f"max err {err:.1e}")

        # mismatched frame must be rejected loudly, not written half-way
        try:
            S.write_trajectory([np.zeros((3, 3))], st, os.path.join(td, "bad.pdb"))
            check("write_trajectory rejects a wrong-sized frame", False)
        except ValueError:
            check("write_trajectory rejects a wrong-sized frame", True)


# ---------------------------------------------------------------------------
# pipeline-level
# ---------------------------------------------------------------------------

def run_prep(tmp: str, extra: list[str], verbose: bool) -> tuple[int, str]:
    design = os.path.join(tmp, "input", "synthetic_design.cif")
    cmd = [sys.executable, os.path.join(HERE, "prep_shrink.py"),
           "--design", design, "--receptor", RECEPTOR,
           "--out-dir", os.path.join(tmp, "output"),
           "--input-dir", os.path.join(tmp, "input")] + extra
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=HERE)
    if verbose:
        print(r.stdout[-3000:], r.stderr[-2000:])
    return r.returncode, r.stdout + r.stderr


def make_inputs(tmp: str, verbose: bool) -> None:
    os.makedirs(os.path.join(tmp, "input"), exist_ok=True)
    cmd = [sys.executable, os.path.join(HERE, "make_testdata.py"),
           "--receptor", RECEPTOR, "--out-dir", os.path.join(tmp, "input"),
           "--steps", "1200"]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=HERE)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:])
        raise RuntimeError("make_testdata.py failed")


def test_pipeline_fallbacks(verbose: bool) -> None:
    if not os.path.exists(RECEPTOR):
        print(f"  [skip] pipeline tests need {RECEPTOR}")
        return
    with tempfile.TemporaryDirectory() as tmp:
        make_inputs(tmp, verbose)
        out = os.path.join(tmp, "output")

        # 1 -- no full-length model, no trajectory: the fully degraded run
        rc, log = run_prep(tmp, [], verbose)
        check("runs with no full-length model and no trajectory", rc == 0)
        check("warns that the full-length model is missing",
              "no full-length model" in log)
        check("falls back to the synthetic condensation",
              "SYNTHETIC" in log or "synthesising" in log)
        with open(os.path.join(out, "metrics.json")) as fh:
            M = json.load(fh)
        check("metrics records the synthetic source",
              M["trajectories"]["condense"]["source"] == "synthetic",
              M["trajectories"]["condense"]["source"])
        check("full_length_residues is null, not zero",
              M["full_length_residues"] is None)
        opens = [l for l in open(os.path.join(out, "00_open.cxc"))
                 if l.startswith("open ")]
        check("00_open.cxc substitutes BIR3 for the missing model #1",
              len(opens) == 7 and "02_bir3_ref.pdb" in opens[0],
              opens[0].strip() if opens else "no open lines")

        # 2 -- a trajectory whose models disagree on atom count
        bad = os.path.join(tmp, "input", "ragged.pdb")
        with open(bad, "w") as fh:
            for m in range(6):
                fh.write(f"MODEL     {m+1:4d}\n")
                for i in range(10 - (m == 3)):     # model 4 is one atom short
                    fh.write(f"ATOM  {i+1:5d}  CA  ALA B{i+1:4d}    "
                             f"{i:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C\n")
                fh.write("ENDMDL\n")
        rc, log = run_prep(tmp, ["--rfd-traj", bad], verbose)
        check("survives a ragged trajectory", rc == 0)
        check("says why the ragged trajectory was rejected",
              "inconsistent atom counts" in log, log[-200:] if rc else "")

        # 3 -- a trajectory with too few models
        few = os.path.join(tmp, "input", "few.pdb")
        with open(few, "w") as fh:
            for m in range(3):
                fh.write(f"MODEL     {m+1:4d}\n")
                for i in range(10):
                    fh.write(f"ATOM  {i+1:5d}  CA  ALA B{i+1:4d}    "
                             f"{i:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C\n")
                fh.write("ENDMDL\n")
        rc, log = run_prep(tmp, ["--rfd-traj", few], verbose)
        check("survives a too-short trajectory", rc == 0)
        check("says why the short trajectory was rejected", "too few models" in log)

        # 4 -- a trajectory pointing at nothing
        rc, _ = run_prep(tmp, ["--rfd-traj", os.path.join(tmp, "nope.pdb")], verbose)
        check("survives a --rfd-traj path that does not exist", rc == 0)

        # 5 -- the good path, and the direction/superposition logic
        traj = os.path.join(tmp, "input", "synthetic_rfd_traj.pdb")
        rc, log = run_prep(tmp, ["--rfd-traj", traj], verbose)
        check("accepts a well-formed trajectory", rc == 0)
        with open(os.path.join(out, "metrics.json")) as fh:
            M = json.load(fh)
        cond = M["trajectories"]["condense"]
        check("uses the real trajectory", cond["source"] == "rfdiffusion")
        check("detects the noise end", cond["disorder_first"] > cond["disorder_last"],
              f"{cond['disorder_first']} vs {cond['disorder_last']}")
        check("superposes the trajectory onto the binder",
              cond["superposition_rmsd"] < 1.0, f"{cond['superposition_rmsd']} A")
        check("resampled trajectory stays smooth",
              cond["max_frame_jump_A"] < 3.0, f"{cond['max_frame_jump_A']} A")

        # 6 -- a trajectory written the other way round must be flipped
        models = S.read_pdb_models(traj)
        rev = os.path.join(tmp, "input", "reversed.pdb")
        S.write_trajectory([m.coords() for m in models][::-1], models[0], rev)
        rc, log = run_prep(tmp, ["--rfd-traj", rev], verbose)
        check("reverses a folded->noise trajectory", rc == 0 and "reversed" in log)

        # 7 -- validate.py and lint_cxc.py must both pass on that output
        r = subprocess.run([sys.executable, os.path.join(HERE, "validate.py"),
                            "--out-dir", out], capture_output=True, text=True, cwd=HERE)
        check("validate.py passes on a good run", r.returncode == 0,
              r.stdout.strip().split("\n")[-1] if r.stdout else "")

        # 8 -- lint must FAIL on a deliberately broken script
        broken = os.path.join(tmp, "10_shots.cxc")
        shutil.copy(os.path.join(HERE, "10_shots.cxc"), broken)
        txt = open(broken).read().replace("color hotspotsRef cHotspot",
                                          "color hotspotsRef cNoSuchColour")
        open(broken, "w").write(txt)
        r = subprocess.run([sys.executable, os.path.join(HERE, "lint_cxc.py"),
                            "--out-dir", out],
                           capture_output=True, text=True, cwd=tmp)
        check("lint_cxc.py catches an undefined colour",
              r.returncode != 0 and "cNoSuchColour" in r.stdout,
              r.stdout.strip().split("\n")[-1] if r.stdout else "")


def make_fulllength_stub(path: str) -> int:
    """
    A throwaway multi-domain structure, built by tiling BIR3 along a line.

    Not a model of anything -- its only job is to exercise the branches that
    run when 01_ciap1_full.pdb exists, so that dropping in the real
    AF-Q13490 does not hit them for the first time mid-render. It is written
    to a temp dir and never leaves it.
    """
    st = S.read_pdb(RECEPTOR)
    prot = [a for a in st.atoms if a.record == "ATOM"]
    rng = np.random.default_rng(2)
    out, resoff = [], 0
    for k in range(7):
        shift = rng.normal(scale=18.0, size=3) + np.array([k * 22.0, 0.0, 0.0])
        for a in prot:
            out.append(S.Atom(record="ATOM", name=a.name, resname=a.resname,
                              chain="A", resseq=a.resseq - 265 + resoff,
                              x=a.x + shift[0], y=a.y + shift[1],
                              z=a.z + shift[2], bfactor=80.0,
                              element=a.element))
        resoff += 88
    big = S.Structure(out).select(lambda a: a.resseq <= 618)
    big.renumber_serials()
    S.write_pdb(big, path, remarks=["THROWAWAY STUB - not a real structure"])
    return big.n_protein_residues()


def test_fulllength_path(verbose: bool) -> None:
    """The branches that only run when a full-length model is supplied."""
    if not os.path.exists(RECEPTOR):
        print(f"  [skip] full-length tests need {RECEPTOR}")
        return
    with tempfile.TemporaryDirectory() as tmp:
        make_inputs(tmp, verbose)
        out = os.path.join(tmp, "output")
        stub = os.path.join(tmp, "input", "AF-STUB.pdb")
        n_res = make_fulllength_stub(stub)
        check("built a multi-domain stub", n_res > 550, f"{n_res} residues")

        rc, log = run_prep(tmp, ["--fulllength", stub,
                                 "--rfd-traj", os.path.join(tmp, "input",
                                                            "synthetic_rfd_traj.pdb")],
                           verbose)
        check("runs with a full-length model", rc == 0)
        check("does not warn about a missing full-length model",
              "no full-length model" not in log)
        check("writes 01_ciap1_full.pdb",
              os.path.exists(os.path.join(out, "01_ciap1_full.pdb")))

        opens = [l for l in open(os.path.join(out, "00_open.cxc"))
                 if l.startswith("open ")]
        check("00_open.cxc opens the full-length model as #1",
              len(opens) == 7 and "01_ciap1_full.pdb" in opens[0],
              opens[0].strip() if opens else "no open lines")

        with open(os.path.join(out, "metrics.json")) as fh:
            M = json.load(fh)
        check("reports full_length_residues", M["full_length_residues"] == n_res,
              str(M["full_length_residues"]))
        check("reports a shrink factor > 1", M.get("shrink_factor", 0) > 1.0,
              f"{M.get('shrink_factor')}x")
        check("counters name the full-length protein",
              "cIAP1 (full length)" in log)

        # The renderer has to frame it. Import is guarded: bpy is optional.
        try:
            import importlib
            import render_blender as RB
            importlib.reload(RB)
        except SystemExit:
            print("  [skip] renderer framing (bpy not installed)")
            return
        A = RB.Assets(out)
        check("renderer sees the full-length model", A.full is not None)
        d = RB.fit_distance(A.x_open, A.full_centre, -35.0, 14.0, 48.0, 1.12)
        span = float(np.linalg.norm(A.x_open - A.full_centre, axis=1).max())
        check("framing distance is finite and sane",
              0.0 < d < span * 40, f"{d:.1f} scene units for span {span:.1f}")
        # An elongated subject must be framed closer than a sphere fit would.
        sphere_d = RB.fit_distance(
            np.array([A.full_centre + [span, 0, 0], A.full_centre - [span, 0, 0],
                      A.full_centre + [0, span, 0], A.full_centre - [0, span, 0],
                      A.full_centre + [0, 0, span], A.full_centre - [0, 0, span]]),
            A.full_centre, -35.0, 14.0, 48.0, 1.12)
        check("projection fit beats a sphere fit on an elongated subject",
              d < sphere_d, f"{d:.1f} vs {sphere_d:.1f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    print("\n[structlib]")
    test_network_offline()
    test_pdb_edge_cases()
    test_cif_dialects()
    test_kabsch()
    test_anm()
    test_trajectory_roundtrip()

    print("\n[pipeline fallbacks]")
    test_pipeline_fallbacks(args.verbose)

    print("\n[full-length model path]")
    test_fulllength_path(args.verbose)

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  failed: {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
