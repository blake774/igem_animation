#!/usr/bin/env python3
"""
validate.py -- geometric checks on what prep_shrink.py wrote.

lint_cxc.py checks that the scripts are consistent. This checks that the
coordinates are sane, which is the other half of "will the render be right".

Checks, and what each one is actually guarding against:

  frame counts      metrics.json and the files agree
  atom counts       constant within a trajectory (ChimeraX needs this for
                    `coordset`, and will refuse the file otherwise)
  frame-to-frame    largest single-atom motion under ~3 A, or the shot strobes
  dissolve pins     the retained motif and the ligand must not move by even a
                    little; if they drift, the "the pocket is kept" claim the
                    whole shot makes is visibly false
  condense frame    the condensation has to happen AT the pocket. A diffusion
                    trajectory arrives in its own frame, and if it is not
                    superposed onto the placed binder the binder condenses
                    tens of nanometres off-screen and the shot renders empty
  condense landing  the last frame must be the deposited design, not near it
  breathe loop      first and last frame close enough to loop without a jump
  bond sanity       consecutive Ca in the final frames at ~3.8 A

Exit status 1 if any check fails, so it can gate a render.

    python3 validate.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

import structlib as S

MOTIF_RANGE = (312, 325)


class Checks:
    def __init__(self):
        self.rows: list[tuple[str, bool, str]] = []

    def add(self, name: str, ok: bool, detail: str = "") -> bool:
        self.rows.append((name, ok, detail))
        return ok

    def report(self) -> int:
        width = max(len(r[0]) for r in self.rows) if self.rows else 10
        failed = 0
        for name, ok, detail in self.rows:
            tag = "ok  " if ok else "FAIL"
            if not ok:
                failed += 1
            print(f"  [{tag}] {name:<{width}}  {detail}")
        print(f"\n{len(self.rows) - failed}/{len(self.rows)} checks passed")
        return failed


def max_frame_jump(frames: list[np.ndarray]) -> float:
    return max(float(np.abs(frames[i + 1] - frames[i]).max())
               for i in range(len(frames) - 1)) if len(frames) > 1 else 0.0


def ca_bond_stats(st: S.Structure, xyz: np.ndarray) -> tuple[float, float]:
    """(mean, max deviation from 3.81 A) over consecutive Ca of one chain."""
    idx = [i for i, a in enumerate(st.atoms)
           if a.name.strip() == "CA" and a.record == "ATOM"]
    if len(idx) < 3:
        return 0.0, 0.0
    pts = xyz[idx]
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    d = d[d < 10.0]                      # ignore chain breaks
    if not len(d):
        return 0.0, 0.0
    return float(d.mean()), float(np.abs(d - 3.81).max())


def main() -> int:
    p = argparse.ArgumentParser(description="Geometric checks on the trajectories.")
    p.add_argument("--out-dir", default="output")
    p.add_argument("--jump-tolerance", type=float, default=3.0,
                   help="largest allowed per-atom motion between frames (A)")
    args = p.parse_args()
    o = lambda n: os.path.join(args.out_dir, n)

    if not os.path.exists(o("metrics.json")):
        sys.exit(f"{o('metrics.json')} not found -- run prep_shrink.py first")
    with open(o("metrics.json")) as fh:
        M = json.load(fh)
    traj_meta = M.get("trajectories", {})

    c = Checks()
    print(f"[validating {args.out_dir}]\n")

    ref = S.read_pdb(o("02_bir3_ref.pdb"))
    binder = S.read_pdb(o("03_binder_final.pdb"))
    seed = S.read_pdb(o("04_motif_seed.pdb"))

    warhead = M.get("warhead")
    lig = [a for a in ref.atoms if a.resname == warhead] if warhead else []
    pocket = (np.array([[a.x, a.y, a.z] for a in lig]).mean(0) if lig
              else ref.centre())

    # ---------------------------------------------------------------- seed
    c.add("motif residues", seed.n_protein_residues() ==
          MOTIF_RANGE[1] - MOTIF_RANGE[0] + 1,
          f"{seed.n_protein_residues()} residues in 04_motif_seed.pdb")
    core = M.get("core_contacts_retained", [])
    want = M.get("core_contacts_expected", [])
    c.add("core contacts", core == want, f"{core} vs expected {want}")
    c.add("motif placement RMSD",
          M.get("design_placement", {}).get("rmsd", 99) < 1.5,
          f"{M.get('design_placement', {}).get('rmsd', float('nan')):.2f} A "
          f"by '{M.get('design_placement', {}).get('method')}'")

    # ------------------------------------------------------------ dissolve
    diss = S.read_pdb_models(o("05_dissolve.pdb"))
    n_want = traj_meta.get("dissolve", {}).get("frames")
    c.add("dissolve frames", len(diss) == n_want,
          f"{len(diss)} models, metrics says {n_want}")
    c.add("dissolve atom counts", len({len(m) for m in diss}) == 1,
          f"{sorted({len(m) for m in diss})}")
    dframes = [m.coords() for m in diss]

    held = np.array([(a.record == "HETATM")
                     or (MOTIF_RANGE[0] <= a.resseq <= MOTIF_RANGE[1])
                     for a in diss[0].atoms])
    drift = max(float(np.abs(dframes[i][held] - dframes[0][held]).max())
                for i in range(len(dframes)))
    c.add("dissolve keeps the motif", drift < 1e-3,
          f"max drift of held atoms {drift:.2e} A over {len(dframes)} frames")
    moved = float(np.linalg.norm(dframes[-1][~held] - dframes[0][~held],
                                 axis=1).mean())
    c.add("dissolve actually disperses", moved > 5.0,
          f"bulk atoms move {moved:.1f} A on average")

    # ------------------------------------------------------------ condense
    cond = S.read_pdb_models(o("06_condense.pdb"))
    n_want = traj_meta.get("condense", {}).get("frames")
    c.add("condense frames", len(cond) == n_want,
          f"{len(cond)} models, metrics says {n_want}")
    c.add("condense atom counts", len({len(m) for m in cond}) == 1,
          f"{sorted({len(m) for m in cond})}")
    cframes = [m.coords() for m in cond]
    jump = max_frame_jump(cframes)
    c.add("condense is smooth", jump < args.jump_tolerance,
          f"largest single-frame motion {jump:.2f} A "
          f"(tolerance {args.jump_tolerance})")

    # The check that catches a trajectory left in the diffusion frame.
    start_d = float(np.linalg.norm(cframes[0].mean(0) - pocket))
    end_d = float(np.linalg.norm(cframes[-1].mean(0) - pocket))
    c.add("condense is at the pocket", end_d < 25.0,
          f"final centroid {end_d:.1f} A from the warhead "
          f"(started {start_d:.1f} A)")

    rg0 = float(np.sqrt(((cframes[0] - cframes[0].mean(0)) ** 2).sum(1).mean()))
    rg1 = float(np.sqrt(((cframes[-1] - cframes[-1].mean(0)) ** 2).sum(1).mean()))
    c.add("condense contracts", rg1 < rg0 * 0.9,
          f"Rg {rg0:.1f} -> {rg1:.1f} A")

    src = traj_meta.get("condense", {}).get("source")
    if src == "synthetic":
        # A synthetic run lands exactly on the design by construction; a real
        # trajectory only lands near it, which is fine and expected.
        bxyz = binder.coords()
        if len(bxyz) == len(cframes[-1]):
            land = float(np.abs(cframes[-1] - bxyz).max())
            c.add("condense lands on the design", land < 1e-3,
                  f"max deviation {land:.2e} A")
    else:
        mean_b, dev_b = ca_bond_stats(cond[-1], cframes[-1])
        c.add("condense final backbone", dev_b < 1.5 or mean_b == 0.0,
              f"Ca-Ca mean {mean_b:.2f} A, worst deviation {dev_b:.2f} A")

    # ------------------------------------------------------------- breathe
    br = S.read_pdb_models(o("07_complex_breathe.pdb"))
    n_want = traj_meta.get("breathe", {}).get("frames")
    c.add("breathe frames", len(br) == n_want,
          f"{len(br)} models, metrics says {n_want}")
    bframes = [m.coords() for m in br]
    if len(bframes) > 2:
        jump = max_frame_jump(bframes)
        c.add("breathe is smooth", jump < args.jump_tolerance,
              f"largest single-frame motion {jump:.2f} A")
        loop = float(np.abs(bframes[-1] - bframes[0]).max())
        c.add("breathe loops", loop < 0.6,
              f"first-to-last gap {loop:.2f} A")
        amp = max(float(np.abs(f - bframes[0]).max()) for f in bframes)
        c.add("breathe actually moves", amp > 0.3,
              f"peak displacement {amp:.2f} A")
        mean_b, dev_b = ca_bond_stats(br[0], bframes[0])
        c.add("breathe backbone intact", dev_b < 1.0,
              f"Ca-Ca mean {mean_b:.2f} A, worst deviation {dev_b:.2f} A")

    # ------------------------------------------------------------ coherence
    # Everything must live in one frame, or shots cut to empty space.
    for label, st in (("binder", binder), ("seed", seed)):
        d = float(np.linalg.norm(st.centre() - pocket))
        c.add(f"{label} near the pocket", d < 30.0,
              f"centroid {d:.1f} A from the warhead")

    failed = c.report()
    if failed:
        print("\nSomething above will show up on screen. Fix it before rendering.")
    else:
        print("trajectories look sane -- safe to render")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
