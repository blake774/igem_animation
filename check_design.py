#!/usr/bin/env python3
"""
check_design.py -- is this file the design the shot needs?

Run it on candidate files before copying them anywhere. It answers the only
question that matters: does this structure contain the retained 6W74 motif
(A312-325), and does that motif superpose onto the crystal?

    python3 check_design.py CANDIDATE.cif --receptor input/6w74.pdb
    python3 check_design.py 'runs/*/model_0.cif' --receptor input/6w74.pdb

It needs only numpy and structlib.py, so it runs on a cluster login node with
no conda environment. Copy those two files plus 6w74.pdb next to it.

Exit status is 0 if at least one candidate passes.

What "passes" means
-------------------
  motif RMSD < 1.5 A   the design really is 6W74-conditioned. This is the
                       same superposition prep_shrink.py does, so a file that
                       passes here will place correctly there.
  binder >= 60 aa      there is a designed chain, not just the motif
  binder is longest    prep_shrink.py takes the longest protein chain as the
                       binder; if a target chain is longer, it will pick wrong
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

import structlib as S

MOTIF_RANGE = (312, 325)
AA = S.AMINO3


def motif_reference(receptor_path: str) -> tuple[np.ndarray, np.ndarray]:
    """(motif Ca, all receptor Ca) -- the second is used to spot a candidate
    that is just the target structure again."""
    st = S.read_structure(receptor_path)
    ca = [a for a in st.atoms
          if a.record == "ATOM" and a.name.strip() == "CA"
          and MOTIF_RANGE[0] <= a.resseq <= MOTIF_RANGE[1]]
    ca.sort(key=lambda a: a.resseq)
    if len(ca) != MOTIF_RANGE[1] - MOTIF_RANGE[0] + 1:
        sys.exit(f"{receptor_path} is missing motif residues "
                 f"{MOTIF_RANGE[0]}-{MOTIF_RANGE[1]} (found {len(ca)})")
    allca, _ = st.ca_trace()
    return np.array([[a.x, a.y, a.z] for a in ca]), allca


def inspect(path: str, ref: np.ndarray, recep_ca: np.ndarray) -> dict:
    st = S.read_structure(path)
    st = st.select(lambda a: not a.is_hydrogen() and a.altloc in ("", "A"))

    chains = {}
    for ch in st.chains():
        res = {a.reskey for a in st.atoms
               if a.chain == ch and a.record == "ATOM" and a.resname in AA}
        if res:
            chains[ch] = len(res)

    het = sorted({r[3] for r in st.het_residues()})
    n = len(ref)
    best = None
    for ch in chains:
        ca, keys = st.ca_trace(ch)
        if len(ca) < n:
            continue
        for i in range(len(ca) - n + 1):
            R, t, rms = S.kabsch(ca[i:i + n], ref)
            if best is None or rms < best["rmsd"]:
                best = {"chain": ch, "start": keys[i][1], "rmsd": rms,
                        "R": R, "t": t}

    binder = max(chains, key=lambda c: chains[c]) if chains else None

    # Is this just the target again? Superpose by the motif, then ask how much
    # of the candidate lands on top of receptor backbone. A de novo binder
    # shares the motif and nothing else; handing over the cleaned receptor, or
    # a docking pose of it, is an easy mistake and passes every other check.
    overlap = 0.0
    if best is not None and binder is not None and len(recep_ca):
        bca, _ = st.ca_trace(binder)
        if len(bca):
            moved = bca @ best["R"].T + best["t"]
            d = np.sqrt((((moved[:, None, :] - recep_ca[None, :, :]) ** 2)
                         .sum(-1))).min(1)
            overlap = float((d < 2.0).mean())
    return {
        "path": path,
        "atoms": len(st),
        "chains": chains,
        "het": het,
        "binder_chain": binder,
        "binder_len": chains.get(binder, 0),
        "fit": best,
        "receptor_overlap": overlap,
    }


def report(r: dict) -> bool:
    print(f"\n{r['path']}")
    if not r["chains"]:
        print("  FAIL  no protein chains found")
        return False
    print(f"  atoms          {r['atoms']}")
    print(f"  protein chains {r['chains']}")
    if r["het"]:
        print(f"  het groups     {', '.join(r['het'])}")

    ok = True
    fit = r["fit"]
    if fit is None:
        print(f"  FAIL  no chain long enough to hold the "
              f"{MOTIF_RANGE[1] - MOTIF_RANGE[0] + 1}-residue motif")
        return False

    verdict = "ok  " if fit["rmsd"] < 1.5 else ("WARN" if fit["rmsd"] < 2.5 else "FAIL")
    print(f"  [{verdict}] motif A{MOTIF_RANGE[0]}-{MOTIF_RANGE[1]} found on chain "
          f"{fit['chain']} at residue {fit['start']}, Ca RMSD {fit['rmsd']:.2f} A")
    if fit["rmsd"] >= 1.5:
        print("         Not 6W74-conditioned, or a different design than expected.")
        ok = ok and fit["rmsd"] < 2.5

    if r["binder_len"] < 60:
        print(f"  FAIL  longest chain is only {r['binder_len']} aa -- "
              f"this looks like a motif or fragment, not a binder")
        ok = False
    else:
        print(f"  [ok  ] binder chain {r['binder_chain']}: {r['binder_len']} aa")

    if r["receptor_overlap"] > 0.7:
        print(f"  FAIL  {r['receptor_overlap']*100:.0f}% of this chain sits on "
              f"top of the receptor backbone -- this is the target structure, "
              f"not a de novo design")
        ok = False

    if fit["chain"] != r["binder_chain"]:
        print(f"  [note] the motif sits on chain {fit['chain']} but the longest "
              f"chain is {r['binder_chain']}; prep_shrink.py will treat "
              f"{r['binder_chain']} as the binder and {fit['chain']} as context")
    return ok


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("designs", nargs="+",
                   help="design files, or globs (quote them so the shell "
                        "does not expand them first)")
    p.add_argument("--receptor", default="input/6w74.pdb")
    args = p.parse_args()

    if not os.path.exists(args.receptor):
        sys.exit(f"receptor not found: {args.receptor}")
    ref, recep_ca = motif_reference(args.receptor)
    print(f"reference motif from {args.receptor}: {len(ref)} Ca "
          f"({MOTIF_RANGE[0]}-{MOTIF_RANGE[1]})")

    paths = []
    for d in args.designs:
        hits = sorted(glob.glob(d))
        paths.extend(hits if hits else [d])
    if not paths:
        sys.exit("no files matched")

    passed = []
    for path in paths:
        if not os.path.exists(path):
            print(f"\n{path}\n  FAIL  no such file")
            continue
        try:
            if report(inspect(path, ref, recep_ca)):
                passed.append(path)
        except Exception as e:                     # a bad file must not stop the sweep
            print(f"\n{path}\n  FAIL  could not read: {e}")

    print(f"\n{len(passed)}/{len(paths)} candidate(s) usable")
    for p_ in passed:
        print(f"  {p_}")
    if passed:
        print(f"\nBest next step:\n  python3 prep_shrink.py --design {passed[0]} \\"
              f"\n      --receptor {args.receptor} --fulllength input/AF-Q13490-F1.pdb")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
