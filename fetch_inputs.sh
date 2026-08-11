#!/usr/bin/env bash
#
# fetch_inputs.sh -- download and verify the two public structures the shot
# needs. Run it on any machine with normal internet (your laptop is fine).
#
#     bash fetch_inputs.sh
#
# It fetches:
#   input/6w74.pdb                cIAP1 BIR3 + PROTAC warhead, from RCSB
#   input/AF-Q13490-F1.pdb        full-length cIAP1 (618 aa), from AlphaFold DB
#
# and then checks each one is actually the structure it claims to be, because
# the usual failure here is not a network error -- it is a 200 response
# containing an HTML error page, which lands on disk as a "PDB file" that
# every downstream tool reads as zero atoms.
#
# The third input, your de novo design, is not public. See INPUTS.md.

set -uo pipefail

IN="${1:-input}"
mkdir -p "$IN"

RCSB="https://files.rcsb.org/download/6W74.pdb"
AFDB_BASE="https://alphafold.ebi.ac.uk/files/AF-Q13490-F1-model"

say()  { printf '  %s\n' "$*"; }
head_() { printf '\n[%s]\n' "$*"; }

# ---------------------------------------------------------------- downloader
get() {
    # get <url> <dest>  -- curl or wget, whichever exists
    local url="$1" dest="$2"
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL --retry 3 --retry-delay 2 --max-time 180 -o "$dest" "$url"
    elif command -v wget >/dev/null 2>&1; then
        wget -q --tries=3 --timeout=180 -O "$dest" "$url"
    else
        echo "need curl or wget" >&2
        return 1
    fi
}

# ------------------------------------------------------------------ verifier
# Counts ATOM/HETATM records and reports the residue range, so an HTML error
# page or a truncated transfer fails loudly instead of quietly.
verify() {
    local f="$1" what="$2" min_atoms="$3"
    python3 - "$f" "$what" "$min_atoms" <<'PY'
import sys
path, what, min_atoms = sys.argv[1], sys.argv[2], int(sys.argv[3])
try:
    text = open(path, errors="replace").read()
except OSError as e:
    sys.exit(f"  FAIL {what}: cannot read {path} ({e})")
if text.lstrip()[:1] == "<":
    sys.exit(f"  FAIL {what}: {path} is HTML, not a structure "
             f"(the server returned an error page)")
atoms, nums = 0, []
for line in text.splitlines():
    if line.startswith(("ATOM", "HETATM")):
        atoms += 1
        try:
            nums.append(int(line[22:26]))
        except ValueError:
            pass
if atoms < min_atoms:
    sys.exit(f"  FAIL {what}: only {atoms} atoms in {path} "
             f"(expected at least {min_atoms}) -- likely a truncated download")
lo, hi = (min(nums), max(nums)) if nums else (0, 0)
print(f"  ok   {what}: {atoms} atoms, residues {lo}-{hi}")
PY
}

# ------------------------------------------------------------------- 6W74
head_ "6W74 -- cIAP1 BIR3 + compound 15 (RCSB)"
if [ -s "$IN/6w74.pdb" ]; then
    say "already present: $IN/6w74.pdb"
else
    say "downloading $RCSB"
    get "$RCSB" "$IN/6w74.pdb" || say "download failed"
fi
verify "$IN/6w74.pdb" "6W74" 600 || FAILED_6W74=1

# Confirm it is the right structure, not just a valid one.
python3 - "$IN/6w74.pdb" <<'PY' || true
import sys
want = {312: "GLY", 313: "LEU", 314: "ARG", 325: "GLU"}
found = {}
for line in open(sys.argv[1], errors="replace"):
    if line.startswith("ATOM"):
        try:
            n = int(line[22:26])
        except ValueError:
            continue
        if n in want:
            found[n] = line[17:20].strip()
miss = [f"{n}:{want[n]}" for n in want if found.get(n) != want[n]]
if miss:
    print(f"  WARN 6W74 motif fingerprint mismatch at {', '.join(miss)} -- "
          f"is this really 6W74?")
else:
    print("  ok   motif fingerprint GLY312/LEU313/ARG314/GLU325 present")
PY

# ------------------------------------------------------- full-length cIAP1
head_ "AF-Q13490 -- full-length cIAP1 / BIRC2, 618 aa (AlphaFold DB)"
if [ -s "$IN/AF-Q13490-F1.pdb" ]; then
    say "already present: $IN/AF-Q13490-F1.pdb"
else
    # AlphaFold DB bumps the model version; try newest first.
    for v in 4 3 2; do
        say "trying ${AFDB_BASE}_v${v}.pdb"
        if get "${AFDB_BASE}_v${v}.pdb" "$IN/AF-Q13490-F1.pdb"; then
            say "got model v${v}"
            break
        fi
    done
fi
verify "$IN/AF-Q13490-F1.pdb" "AF-Q13490" 4000 || FAILED_AF=1

# ------------------------------------------------------------------- done
head_ "next"
if [ -n "${FAILED_6W74:-}" ]; then
    cat <<'EOT'
  6W74 is REQUIRED. If the download failed, open this in a browser:
      https://www.rcsb.org/structure/6W74     ("Download Files" -> "PDB Format")
  and save it as input/6w74.pdb
EOT
fi
if [ -n "${FAILED_AF:-}" ]; then
    cat <<'EOT'
  AF-Q13490 is optional but strongly recommended -- without it the opening
  shot is the 87-residue BIR3 domain instead of the 618-residue protein, and
  the shrink is not visible. In a browser:
      https://alphafold.ebi.ac.uk/entry/Q13490    ("Download" -> "PDB file")
  and save it as input/AF-Q13490-F1.pdb
EOT
fi

cat <<'EOT'

  Your de novo design is not public -- copy it off the cluster. See INPUTS.md.

  Then, with no design file yet (placeholder binder, renders today):
      python3 make_testdata.py
      python3 prep_shrink.py --design input/synthetic_design.cif \
          --receptor input/6w74.pdb --fulllength input/AF-Q13490-F1.pdb \
          --rfd-traj input/synthetic_rfd_traj.pdb

  Or with the real one:
      python3 prep_shrink.py --design YOUR_DESIGN.cif \
          --receptor input/6w74.pdb --fulllength input/AF-Q13490-F1.pdb \
          --rfd-traj YOUR_pX0_traj.pdb

  Then:
      python3 validate.py && python3 lint_cxc.py
      python3 render_blender.py --quality preview
EOT
