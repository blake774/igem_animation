#!/usr/bin/env bash
#
# make_final.sh -- build every asset and render the final 1080p movie in one
# shot. Run it once the real inputs are in input/:
#
#     input/6w74.pdb            (in the repo)
#     input/AF-Q13490-F1.pdb    (bash fetch_inputs.sh)
#     input/ciap1_design.cif    (your AF3 design, off the cluster)
#     input/ciap1_pX0_traj.pdb  (optional; omit and shot 4 is synthetic)
#
#     bash make_final.sh
#
# Override anything with an env var, e.g.  DESIGN=input/other.cif bash make_final.sh
set -euo pipefail
cd "$(dirname "$0")"

DESIGN="${DESIGN:-input/ciap1_design.cif}"
RECEPTOR="${RECEPTOR:-input/6w74.pdb}"
FULL="${FULL:-input/AF-Q13490-F1.pdb}"
TRAJ="${TRAJ:-input/ciap1_pX0_traj.pdb}"
QUALITY="${QUALITY:-hd}"
DEVICE="${DEVICE:-cpu}"   # on Apple Silicon set DEVICE=metal for a much faster render
MOVIE="${MOVIE:-output/e3_shrink_1080p.mp4}"

say(){ printf '\n=== %s ===\n' "$*"; }

[ -f "$DESIGN" ]   || { echo "missing design: $DESIGN"; exit 1; }
[ -f "$RECEPTOR" ] || { echo "missing receptor: $RECEPTOR"; exit 1; }

prep_args=( --design "$DESIGN" --receptor "$RECEPTOR" )
[ -f "$FULL" ] && prep_args+=( --fulllength "$FULL" ) \
               || echo "note: no full-length model; shot 1 falls back to BIR3"
# Note: the Blender renderer fades the binder in for shot 4 (it does not play
# the synthetic condensation trajectory), so the old --condense-sec flicker
# workaround is no longer needed here. Add it back only if you render shot 4
# through ChimeraX from a synthetic condensation.
[ -f "$TRAJ" ] && prep_args+=( --rfd-traj "$TRAJ" ) \
               || echo "note: no trajectory; shot 4 will be synthetic"

say "prep_shrink.py"
python3 prep_shrink.py "${prep_args[@]}"

say "validate.py"
python3 validate.py || echo "validate reported an issue -- read it before shipping"

say "lint_cxc.py"
python3 lint_cxc.py || echo "lint reported issues (only affects the ChimeraX path, not this render)"

say "render ($QUALITY) -> $MOVIE"
python3 render_blender.py --quality "$QUALITY" --device "$DEVICE" --movie "$MOVIE"

say "done"
echo "wrote $MOVIE"
