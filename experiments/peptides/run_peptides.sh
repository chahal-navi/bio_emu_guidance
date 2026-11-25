#!/usr/bin/env bash
# Guidance across several membrane peptides, at the operating point established on
# melittin (Ez on Ca, OPM reference frame, lambda=20).
#
# Melittin was a weak test case: BioEmu's aqueous ensemble already sits near the limit a
# connected helix can reach, so guidance had almost nothing to fix. These peptides were
# chosen for HEADROOM - a large expected gap between the aqueous ensemble BioEmu samples
# and the membrane-bound state.
#
# pardaxin is the important one. OPM holds two entries with the IDENTICAL 33-residue
# sequence in two different binding modes (2kns surface, 1xc0 transmembrane), so one
# aqueous ensemble can be guided toward two different membrane targets. That isolates the
# guidance effect from sequence effects - the control melittin could not provide.
set -euo pipefail
REPO=${REPO:-/workspace/bio_emu_guidance}
REF=$REPO/experiments/peptides/reference
OUT=${OUT:-/workspace/results_peptides}
N=${N:-50}
LAM=${LAM:-20}
cd "$REPO"

run_one () {   # name  sequence  reference_pdb
  local name=$1 seq=$2 ref=$3
  echo "=============================================================="
  echo "=== $name  (${#seq} residues)  ref=$(basename "$ref")"
  echo "=============================================================="
  test -f "$ref" || { echo "MISSING reference $ref"; return 1; }
  export BIOEMU_OPM_REF="$ref"
  python3 experiments/melittin_membrane/run.py \
      --sequence "$seq" --methods baseline dps_ez_opm --dps-scales "$LAM" \
      --num-samples "$N" --out "$OUT/$name"
  echo "--- paired analysis: $name"
  python3 -W ignore experiments/melittin_membrane/paired_analysis.py \
      --results "$OUT/$name" --reference "$ref" --sequence "$seq" || true
}

PARDAXIN=GFFALIPKIISSPLFKTLLSAVGSALSSSGGQE

# transmembrane and surface targets for the SAME sequence
run_one pardaxin_tm      "$PARDAXIN" "$REF/pardaxin/1xc0_opm.pdb"
run_one pardaxin_surface "$PARDAXIN" "$REF/pardaxin/2kns_opm.pdb"
# extreme transmembrane stress test
run_one mgtr      MNRSPDKIIALIFLLISLLVLCLALWQIVF "$REF/mgtr/2mc7_opm.pdb"
# surface-bound, but genuinely coil in water - the direct melittin contrast
run_one magainin2 GIGKFLHSAKKFGKAFVGEIMNS        "$REF/magainin2/2mag_opm.pdb"

echo "=== ALL DONE"
