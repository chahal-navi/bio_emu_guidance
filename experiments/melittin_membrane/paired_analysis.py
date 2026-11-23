"""Paired comparison of guided runs against their baseline.

    python experiments/melittin_membrane/paired_analysis.py

Why paired. Every run in a sweep shares `base_seed`, and `generate_batch` seeds each batch
as base_seed + start_idx, so sample j of the guided run and sample j of the baseline start
from the SAME prior draw and differ only by the guidance applied along the trajectory.
They are matched pairs, not independent samples.

This matters enormously here. Between-structure spread in Ez is about 1.1 kcal/mol while
the guidance effect is about 0.17, so an unpaired test is swamped by variance that pairing
removes outright: at lambda=20 the unpaired p is 0.51 and the paired p is 4e-08, on the
same numbers. Every unpaired result reported before this script was underpowered.

Scores use Ez under the OPM reference frame - the quantity guidance actually minimises.
Scoring guided runs with a SEARCHED frame instead lets the metric credit improvements that
came from moving the membrane, which is the artefact that made the early sweeps look
better than they were.

Only pairs where BOTH structures pass the physicality filter are compared, since a broken
structure's score is not meaningful.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import mdtraj
import numpy as np
import torch
from scipy import stats

from bioemu.convert_chemgraph import _get_physical_traj_indices
from bioemu.steering.membrane_objectives import (
    OBJECTIVES_BY_KEY,
    load_reference_ca,
    reference_frame,
)

HERE = Path(__file__).resolve().parent
MELITTIN = "GIGAVLKVLTTGLPALISWIKRKRQQ"
DEFAULT_REF = HERE / "reference" / "opm" / "2mlt_opm.pdb"


def score_run(run_dir: Path, ref_ca, objective) -> tuple[np.ndarray, np.ndarray]:
    files = sorted(glob.glob(f"{run_dir}/batch_*.npz"))
    pos = np.concatenate([np.load(f)["pos"] for f in files])
    tr = mdtraj.load_xtc(str(run_dir / "samples.xtc"), top=str(run_dir / "topology.pdb"))
    physical = np.zeros(len(pos), dtype=bool)
    physical[_get_physical_traj_indices(tr)] = True
    x = torch.tensor(pos, dtype=torch.float32)
    n, c = reference_frame(x * 10.0, ref_ca)
    e, _ = objective(x, sequence=MELITTIN, normal=n, center=c)
    return physical, e.detach().numpy()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True, help="a results directory containing baseline/")
    p.add_argument("--reference", default=str(DEFAULT_REF))
    p.add_argument("--objective", default="ez_ca")
    args = p.parse_args()

    res = Path(args.results)
    ref = load_reference_ca(args.reference)
    obj = OBJECTIVES_BY_KEY[args.objective]

    b_phys, b_e = score_run(res / "baseline", ref, obj)
    print(f"{res.name} | objective {args.objective} | OPM frame | baseline n={len(b_e)}")
    print(f"baseline Ez {b_e[b_phys].mean():+.3f} +/- {b_e[b_phys].std():.3f} "
          f"(physical {b_phys.mean():.2f})\n")
    print(f"{'run':28} {'pairs':>6} {'mean delta':>11} {'won':>6} {'p paired':>10} {'p unpaired':>11}")
    for run in sorted(d for d in res.iterdir() if d.is_dir() and d.name not in ("baseline", "analysis")):
        g_phys, g_e = score_run(run, ref, obj)
        both = b_phys & g_phys
        if both.sum() < 5:
            print(f"{run.name:28} {both.sum():6d}   too few physical pairs to compare")
            continue
        d = g_e[both] - b_e[both]
        _, p_paired = stats.ttest_rel(g_e[both], b_e[both])
        _, p_unpaired = stats.ttest_ind(g_e[g_phys], b_e[b_phys], equal_var=False)
        print(f"{run.name:28} {both.sum():6d} {d.mean():+11.3f} {(d < 0).mean() * 100:5.0f}% "
              f"{p_paired:10.2g} {p_unpaired:11.3f}")
    print("\nnegative delta = guidance lowered the energy. 'won' = share of pairs improved.")


if __name__ == "__main__":
    main()
