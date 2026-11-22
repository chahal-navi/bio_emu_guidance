"""A/B the per-structure optimised membrane frame against the OPM reference frame.

    python experiments/melittin_membrane/compare_frames.py

The frame search was measured to recover membrane normals 66-83 degrees from the
experimentally determined one (see objective_functions_report.html, T6). This script asks
whether transferring the frame from the OPM-oriented melittin structure instead fixes the
objectives - and, separately, what the two frames imply about the generated ensemble.

It reports per objective, under each frame:
  r(helicity)   must be < 0 (membrane-bound melittin is helical)
  stretch       score change when the chain is pulled 3x along the normal, in ensemble
                sigma; negative means stretching improves the score, i.e. gameable
  tilt          helix-axis tilt from the membrane normal, mean +/- sd. OPM's experimental
                value for melittin is 65 degrees.
"""

from __future__ import annotations

import json
from pathlib import Path

import mdtraj
import numpy as np
import torch
from scipy import stats

from bioemu.steering.membrane_objectives import (
    OBJECTIVES,
    load_reference_ca,
    reference_frame,
)

HERE = Path(__file__).resolve().parent
MELITTIN = "GIGAVLKVLTTGLPALISWIKRKRQQ"
OPM = HERE / "reference" / "opm" / "2mlt_opm.pdb"
OPM_TILT_DEG = 65.0  # OPM entry 149, experimental reference value


def load_ensemble(dirs):
    ca, hel = [], []
    for res in dirs:
        for run in sorted(p for p in res.iterdir() if p.is_dir() and p.name != "analysis"):
            xtc, top = run / "samples.xtc", run / "topology.pdb"
            if not (xtc.exists() and top.exists()):
                continue
            tr = mdtraj.load_xtc(str(xtc), top=str(top))
            idx = np.array([[a.index for a in r.atoms if a.name == "CA"][0]
                            for r in tr.topology.residues])  # fmt: skip
            if len(idx) != len(MELITTIN):
                continue
            ca.append(tr.xyz[:, idx, :])
            hel.append((mdtraj.compute_dssp(tr, simplified=True) == "H").mean(1))
    return np.concatenate(ca), np.concatenate(hel)


def helix_axis(ca):
    c = ca - ca.mean(0)
    ax = np.linalg.svd(c, full_matrices=False)[2][0]
    return ax if np.dot(ca[-1] - ca[0], ax) > 0 else -ax


def score(obj, ca_nm, ref_ca=None, chunk=64):
    """Score an ensemble; if ref_ca is given, supply the OPM-transferred frame."""
    out, normals, centers = [], [], []
    for s in range(0, len(ca_nm), chunk):
        x = torch.tensor(ca_nm[s : s + chunk], dtype=torch.float32)
        if ref_ca is None:
            e, info = obj(x, sequence=MELITTIN)
        else:
            n, c = reference_frame(x * 10.0, ref_ca)
            e, info = obj(x, sequence=MELITTIN, normal=n, center=c)
        out.append(e.detach().numpy())
        if "normal" in info:
            normals.append(info["normal"].detach().numpy())
            cc = info.get("center_A")
            centers.append(cc.detach().numpy() if cc is not None
                           else np.zeros(len(info["normal"])))  # fmt: skip
    n = np.concatenate(normals) if normals else None
    c = np.concatenate(centers) if centers else None
    return np.concatenate(out), n, c


def stretch_delta(obj, ca_nm, ref_ca, normals, centers):
    """Score change on a 3x depth stretch, in ensemble sigma."""
    if normals is None:
        return None
    ca_A = ca_nm * 10.0
    z = np.einsum("bld,bd->bl", ca_A, normals) - centers[:, None]
    moved = ca_A + (2.0 * z)[..., None] * normals[:, None, :]
    base, _, _ = score(obj, ca_nm, ref_ca)
    out, _, _ = score(obj, moved / 10.0, ref_ca)
    sd = base.std() or 1e-9
    return float((out.mean() - base.mean()) / sd)


def main():
    dirs = [HERE / "results_lam", HERE / "results_pilot2"]
    dirs = [d for d in dirs if d.exists()]
    ca, hel = load_ensemble(dirs)
    ref_ca = load_reference_ca(str(OPM))
    print(f"ensemble {len(ca)} structures | OPM reference {OPM.name}\n")

    # Tilt of the generated ensemble under the OPM-transferred frame.
    n_ref, c_ref = reference_frame(torch.tensor(ca * 10.0, dtype=torch.float32), ref_ca)
    n_ref_np = n_ref.numpy()
    tilt_ref = np.degrees(np.arccos(np.clip(
        np.abs([np.dot(helix_axis(ca[i] * 10.0), n_ref_np[i]) for i in range(len(ca))]), 0, 1)))

    rows = []
    print(f"{'objective':22} {'frame':>10} {'r(hel)':>8} {'stretch':>9} {'tilt deg':>12}")
    for obj in OBJECTIVES:
        rec = {"key": obj.key, "name": obj.name}
        for mode, ref in (("optimised", None), ("OPM ref", ref_ca)):
            e, n, c = score(obj, ca, ref)
            r, p = stats.pearsonr(e, hel)
            ds = stretch_delta(obj, ca[:128], ref, None if n is None else n[:128],
                               None if c is None else c[:128])  # fmt: skip
            if n is not None:
                t = np.degrees(np.arccos(np.clip(
                    np.abs([np.dot(helix_axis(ca[i] * 10.0), n[i]) for i in range(len(ca))]), 0, 1)))
                tilt = f"{t.mean():.0f} +/- {t.std():.0f}"
            else:
                tilt = "n/a"
            key = "opt" if ref is None else "opm"
            rec[f"r_{key}"] = float(r)
            rec[f"stretch_{key}"] = ds
            rec[f"tilt_{key}"] = tilt
            print(f"{obj.key:22} {mode:>10} {r:+8.3f} "
                  f"{('%+.2f' % ds) if ds is not None else '     n/a':>9} {tilt:>12}")
        rows.append(rec)
        print()

    out = {
        "n_structures": int(len(ca)),
        "opm_reference": OPM.name,
        "opm_experimental_tilt_deg": OPM_TILT_DEG,
        "ensemble_tilt_under_opm_frame": [float(tilt_ref.mean()), float(tilt_ref.std())],
        "objectives": rows,
    }
    dest = HERE / "objective_validation" / "frame_comparison.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"ensemble tilt under OPM frame: {tilt_ref.mean():.0f} +/- {tilt_ref.std():.0f} deg "
          f"(OPM experimental value for melittin: {OPM_TILT_DEG:.0f} deg)")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
