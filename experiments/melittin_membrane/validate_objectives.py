"""Validate candidate membrane guidance objectives BEFORE using any of them to guide.

    python experiments/melittin_membrane/validate_objectives.py --out <dir>

An objective is only usable for guidance if its minimum is reachable by a connected,
folded chain. The Ez potential we originally guided with was not: it rewarded both
unfolding (r = +0.37 with helicity) and chain-breaking. These tests are designed to catch
exactly that class of failure, cheaply, on structures we already have.

Tests
-----
T1 structure_corr  Pearson r between objective and DSSP helicity over the generated
                   ensemble. For an amphipathic peptide like melittin the membrane-bound
                   state is helical, so a usable objective must score helical structures
                   BETTER: r < 0 required (energy is lower-is-better).
T2 stretch_gaming  Scale every residue's depth along the membrane normal by s in [1, 3].
                   This separates residues exactly the way the Ez gradient wanted to, and
                   breaks the chain doing it. A usable objective must get WORSE.
T3 expel_gaming    Translate the whole peptide away from the bilayer. A usable objective
                   must not improve (Ez did: all terms decay to zero).
T4 ref_rank        Percentile rank of an experimental membrane-bound structure within the
                   generated ensemble. A usable objective should place it near the top.
                   Skipped if no reference structure is available.
T5 headroom        Spread of the objective over the ensemble, and the gap between the
                   ensemble mean and the best single structure, in units of the ensemble
                   standard deviation. Near-zero headroom means guidance has nothing to do.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mdtraj
import numpy as np
import torch
from scipy import stats

from bioemu.steering.membrane_objectives import OBJECTIVES

HERE = Path(__file__).resolve().parent
MELITTIN = "GIGAVLKVLTTGLPALISWIKRKRQQ"


def load_ensemble(results_dirs: list[Path]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Pool every generated structure we have. Returns (ca_nm, helicity, labels)."""
    all_ca, all_hel, labels = [], [], []
    for res in results_dirs:
        for run in sorted(p for p in res.iterdir() if p.is_dir() and p.name != "analysis"):
            xtc, top = run / "samples.xtc", run / "topology.pdb"
            if not (xtc.exists() and top.exists()):
                continue
            tr = mdtraj.load_xtc(str(xtc), top=str(top))
            ca_idx = np.array(
                [[a.index for a in r.atoms if a.name == "CA"][0] for r in tr.topology.residues]
            )
            if len(ca_idx) != len(MELITTIN):
                continue
            all_ca.append(tr.xyz[:, ca_idx, :])
            all_hel.append((mdtraj.compute_dssp(tr, simplified=True) == "H").mean(1))
            labels += [f"{res.name}/{run.name}"] * tr.n_frames
    return np.concatenate(all_ca), np.concatenate(all_hel), labels


def load_reference(ref_dir: Path) -> tuple[np.ndarray | None, str]:
    """Best available experimental melittin Ca trace, (1, n_res, 3) in nm."""
    if not ref_dir.exists():
        return None, "no reference directory"
    for pdb in sorted(ref_dir.glob("*.pdb")):
        try:
            tr = mdtraj.load_pdb(str(pdb))
        except Exception:
            continue
        for chain in tr.topology.chains:
            ca = [a.index for a in chain.atoms if a.name == "CA"]
            if len(ca) == len(MELITTIN):
                return tr.xyz[:1, np.array(ca), :], f"{pdb.name} chain {chain.index}"
    return None, "no chain with 26 CA atoms found"


def score(obj, ca_nm: np.ndarray, chunk: int = 64) -> np.ndarray:
    out = []
    for s in range(0, len(ca_nm), chunk):
        e, _ = obj(torch.tensor(ca_nm[s : s + chunk], dtype=torch.float32), sequence=MELITTIN)
        out.append(e.detach().numpy())
    return np.concatenate(out)


def frame_of(obj, ca_nm: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    _, info = obj(torch.tensor(ca_nm, dtype=torch.float32), sequence=MELITTIN)
    if "normal" not in info:
        return None
    n = info["normal"].detach().numpy()
    c = info.get("center_A")
    c = c.detach().numpy() if c is not None else np.zeros(len(n), dtype=np.float32)
    return n, c


def t2_stretch(obj, ca_nm: np.ndarray, factors=(1.0, 1.5, 2.0, 3.0)) -> dict:
    """Scale depth along the membrane normal. Chain-breaking by construction."""
    fr = frame_of(obj, ca_nm)
    if fr is None:  # frame-free objective: use its own moment direction as the axis
        _, info = obj(torch.tensor(ca_nm, dtype=torch.float32), sequence=MELITTIN)
        mu = info["moment_vec"].detach().numpy()
        n = mu / np.linalg.norm(mu, axis=-1, keepdims=True)
        c = np.einsum("bld,bd->bl", ca_nm * 10.0, n).mean(-1)
    else:
        n, c = fr
    ca_A = ca_nm * 10.0
    z = np.einsum("bld,bd->bl", ca_A, n) - c[:, None]
    out = {}
    for s in factors:
        moved = ca_A + ((s - 1.0) * z)[..., None] * n[:, None, :]
        out[s] = float(score(obj, moved / 10.0).mean())
    return out


def t3_expel(obj, ca_nm: np.ndarray, shifts_A=(0.0, 20.0, 60.0, 200.0)) -> dict:
    fr = frame_of(obj, ca_nm)
    if fr is None:
        return {}
    n, _ = fr
    ca_A = ca_nm * 10.0
    return {
        d: float(score(obj, (ca_A + d * n[:, None, :]) / 10.0).mean()) for d in shifts_A
    }


def load_opm(ref_dir: Path) -> tuple[np.ndarray | None, str]:
    """OPM-oriented melittin: already in the membrane frame (normal = +z, midplane z = 0)."""
    f = ref_dir / "opm" / "2mlt_opm.pdb"
    if not f.exists():
        return None, "no OPM file"
    tr = mdtraj.load_pdb(str(f))
    ca = np.array([a.index for a in tr.topology.atoms if a.name == "CA"])
    if len(ca) != len(MELITTIN):
        return None, f"OPM file has {len(ca)} CA atoms, expected {len(MELITTIN)}"
    return tr.xyz[:1, ca, :], f.name


def t6_frame_recovery(obj, opm_ca_nm: np.ndarray) -> dict:
    """Given a structure ALREADY in the known membrane frame, does the objective find it?

    Ground truth from OPM: membrane normal = z axis, bilayer midplane at z = 0. A usable
    objective should recover both. Reported as the angle between the predicted normal and
    z (0 deg = perfect; the normal's sign is arbitrary so we fold to [0, 90]) and the
    absolute error in the midplane position.
    """
    fr = frame_of(obj, opm_ca_nm)
    if fr is None:
        return {}
    n, c = fr
    cos = abs(float(n[0][2]))
    return {
        "normal_err_deg": float(np.degrees(np.arccos(np.clip(cos, 0, 1)))),
        "center_err_A": float(abs(c[0])),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", nargs="+",
                   default=[str(HERE / "results_lam"), str(HERE / "results_pilot2")])  # fmt: skip
    p.add_argument("--reference", default=str(HERE / "reference"))
    p.add_argument("--out", default=str(HERE / "objective_validation"))
    p.add_argument("--subsample", type=int, default=0, help="use only N structures (0 = all)")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dirs = [Path(d) for d in args.results if Path(d).exists()]
    ca, hel, labels = load_ensemble(dirs)
    if args.subsample:
        idx = np.linspace(0, len(ca) - 1, args.subsample).astype(int)
        ca, hel = ca[idx], hel[idx]
        labels = [labels[i] for i in idx]
    print(f"ensemble: {len(ca)} structures, helicity {hel.min():.2f}-{hel.max():.2f}")

    ref_ca, ref_desc = load_reference(Path(args.reference))
    print(f"reference: {ref_desc}")
    opm_ca, opm_desc = load_opm(Path(args.reference))
    print(f"OPM membrane-frame reference: {opm_desc}")

    results = []
    for obj in OBJECTIVES:
        print(f"  scoring {obj.key} ...", flush=True)
        e = score(obj, ca)
        r, p_val = stats.pearsonr(e, hel)
        rec = {
            "key": obj.key, "name": obj.name, "rationale": obj.rationale,
            "couples_residues": obj.couples_residues,
            "mean": float(e.mean()), "std": float(e.std()),
            "best": float(e.min()), "worst": float(e.max()),
            "t1_r_helicity": float(r), "t1_p": float(p_val),
            "t2_stretch": t2_stretch(obj, ca[:128]),
            "t3_expel": t3_expel(obj, ca[:128]),
            "t5_headroom_sd": float((e.mean() - e.min()) / (e.std() + 1e-12)),
        }  # fmt: skip
        if opm_ca is not None:
            rec["t6_frame"] = t6_frame_recovery(obj, opm_ca)
        if ref_ca is not None:
            e_ref = float(score(obj, ref_ca)[0])
            rec["t4_ref_score"] = e_ref
            rec["t4_ref_percentile"] = float((e > e_ref).mean() * 100)
        results.append(rec)

    payload = {
        "n_structures": int(len(ca)),
        "helicity_range": [float(hel.min()), float(hel.max())],
        "reference": ref_desc,
        "opm_reference": opm_desc,
        "sequence": MELITTIN,
        "objectives": results,
    }
    (out / "validation.json").write_text(json.dumps(payload, indent=2))
    np.savez(out / "scores.npz", helicity=hel,
             **{o["key"]: score(OBJECTIVES[i], ca) for i, o in enumerate(results)})  # fmt: skip

    print(f"\n{'objective':24} {'r(hel)':>7} {'stretch':>8} {'OPM n err':>10} {'verdict':>8}")
    for rec in results:
        st = rec["t2_stretch"]
        gamed = bool(st) and st.get(3.0, 0) < st.get(1.0, 0)
        ok = rec["t1_r_helicity"] < 0 and not gamed
        rec["verdict"] = "PASS" if ok else "FAIL"
        ne = (rec.get("t6_frame") or {}).get("normal_err_deg")
        print(f"{rec['key']:24} {rec['t1_r_helicity']:+7.3f} "
              f"{'GAMED' if gamed else 'safe':>8} "
              f"{(f'{ne:.0f} deg' if ne is not None else 'n/a'):>10} {rec['verdict']:>8}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
