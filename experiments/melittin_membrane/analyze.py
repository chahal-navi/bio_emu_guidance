"""Analyze melittin ensembles produced by run.py.

    python experiments/melittin_membrane/analyze.py [--results DIR]

Writes to <results>/analysis/: summary.csv, summary.md, per_residue.csv, and figures.

Metrics (per sample; summarized over physically valid samples unless noted):
  ez_kcal        Ez membrane energy at the optimal membrane placement (the guidance objective)
  helicity       fraction of residues in DSSP H/G/I (mdtraj, simplified 'H')
  rg_A           Cα radius of gyration
  e2e_A          Cα end-to-end distance (res 1 -> 26)
  kink_deg       angle between helix axes of residues 2-12 and 16-25 (Pro14 hinge; 0 = straight)
  depth_hyd_A    mean |z| of hydrophobic residues (A V L I F M W) in the optimal frame
  depth_chg_A    mean |z| of K/R in the optimal frame
  physical       passes BioEmu's chain-break / clash filter (reported over all samples)
  unique         fraction of distinct structures (SMC/FKC resampling duplicates particles)
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mdtraj  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from bioemu.convert_chemgraph import _get_physical_traj_indices  # noqa: E402
from bioemu.steering.custom_potentials import EzMembranePotential  # noqa: E402

HERE = Path(__file__).resolve().parent
HYDROPHOBIC = set("AVLIFMW")
CHARGED = set("KR")

# Fixed categorical order (color follows the method, not its rank). Validated default palette.
METHOD_COLORS = {
    "baseline": "#2a78d6",          # blue
    "dps_lam1.69": "#eb6834",       # orange
    "dps_lam5": "#1baf7a",          # aqua
    "dps_phys_lam1.69": "#eda100",  # yellow
    "dps_phys_lam5": "#e87ba4",     # magenta
    "smc": "#008300",               # green
    "fkc": "#6250d6",               # violet
    "dps_lam0.5": "#e34948",        # red (only when the 0.5 arm is run)
}
EXTRA_COLORS = ["#008300", "#e34948"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"


def color_for(name: str, i: int) -> str:
    return METHOD_COLORS.get(name, EXTRA_COLORS[i % len(EXTRA_COLORS)])


def load_run(run_dir: Path):
    files = sorted(run_dir.glob("batch_*.npz"))
    pos = np.concatenate([np.load(f)["pos"] for f in files])  # (S, L, 3) nm
    sequence = str(np.load(files[0])["sequence"])
    traj = mdtraj.load_xtc(str(run_dir / "samples.xtc"), top=str(run_dir / "topology.pdb"))
    assert traj.n_frames == len(pos), (traj.n_frames, len(pos))
    return pos, sequence, traj


def helix_axis(ca: np.ndarray) -> np.ndarray:
    c = ca - ca.mean(0)
    axis = np.linalg.svd(c, full_matrices=False)[2][0]
    return axis if np.dot(ca[-1] - ca[0], axis) > 0 else -axis


def analyze_run(run_dir: Path, pot: EzMembranePotential, device: str) -> tuple[pd.DataFrame, dict]:
    pos, seq, traj = load_run(run_dir)
    S, L, _ = pos.shape

    # Ez at optimal membrane placement
    energies, depths = [], []
    for s in range(0, S, 256):
        out = pot.evaluate(torch.tensor(pos[s : s + 256], device=device), sequence=seq)
        energies.append(out["energy"].cpu().numpy())
        depths.append(out["depth_A"].cpu().numpy())
    ez = np.concatenate(energies)
    depth = np.abs(np.concatenate(depths))  # (S, L)

    dssp = mdtraj.compute_dssp(traj, simplified=True)  # (S, L)
    helix = dssp == "H"

    ca = pos * 10.0
    rg = np.sqrt(((ca - ca.mean(1, keepdims=True)) ** 2).sum(-1).mean(-1))
    e2e = np.linalg.norm(ca[:, -1] - ca[:, 0], axis=-1)
    if L >= 25:  # melittin numbering: helix 2-12 | Pro14 | helix 16-25
        kink = np.array(
            [
                np.degrees(
                    np.arccos(np.clip(np.dot(helix_axis(x[1:12]), helix_axis(x[15:25])), -1, 1))
                )
                for x in ca
            ]
        )
    else:
        kink = np.full(S, np.nan)
    hyd = np.array([a in HYDROPHOBIC for a in seq])
    chg = np.array([a in CHARGED for a in seq])

    physical = np.zeros(S, dtype=bool)
    physical[_get_physical_traj_indices(traj)] = True
    unique = len(np.unique(np.round(ca, 2).reshape(S, -1), axis=0)) / S

    df = pd.DataFrame(
        {
            "ez_kcal": ez,
            "helicity": helix.mean(1),
            "rg_A": rg,
            "e2e_A": e2e,
            "kink_deg": kink,
            "depth_hyd_A": depth[:, hyd].mean(1),
            "depth_chg_A": depth[:, chg].mean(1),
            "physical": physical,
        }
    )
    per_res = {
        "helix_frac": helix[physical].mean(0) if physical.any() else helix.mean(0),
        "depth_A": depth[physical].mean(0) if physical.any() else depth.mean(0),
    }
    info_path = run_dir / "run_info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    meta = {"sequence": seq, "unique": unique, "wall_time_s": info.get("wall_time_s"), **per_res}
    return df, meta


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=str(HERE / "results"))
    args = p.parse_args()
    results = Path(args.results)
    out = results / "analysis"
    out.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pot = EzMembranePotential(frame="optimize")

    order = list(METHOD_COLORS)
    run_dirs = [d for d in results.iterdir() if d.is_dir() and list(d.glob("batch_*.npz"))]
    run_dirs.sort(key=lambda d: (order.index(d.name) if d.name in order else 99, d.name))

    dfs, metas, rows = {}, {}, []
    for d in run_dirs:
        print(f"analyzing {d.name}", flush=True)
        df, meta = analyze_run(d, pot, device)
        dfs[d.name], metas[d.name] = df, meta
        phys = df[df.physical]
        row = {"method": d.name, "n": len(df), "physical_frac": df.physical.mean(),
               "unique_frac": meta["unique"]}  # fmt: skip
        for col in ["ez_kcal", "helicity", "rg_A", "e2e_A", "kink_deg", "depth_hyd_A", "depth_chg_A"]:
            row[f"{col}_mean"] = phys[col].mean()
            row[f"{col}_std"] = phys[col].std()
        row["ez_all_mean"] = df.ez_kcal.mean()
        row["wall_time_s"] = meta["wall_time_s"]
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False)
    seq = next(iter(metas.values()))["sequence"]
    per_res = pd.DataFrame({"resid": np.arange(1, len(seq) + 1), "aa": list(seq)})
    for name, m in metas.items():
        per_res[f"{name}_helix"] = m["helix_frac"]
        per_res[f"{name}_depth_A"] = m["depth_A"]
    per_res.to_csv(out / "per_residue.csv", index=False)

    def fmt(r, c, nd=2):
        return f"{r[c + '_mean']:.{nd}f} ± {r[c + '_std']:.{nd}f}"

    lines = [
        "| method | n | physical | unique | Ez (kcal/mol) | helicity | Rg (Å) | kink (°) | "
        "depth hydrophobic (Å) | depth K/R (Å) | Ez all samples | time (s) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in summary.iterrows():
        t = f"{r.wall_time_s:.0f}" if pd.notna(r.wall_time_s) else "–"
        lines.append(
            f"| {r.method} | {r.n} | {r.physical_frac:.2f} | {r.unique_frac:.2f} | {fmt(r, 'ez_kcal')} | "
            f"{fmt(r, 'helicity')} | {fmt(r, 'rg_A', 1)} | {fmt(r, 'kink_deg', 0)} | "
            f"{fmt(r, 'depth_hyd_A', 1)} | {fmt(r, 'depth_chg_A', 1)} | {r.ez_all_mean:.2f} | {t} |"
        )
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    names = list(dfs)
    colors = {n: color_for(n, i) for i, n in enumerate(names)}

    # 1. Ez distribution per method
    fig, ax = plt.subplots(figsize=(7, 4))
    data = [dfs[n][dfs[n].physical].ez_kcal.values for n in names]
    bp = ax.boxplot(data, vert=False, widths=0.5, patch_artist=True, showfliers=False)
    for patch, n in zip(bp["boxes"], names):
        patch.set(facecolor=colors[n], edgecolor="white", linewidth=2, alpha=0.9)
    for el in ("whiskers", "caps", "medians"):
        for line in bp[el]:
            line.set(color=INK if el == "medians" else MUTED, linewidth=1.5 if el == "medians" else 1)
    ax.set_yticks(range(1, len(names) + 1), names)
    style(ax, "Ez membrane energy at optimal placement (lower = more membrane-compatible)",
          "Ez (kcal/mol)", "")  # fmt: skip
    fig.tight_layout()
    fig.savefig(out / "ez_energy.png", dpi=160)
    plt.close(fig)

    # 2. Per-residue helicity and depth
    for key, ylabel, fname, title in [
        ("helix_frac", "fraction helical (DSSP)", "per_residue_helicity.png", "Per-residue helicity"),
        ("depth_A", "mean |z| from midplane (Å)", "per_residue_depth.png",
         "Per-residue depth in optimal membrane frame"),
    ]:  # fmt: skip
        fig, ax = plt.subplots(figsize=(9, 4))
        x = np.arange(1, len(seq) + 1)
        for n in names:
            ax.plot(x, metas[n][key], color=colors[n], linewidth=2, marker="o", markersize=4, label=n)
        ax.set_xticks(x, [f"{a}\n{i}" for i, a in zip(x, seq)], fontsize=7)
        if key == "depth_A":
            ax.axhline(15.0, color=MUTED, linestyle="--", linewidth=1)
            ax.text(len(seq) + 0.3, 15.0, "core edge", color=MUTED, fontsize=8, va="center")
        style(ax, title, "residue", ylabel)
        ax.legend(frameon=False, fontsize=8, ncol=min(3, len(names)))
        fig.tight_layout()
        fig.savefig(out / fname, dpi=160)
        plt.close(fig)

    # 3. Ez vs helicity (small multiples, one per method, shared axes)
    fig, axes = plt.subplots(1, len(names), figsize=(3 * len(names), 3), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, n in zip(axes, names):
        d = dfs[n][dfs[n].physical]
        ax.scatter(d.helicity, d.ez_kcal, s=10, color=colors[n], alpha=0.5, linewidths=0)
        style(ax, n, "helicity", "Ez (kcal/mol)" if ax is axes[0] else "")
    fig.tight_layout()
    fig.savefig(out / "ez_vs_helicity.png", dpi=160)
    plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
