"""Does each objective rank the TRUE membrane placement above random ones?

    python experiments/melittin_membrane/screen_placements.py

The helicity screen used a proxy (melittin's membrane-bound state is helical, so a usable
objective should prefer helices) measured on generated structures. This test instead uses
experimental ground truth, and isolates the objective from the frame-search algorithm.

Method: take the OPM-oriented melittin structure, where the membrane placement is already
known (normal = +z, midplane z = 0). Hold the structure fixed and sweep the MEMBRANE
instead: thousands of random normals and offsets. Score every placement. A usable
objective must put the true placement near the top.

Reported per objective:
  true rank    percentile of the true placement among the random ones
               (100 = the true placement is the single best)
  gap          how much better the best random placement is than the true one, in units
               of the random-placement spread. 0 means nothing beat the truth.
  best tilt    helix-axis tilt implied by the best-scoring placement. OPM's experimental
               value for melittin is 65 degrees.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from bioemu.steering.membrane_objectives import OBJECTIVES, load_reference_ca

HERE = Path(__file__).resolve().parent
MELITTIN = "GIGAVLKVLTTGLPALISWIKRKRQQ"
OPM = HERE / "reference" / "opm" / "2mlt_opm.pdb"
OPM_TILT_DEG = 65.0
N_RANDOM = 4000
OFFSET_RANGE = 40.0  # Angstrom either side of the structure centroid


def random_placements(n: int, seed: int = 0):
    """Uniform random unit normals on the sphere, uniform offsets."""
    g = np.random.default_rng(seed)
    v = g.normal(size=(n, 3))
    normals = v / np.linalg.norm(v, axis=-1, keepdims=True)
    centers = g.uniform(-OFFSET_RANGE, OFFSET_RANGE, size=n)
    return normals.astype(np.float32), centers.astype(np.float32)


def helix_axis(ca_A: np.ndarray) -> np.ndarray:
    c = ca_A - ca_A.mean(0)
    ax = np.linalg.svd(c, full_matrices=False)[2][0]
    return ax if np.dot(ca_A[-1] - ca_A[0], ax) > 0 else -ax


def main() -> None:
    ref_A = load_reference_ca(str(OPM))            # (26,3), already in the OPM frame
    ca_nm = (ref_A / 10.0)[None]                   # (1,26,3)
    axis = helix_axis(ref_A.numpy())

    normals, centers = random_placements(N_RANDOM)
    # The true placement, in this structure's own (OPM) coordinates.
    true_n = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    true_c = np.array([0.0], dtype=np.float32)

    def tilt_of(n):
        return float(np.degrees(np.arccos(np.clip(abs(np.dot(axis, n)), 0, 1))))

    print(f"structure: {OPM.name} | {N_RANDOM} random placements | true = +z, midplane 0")
    print(f"true placement tilt: {tilt_of(true_n[0]):.0f} deg "
          f"(OPM experimental: {OPM_TILT_DEG:.0f})\n")
    print(f"{'objective':22} {'true rank':>10} {'gap':>8} {'best tilt':>10}  verdict")

    results = []
    for obj in OBJECTIVES:
        # score the true placement
        e_true, info = obj(ca_nm.clone(), sequence=MELITTIN,
                           normal=torch.tensor(true_n), center=torch.tensor(true_c))
        if "normal" not in info:
            print(f"{obj.key:22} {'n/a':>10} {'':>8} {'':>10}  frame-free: ignores placement")
            results.append({"key": obj.key, "frame_free": True})
            continue
        e_true = float(e_true.item())

        # score every random placement (structure repeated, membrane varied)
        scores = []
        for s in range(0, N_RANDOM, 256):
            n = torch.tensor(normals[s : s + 256])
            c = torch.tensor(centers[s : s + 256])
            x = ca_nm.expand(len(n), -1, -1).contiguous()
            e, _ = obj(x, sequence=MELITTIN, normal=n, center=c)
            scores.append(e.detach().numpy())
        scores = np.concatenate(scores)

        pct = float((scores > e_true).mean() * 100)      # % of random placements that are worse
        spread = scores.std() or 1e-9
        gap = float(max(0.0, e_true - scores.min()) / spread)
        best_tilt = tilt_of(normals[int(scores.argmin())])
        ok = pct >= 95 and gap < 0.5
        print(f"{obj.key:22} {pct:9.1f}% {gap:8.2f} {best_tilt:9.0f}°  "
              f"{'PASS' if ok else 'fail'}")
        results.append({
            "key": obj.key, "name": obj.name, "true_score": e_true,
            "true_percentile": pct, "gap_sd": gap, "best_tilt_deg": best_tilt,
            "random_mean": float(scores.mean()), "random_best": float(scores.min()),
            "pass": bool(ok),
        })

    out = {
        "structure": OPM.name, "n_random": N_RANDOM,
        "true_placement": {"normal": [0, 0, 1], "center_A": 0.0,
                           "tilt_deg": tilt_of(true_n[0])},
        "opm_experimental_tilt_deg": OPM_TILT_DEG,
        "objectives": results,
    }
    dest = HERE / "objective_validation" / "placement_screen.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nPASS = true placement in the top 5% AND nothing beats it by more than 0.5 sd")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
