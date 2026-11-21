"""Tests for the Ez membrane potential and the DPS-guided DPM solver."""

import csv
import math

import pytest
import torch
from torch_geometric.data import Batch

from bioemu.chemgraph import ChemGraph
from bioemu.denoiser import dpm_solver, membrane_guided_dpm_solver
from bioemu.sde_lib import CosineVPSDE
from bioemu.so3_sde import DiGSO3SDE, rotmat_to_rotvec
from bioemu.steering.custom_potentials import (
    EzMembranePotential,
    MembranePotentialPlaceholder,
    kabsch,
)

MELITTIN = "GIGAVLKVLTTGLPALISWIKRKRQQ"


def _random_rotation(gen=None) -> torch.Tensor:
    q = torch.linalg.qr(torch.randn(3, 3, generator=gen))[0]
    return q * torch.sign(torch.linalg.det(q))


def _ideal_helix_nm(n_res: int) -> torch.Tensor:
    """Ideal α-helix Cα trace (radius 2.3 Å, rise 1.5 Å, 100°/residue), in nm, axis along x."""
    i = torch.arange(n_res, dtype=torch.float32)
    ang = torch.deg2rad(100.0 * i)
    return torch.stack([1.5 * i, 2.3 * torch.cos(ang), 2.3 * torch.sin(ang)], -1) / 10.0


# ------------------------------------------------------------------------------------------
# Potential
# ------------------------------------------------------------------------------------------


def test_kabsch_recovers_rigid_transform():
    gen = torch.Generator().manual_seed(0)
    ref = torch.randn(20, 3, generator=gen, dtype=torch.float64)
    R = _random_rotation(gen).double()
    tr = torch.randn(3, generator=gen, dtype=torch.float64)
    mobile = (ref - tr) @ R.T  # so that mobile @ R + tr == ref
    rot, tran = kabsch(mobile[None], ref)
    torch.testing.assert_close(mobile @ rot[0] + tran[0], ref, atol=1e-5, rtol=0)


def test_ez_fixed_frame_values():
    """Leu at the midplane gets dE0; far outside the membrane everything is ~0."""
    pot = EzMembranePotential(sequence="LLL", frame="fixed", refine_steps=0)
    at_center = torch.zeros(1, 3, 3)
    torch.testing.assert_close(pot(at_center), torch.tensor([-0.64 * 3]), atol=1e-3, rtol=0)
    far = torch.zeros(1, 3, 3)
    far[..., 2] = 10.0  # 100 Å
    assert pot(far).abs().item() < 1e-6
    # Trp: Gaussian form peaking at 11.65 Å.
    trp = EzMembranePotential(sequence="W", frame="fixed")
    z = torch.zeros(1, 1, 3)
    z[..., 2] = 1.165
    torch.testing.assert_close(trp(z), torch.tensor([-0.85]), atol=1e-4, rtol=0)


def test_optimized_frame_is_rigid_invariant_and_force_free():
    torch.manual_seed(0)
    x = _ideal_helix_nm(len(MELITTIN))[None]
    pot = EzMembranePotential(sequence=MELITTIN, refine_steps=100)
    e0 = pot(x)
    assert e0.item() < -1.0  # amphipathic helix should find a favourable placement

    R = _random_rotation()
    x2 = x @ R.T + torch.tensor([0.7, -1.2, 3.0])
    torch.testing.assert_close(pot(x2), e0, atol=0.05, rtol=0)

    # Envelope theorem: at the optimal frame, net force and torque vanish.
    xg = x.clone().requires_grad_(True)
    (g,) = torch.autograd.grad(pot(xg).sum(), xg)
    g = g[0]
    assert g.abs().max() > 1e-2  # non-trivial per-residue forces ...
    net_force = g.sum(0).norm() / g.norm(dim=-1).sum()
    torque = torch.cross(x[0] - x[0].mean(0), g, dim=-1).sum(0).norm()
    torque_scale = (torch.cross(x[0] - x[0].mean(0), g, dim=-1).norm(dim=-1)).sum()
    assert net_force < 0.05, net_force  # ... that cancel in sum
    assert torque / torque_scale < 0.05


def test_optimized_frame_not_worse_than_any_fixed_frame():
    torch.manual_seed(1)
    x = _ideal_helix_nm(len(MELITTIN))[None]
    opt = EzMembranePotential(sequence=MELITTIN)(x)
    fixed = EzMembranePotential(sequence=MELITTIN, frame="fixed")
    for _ in range(20):
        R = _random_rotation()
        shift = torch.randn(3) * 1.5
        assert opt.item() <= fixed(x @ R.T + shift).item() + 1e-3


def test_kabsch_frame_from_csv(tmp_path):
    """Pratham's CSV/Kabsch mode: energy is invariant to rigid transforms of the input and
    equals the fixed-frame energy of the reference itself."""
    x_ref_A = _ideal_helix_nm(len(MELITTIN)) * 10.0 + torch.tensor([0.0, 0.0, 12.0])
    path = tmp_path / "ref.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["resid", "resname", "has_ref", "pdb_resnum", "ref_x", "ref_y", "ref_z",
                    "membrane_interacting"])  # fmt: skip
        for i, aa in enumerate(MELITTIN):
            w.writerow([i, aa, True, i + 1, *x_ref_A[i].tolist(), True])
    pot = MembranePotentialPlaceholder(csv_path=str(path))
    fixed = EzMembranePotential(sequence=MELITTIN, frame="fixed")
    x_ref_nm = (x_ref_A / 10.0)[None]
    torch.testing.assert_close(pot(x_ref_nm), fixed(x_ref_nm), atol=1e-4, rtol=0)
    moved = x_ref_nm @ _random_rotation().T + 2.0
    torch.testing.assert_close(pot(moved), fixed(x_ref_nm), atol=1e-4, rtol=0)


# ------------------------------------------------------------------------------------------
# Guided solver
# ------------------------------------------------------------------------------------------


class _LinearPotential:
    """E(x) = c * sum(x) over all coordinates of each graph."""

    def __init__(self, c: float):
        self.c = c

    def __call__(self, x, *, t=None, sequence=None):
        return self.c * x.sum(dim=(-1, -2))


def _gaussian_problem(batch_size=2000):
    x0_mean, x0_std = torch.tensor(-3.0), torch.tensor(4.3)
    r3sde = CosineVPSDE()
    so3sde = DiGSO3SDE(num_sigma=10)
    sdes = {"pos": r3sde, "node_orientations": so3sde}

    def score_fn(x: ChemGraph, t: torch.Tensor) -> ChemGraph:
        a_t, s_t = r3sde.marginal_prob(x=torch.ones_like(x.pos), t=t)
        x0 = (x0_mean * s_t**2 + x.pos * a_t * x0_std**2) / (s_t**2 + a_t**2 * x0_std**2)
        so3 = so3sde.compute_score(rotmat_to_rotvec(x.node_orientations), t)
        return {"pos": (x0 * a_t - x.pos) / s_t, "node_orientations": so3}

    batch = Batch.from_data_list(
        [
            ChemGraph(
                pos=torch.randn(batch_size, 1),
                node_orientations=so3sde.prior_sampling((batch_size,)),
            )
        ]
    )
    return sdes, score_fn, batch, x0_mean, x0_std


@pytest.mark.parametrize("backprop", [True, False])
def test_guidance_matches_analytic_tilt(backprop):
    """For Gaussian data and linear E, DPS is exact: p(x) exp(-lam c x) has mean mu - lam c s^2."""
    torch.manual_seed(0)
    sdes, score_fn, batch, mu, s = _gaussian_problem()
    lam, c = 1.0, 0.05
    out = membrane_guided_dpm_solver(
        sdes=sdes, batch=batch, N=200, score_model=score_fn, max_t=0.99, eps_t=0.001,
        device=torch.device("cpu"), potential=_LinearPotential(c), guidance_scale=lam,
        backprop_through_model=backprop,
    )  # fmt: skip
    expected = mu - lam * c * s**2  # -3.92
    if backprop:
        assert abs(out.pos.mean().item() - expected.item()) < 0.15, out.pos.mean()
        assert abs(out.pos.std().item() - s.item()) < 0.3
    else:
        # Score-as-constant approximation overshoots the tilt but moves in the right direction.
        assert out.pos.mean().item() < mu.item() - 0.5


def test_guidance_without_midpoint_is_too_weak():
    """Documents why guide_midpoint=True is needed with DPM-Solver-2."""
    torch.manual_seed(0)
    sdes, score_fn, batch, mu, s = _gaussian_problem()
    out = membrane_guided_dpm_solver(
        sdes=sdes, batch=batch, N=200, score_model=score_fn, max_t=0.99, eps_t=0.001,
        device=torch.device("cpu"), potential=_LinearPotential(0.05), guide_midpoint=False,
    )  # fmt: skip
    shift = out.pos.mean().item() - mu.item()
    assert abs(shift) < 0.5 * 0.05 * s.item() ** 2  # far less than the -0.92 it should be


def test_unguided_path_matches_dpm_solver():
    sdes, score_fn, batch, _, _ = _gaussian_problem(batch_size=300)
    kw = dict(sdes=sdes, batch=batch, N=30, score_model=score_fn, max_t=0.99, eps_t=0.001,
              device=torch.device("cpu"))  # fmt: skip
    torch.manual_seed(3)
    a = dpm_solver(**kw)
    torch.manual_seed(3)
    b = membrane_guided_dpm_solver(**kw)
    torch.testing.assert_close(a.pos, b.pos)
    torch.testing.assert_close(a.node_orientations, b.node_orientations)


def test_guided_solver_runs_with_tiny_model_and_ez(default_batch, tiny_model):
    """End-to-end smoke test with the real (tiny) score network and the Ez potential."""
    sdes = {"pos": CosineVPSDE(s=0.008), "node_orientations": DiGSO3SDE(num_sigma=10)}
    n_res = default_batch.pos.shape[0] // default_batch.num_graphs
    seq = MELITTIN[:n_res]  # test batch carries no sequence attribute
    torch.manual_seed(0)
    out = membrane_guided_dpm_solver(
        sdes=sdes, batch=default_batch, N=5, score_model=tiny_model, max_t=0.99, eps_t=0.001,
        device=torch.device("cpu"), potential=EzMembranePotential(sequence=seq, refine_steps=5),
        guidance_scale=1.0,
    )  # fmt: skip
    assert torch.isfinite(out.pos).all()
    assert not math.isnan(out.node_orientations.sum().item())
