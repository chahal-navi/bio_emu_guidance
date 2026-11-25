"""Membrane-insertion potentials for guiding BioEmu.

`EzMembranePotential` evaluates the depth-dependent Ez potential of
Senes A, Chadi DC, Law PB, Walters RF, Nanda V, DeGrado WF,
"Ez, a depth-dependent potential for assessing the energies of insertion of
amino acid side-chains into membranes", J Mol Biol 2007 (366:436-448), Table 1,
on Cα positions. The parameter table and the CSV / Kabsch-reference workflow
are from Pratham's original implementation; this version is pure torch (no
numpy round-trip, no pandas/Biopython import at module load) and adds a
reference-free mode that finds the best membrane placement per structure.

All three frames reduce to a membrane normal ``n`` (unit vector) and a midplane
offset ``c`` (Å) per structure, so the depth of residue i is
``z_i = n · x_i - c`` and its energy is ``Ez_type(i)(|z_i|)``.

frame="optimize" (default, reference-free)
    (n, c) = argmin_{n, c} sum_i Ez(|n · x_i - c|), found by a grid search over
    directions × offsets followed by a few gradient refinement steps. The
    frame is held fixed (detached) when differentiating, which by the envelope
    theorem gives the exact gradient of min_{n,c} E at the optimum. At the
    optimum the net force and torque of the guidance on the structure vanish,
    so guidance changes the *conformation*, not the rigid-body pose.

frame="kabsch" (Pratham's original)
    Superimpose the structure onto a membrane-placed reference (e.g. an OPM
    entry, in which the bilayer midplane is z = 0) using the residues with a
    reference coordinate in the CSV, then use z of the aligned coordinates.
    The alignment is detached, as in the original numpy implementation.

frame="fixed"
    Membrane normal = coordinate axis `membrane_axis`, midplane at
    `membrane_center` (Å) in the lab frame. Mainly for tests.

Energies are in kcal/mol. Lower is more favourable (upstream steering uses
reward = -potential).
"""

from __future__ import annotations

import csv
import math
import warnings
from typing import Any

import torch

from .potentials import Potential

# --- Senes et al. 2007, Table 1 -------------------------------------------------
# Sigmoidal form (Eq. 2): dE(z) = dE0 / (1 + (|z| / Zmid) ** n)     resname -> (dE0, Zmid, n)
EZ_SIGMOID_PARAMS: dict[str, tuple[float, float, float]] = {
    "ALA": (-0.29, 10.22, 4.67),
    "ASP": (1.19, 14.25, 8.98),
    "GLU": (1.30, 14.66, 4.16),
    "PHE": (-0.80, 19.67, 7.12),
    "GLY": (-0.01, 13.86, 6.00),
    "HIS": (0.75, 12.26, 2.77),
    "ILE": (-0.56, 14.34, 10.69),
    "LYS": (1.66, 11.11, 2.09),
    "LEU": (-0.64, 17.34, 8.61),
    "MET": (-0.28, 18.04, 7.13),
    "ASN": (0.89, 12.78, 6.28),
    "PRO": (0.83, 18.09, 3.53),
    "GLN": (1.21, 10.46, 2.59),
    "ARG": (1.55, 9.34, 4.68),
    "SER": (0.10, 13.86, 6.00),
    "THR": (0.01, 13.86, 6.00),
    "VAL": (-0.47, 11.35, 4.97),
}
# Gaussian form (Eq. 3): dE(z) = dEmin * exp(-(|z| - Zmin)^2 / (2 sigma^2))   resname -> (dEmin, Zmin, sigma)
EZ_GAUSSIAN_PARAMS: dict[str, tuple[float, float, float]] = {
    "TRP": (-0.85, 11.65, 7.20),
    "TYR": (-0.42, 13.04, 6.20),
}
# Cys: scattered distribution in the paper -> flat zero curve.
EZ_ZERO_RESIDUES = {"CYS"}

ONE_TO_THREE = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}  # fmt: skip
RESNAME_ALIASES = {
    "HSD": "HIS", "HSE": "HIS", "HSP": "HIS", "HID": "HIS", "HIE": "HIS", "HIP": "HIS",
    "CYX": "CYS", "CYM": "CYS", "ASH": "ASP", "GLH": "GLU", "LYN": "LYS",
}  # fmt: skip

_SMOOTH_ABS_EPS_A2 = 1e-2  # (0.1 Å)^2: avoids the kink of |z| at the midplane


def normalize_resname(name: str) -> str:
    name = name.strip().upper()
    if len(name) == 1:
        name = ONE_TO_THREE.get(name, name)
    return RESNAME_ALIASES.get(name, name)


def ez_parameters(resnames: list[str]) -> dict[str, torch.Tensor]:
    """Per-residue Ez parameter tensors (length n_res)."""
    n = len(resnames)
    # p2 and p3 must be NON-ZERO even for residues with no parameters (Cys, unknowns).
    # p2 is a divisor: (z/p2)**p3. Leaving it at 0 gives inf in the forward pass, which
    # the is_known mask hides (0 * inf -> 0 for the energy), but the BACKWARD pass then
    # yields NaN and poisons the whole gradient. The energy looks fine and guidance
    # silently dies. This only bites on sequences containing Cys, which is why melittin,
    # pardaxin and magainin-2 all ran clean and MgtR produced all-NaN coordinates.
    p1 = torch.zeros(n)
    p2, p3 = torch.ones(n), torch.ones(n)
    is_gaussian = torch.zeros(n, dtype=torch.bool)
    is_known = torch.zeros(n)
    unknown = set()
    for i, name in enumerate(resnames):
        if name in EZ_ZERO_RESIDUES:
            continue
        if name in EZ_GAUSSIAN_PARAMS:
            p1[i], p2[i], p3[i] = EZ_GAUSSIAN_PARAMS[name]
            is_gaussian[i] = True
            is_known[i] = 1.0
        elif name in EZ_SIGMOID_PARAMS:
            p1[i], p2[i], p3[i] = EZ_SIGMOID_PARAMS[name]
            is_known[i] = 1.0
        else:
            unknown.add(name)
    if unknown:
        warnings.warn(f"No Ez parameters for {sorted(unknown)}; they contribute zero energy.")
    return {"p1": p1, "p2": p2, "p3": p3, "is_gaussian": is_gaussian, "is_known": is_known}


def ez_energy_per_residue(z_abs: torch.Tensor, params: dict[str, torch.Tensor]) -> torch.Tensor:
    """Ez energy (kcal/mol) for absolute depths z_abs (Å), shape (..., n_res)."""
    p1, p2, p3 = params["p1"], params["p2"], params["p3"]
    sigmoid = p1 / (1.0 + (z_abs / p2) ** p3)
    gaussian = p1 * torch.exp(-((z_abs - p2) ** 2) / (2.0 * p3**2))
    return torch.where(params["is_gaussian"], gaussian, sigmoid) * params["is_known"]


def fibonacci_hemisphere(n: int) -> torch.Tensor:
    """n roughly uniform unit vectors on the upper hemisphere (energy is symmetric in ±n)."""
    i = torch.arange(n, dtype=torch.float64) + 0.5
    z = i / n  # uniform in (0, 1] -> uniform area on the hemisphere
    r = torch.sqrt(1 - z**2)
    phi = i * math.pi * (3.0 - math.sqrt(5.0))
    return torch.stack([r * torch.cos(phi), r * torch.sin(phi), z], dim=-1).float()


def kabsch(mobile: torch.Tensor, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched Kabsch. mobile: (B, N, 3), reference: (N, 3) or (B, N, 3).

    Returns rot (B, 3, 3), tran (B, 3) with ``mobile @ rot + tran ≈ reference``
    (row-vector convention, same as Bio.SVDSuperimposer.get_rotran).
    """
    reference = reference.expand_as(mobile)
    mu_m, mu_r = mobile.mean(dim=-2, keepdim=True), reference.mean(dim=-2, keepdim=True)
    H = (mobile - mu_m).transpose(-1, -2) @ (reference - mu_r)
    U, _, Vh = torch.linalg.svd(H)
    d = torch.sign(torch.linalg.det(U @ Vh))
    D = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], dim=-1))
    rot = U @ D @ Vh
    tran = (mu_r - mu_m @ rot).squeeze(-2)
    return rot, tran


class EzMembranePotential(Potential):
    """Senes et al. 2007 Ez membrane-insertion potential on Cα coordinates.

    Args:
        sequence: Default residue sequence (1-letter string or list of names). Overridden by
            the ``sequence`` passed at call time (BioEmu's steering code passes it).
        csv_path: Optional per-residue CSV (columns resid, resname, has_ref, ref_x, ref_y,
            ref_z, membrane_interacting) as written by ``bioemu.membrane_csv``. Required for
            frame="kabsch"; also supplies the membrane_interacting mask and default sequence.
        frame: "optimize" | "kabsch" | "fixed" (see module docstring).
        only_interacting: Score only CSV-flagged membrane_interacting residues. Defaults to
            True when a CSV is given, False otherwise.
        weight: Multiplier on the energy.
        membrane_axis, membrane_center: For "fixed" and "kabsch": normal axis and midplane
            position (Å) in the (aligned) frame. OPM files have the midplane at z = 0.
        n_directions, max_offset, offset_step, refine_steps: Frame search settings
            for frame="optimize".
    """

    def __init__(
        self,
        sequence: str | list[str] | None = None,
        csv_path: str | None = None,
        frame: str = "optimize",
        only_interacting: bool | None = None,
        weight: float = 1.0,
        membrane_axis: int = 2,
        membrane_center: float = 0.0,
        n_directions: int = 300,
        max_offset: float = 60.0,
        offset_step: float = 1.5,
        refine_steps: int = 30,
        guidance_steering: bool = False,
        **_: Any,
    ) -> None:
        if frame not in ("optimize", "kabsch", "fixed"):
            raise ValueError(f"Unknown frame {frame!r}")
        self.frame = frame
        self.weight = weight
        self.membrane_axis = membrane_axis
        self.membrane_center = membrane_center
        self.n_directions = n_directions
        self.max_offset = max_offset
        self.offset_step = offset_step
        self.refine_steps = refine_steps
        self.guidance_steering = guidance_steering

        self._csv_resnames: list[str] | None = None
        self._csv_mask: torch.Tensor | None = None
        self._fit_idx: torch.Tensor | None = None
        self._ref_fit_A: torch.Tensor | None = None
        if csv_path is not None:
            self._load_csv(csv_path)
        if frame == "kabsch" and self._ref_fit_A is None:
            raise ValueError("frame='kabsch' requires a csv_path with reference coordinates.")

        self.only_interacting = (csv_path is not None) if only_interacting is None else only_interacting
        if self.only_interacting and self._csv_mask is None:
            raise ValueError("only_interacting=True requires a csv_path.")

        if sequence is not None:
            self.default_resnames = [normalize_resname(s) for s in sequence]
        else:
            self.default_resnames = self._csv_resnames
        self._param_cache: dict[tuple[str, ...], dict[str, torch.Tensor]] = {}
        self._directions = fibonacci_hemisphere(n_directions)
        self._offsets = torch.arange(-max_offset, max_offset + 1e-6, offset_step)

    # -- setup ---------------------------------------------------------------------------
    def _load_csv(self, csv_path: str) -> None:
        with open(csv_path, newline="") as f:
            rows = sorted(csv.DictReader(f), key=lambda r: int(r["resid"]))
        self._csv_resnames = [normalize_resname(r["resname"]) for r in rows]
        self._csv_mask = torch.tensor(
            [r["membrane_interacting"].strip().lower() == "true" for r in rows]
        )
        if "has_ref" in rows[0]:
            has_ref = torch.tensor([r["has_ref"].strip().lower() == "true" for r in rows])
            xyz = torch.tensor(
                [
                    [float(r["ref_x"]), float(r["ref_y"]), float(r["ref_z"])]
                    if h
                    else [0.0, 0.0, 0.0]
                    for r, h in zip(rows, has_ref.tolist())
                ]
            )
            self._fit_idx = torch.nonzero(has_ref).squeeze(-1)
            self._ref_fit_A = xyz[has_ref]
            if len(self._fit_idx) < 3:
                raise ValueError("Need at least 3 residues with a reference for Kabsch.")

    def _params(self, sequence, n_res: int, device, dtype) -> dict[str, torch.Tensor]:
        if sequence is not None:
            resnames = [normalize_resname(s) for s in sequence]
        elif self.default_resnames is not None:
            resnames = self.default_resnames
        else:
            raise ValueError("EzMembranePotential needs a sequence (init, call, or CSV).")
        if len(resnames) != n_res:
            raise ValueError(f"Sequence has {len(resnames)} residues but coords have {n_res}.")
        key = tuple(resnames)
        if key not in self._param_cache:
            self._param_cache[key] = ez_parameters(resnames)
        return {
            k: v.to(device=device, dtype=dtype if v.dtype != torch.bool else torch.bool)
            for k, v in self._param_cache[key].items()
        }

    def _mask(self, n_res: int, device, dtype) -> torch.Tensor:
        if self.only_interacting:
            assert self._csv_mask is not None and len(self._csv_mask) == n_res
            return self._csv_mask.to(device=device, dtype=dtype)
        return torch.ones(n_res, device=device, dtype=dtype)

    # -- membrane frame --------------------------------------------------------------------
    @staticmethod
    def _smooth_abs(z: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(z**2 + _SMOOTH_ABS_EPS_A2)

    def _frame_fixed(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = x.shape[0]
        n = torch.zeros(B, 3, device=x.device, dtype=x.dtype)
        n[:, self.membrane_axis] = 1.0
        c = torch.full((B,), self.membrane_center, device=x.device, dtype=x.dtype)
        return n, c

    @torch.no_grad()
    def _frame_kabsch(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._fit_idx is not None and self._ref_fit_A is not None
        idx = self._fit_idx.to(x.device)
        ref = self._ref_fit_A.to(device=x.device, dtype=torch.float64)
        rot, tran = kabsch(x[:, idx].double(), ref)
        # aligned_z_i = x_i @ rot[:, axis] + tran[axis]  =>  n = rot[:, :, axis], c = -tran[axis] + center
        n = rot[:, :, self.membrane_axis].to(x.dtype)
        c = (-tran[:, self.membrane_axis] + self.membrane_center).to(x.dtype)
        return n, c

    def _frame_optimize(self, x, params, mask) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.detach()
        B, L, _ = x.shape
        dirs = self._directions.to(device=x.device, dtype=x.dtype)  # (K, 3)
        offsets = self._offsets.to(device=x.device, dtype=x.dtype)  # (M,)
        K, M = dirs.shape[0], offsets.shape[0]
        chunk = max(1, int(2e7 // (K * M * L)))
        best_n, best_c = [], []
        with torch.no_grad():
            for s in range(0, B, chunk):
                xb = x[s : s + chunk]
                proj = torch.einsum("bld,kd->bkl", xb, dirs)  # (b, K, L)
                centers = proj.mean(-1, keepdim=True) + offsets  # (b, K, M)
                z = proj[:, :, None, :] - centers[..., None]  # (b, K, M, L)
                e = (ez_energy_per_residue(self._smooth_abs(z), params) * mask).sum(-1)
                flat = e.flatten(1).argmin(-1)  # (b,)
                k_idx, m_idx = flat // M, flat % M
                ar = torch.arange(xb.shape[0], device=x.device)
                best_n.append(dirs[k_idx])
                best_c.append(centers[ar, k_idx, m_idx])
        n = torch.cat(best_n)
        c = torch.cat(best_c)

        if self.refine_steps > 0:
            with torch.enable_grad():
                v = n.clone().requires_grad_(True)
                c = c.clone().requires_grad_(True)
                opt = torch.optim.Adam([{"params": [v], "lr": 0.02}, {"params": [c], "lr": 0.3}])
                for _ in range(self.refine_steps):
                    opt.zero_grad()
                    nn_ = v / v.norm(dim=-1, keepdim=True)
                    z = torch.einsum("bld,bd->bl", x, nn_) - c[:, None]
                    e = (ez_energy_per_residue(self._smooth_abs(z), params) * mask).sum()
                    e.backward()
                    opt.step()
            n = (v / v.norm(dim=-1, keepdim=True)).detach()
            c = c.detach()
        return n, c

    # -- public API ------------------------------------------------------------------------
    def evaluate(self, ca_pos_nm: torch.Tensor, sequence=None) -> dict[str, torch.Tensor]:
        """Energy plus diagnostics. ca_pos_nm: (..., n_res, 3) in nm.

        Returns dict with energy (...,), per_residue (..., n_res), depth_A (..., n_res; signed),
        normal (..., 3), center_A (...,).
        """
        batch_shape, n_res = ca_pos_nm.shape[:-2], ca_pos_nm.shape[-2]
        x = (ca_pos_nm * 10.0).reshape(-1, n_res, 3)  # nm -> Å
        params = self._params(sequence, n_res, x.device, x.dtype)
        mask = self._mask(n_res, x.device, x.dtype)

        if self.frame == "fixed":
            n, c = self._frame_fixed(x)
        elif self.frame == "kabsch":
            n, c = self._frame_kabsch(x)
        else:
            n, c = self._frame_optimize(x, params, mask)

        z = torch.einsum("bld,bd->bl", x, n) - c[:, None]  # differentiable w.r.t. x only
        per_res = ez_energy_per_residue(self._smooth_abs(z), params) * mask * self.weight
        return {
            "energy": per_res.sum(-1).reshape(batch_shape),
            "per_residue": per_res.reshape(*batch_shape, n_res),
            "depth_A": z.reshape(*batch_shape, n_res),
            "normal": n.reshape(*batch_shape, 3),
            "center_A": c.reshape(batch_shape),
        }

    def __call__(self, ca_pos_nm: torch.Tensor, *, t=None, sequence=None) -> torch.Tensor:
        return self.evaluate(ca_pos_nm, sequence=sequence)["energy"]

    def loss_fn(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def energy_from_cv(self, cv_values: torch.Tensor, t: float | None = None) -> torch.Tensor:
        """cv_values: absolute depths |z| (Å) from the midplane, shape (..., n_res)."""
        n_res = cv_values.shape[-1]
        params = self._params(None, n_res, cv_values.device, cv_values.dtype)
        mask = self._mask(n_res, cv_values.device, cv_values.dtype)
        return (ez_energy_per_residue(cv_values, params) * mask).sum(-1) * self.weight


class MembranePotentialPlaceholder(EzMembranePotential):
    """Backward-compatible name/constructor for Pratham's original CSV + Kabsch potential."""

    def __init__(
        self,
        csv_path: str,
        membrane_axis: int = 2,
        membrane_center: float = 0.0,
        only_interacting: bool = True,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("frame", "kabsch")
        super().__init__(
            csv_path=csv_path,
            membrane_axis=membrane_axis,
            membrane_center=membrane_center,
            only_interacting=only_interacting,
            **kwargs,
        )


class SumPotential(Potential):
    """Weighted sum of several potentials evaluated on the same Cα coordinates.

    Lets membrane guidance be combined with BioEmu's own physicality potentials
    (chain-break / clash), which is what keeps an unbounded membrane term from
    simply stretching the peptide apart: the Ez potential rewards moving charged
    residues infinitely far from the bilayer, and only a chain-integrity term
    opposes that.

    Every sub-potential must accept ``(ca_pos_nm, t=..., sequence=...)`` and return
    one energy per structure, which is true of both `EzMembranePotential` and the
    upstream `UmbrellaPotential`.
    """

    def __init__(self, potentials, weights=None, guidance_steering: bool = False, **_: Any) -> None:
        self.potentials = list(potentials)
        if not self.potentials:
            raise ValueError("SumPotential needs at least one potential.")
        self.weights = [1.0] * len(self.potentials) if weights is None else list(weights)
        if len(self.weights) != len(self.potentials):
            raise ValueError(
                f"Got {len(self.weights)} weights for {len(self.potentials)} potentials."
            )
        self.guidance_steering = guidance_steering

    def __call__(self, ca_pos_nm: torch.Tensor, *, t=None, sequence=None) -> torch.Tensor:
        total = None
        for w, p in zip(self.weights, self.potentials):
            e = w * p(ca_pos_nm, t=t, sequence=sequence)
            total = e if total is None else total + e
        return total

    def terms(self, ca_pos_nm: torch.Tensor, *, t=None, sequence=None) -> dict[str, torch.Tensor]:
        """Per-term energies, for diagnosing which term dominates the gradient."""
        return {
            f"{i}_{type(p).__name__}": (w * p(ca_pos_nm, t=t, sequence=sequence)).detach()
            for i, (w, p) in enumerate(zip(self.weights, self.potentials))
        }

    def loss_fn(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def energy_from_cv(self, cv_values: torch.Tensor, t: float | None = None) -> torch.Tensor:
        raise NotImplementedError("SumPotential has no single collective variable.")
