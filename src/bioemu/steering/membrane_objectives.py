"""Candidate membrane-insertion objectives for guiding BioEmu, and the machinery to
validate them *before* they are used for guidance.

Motivation
----------
The Ez depth potential (Senes et al. 2007) as applied in `custom_potentials.py` turned out
to be an invalid guidance objective. Measured on 650 generated melittin structures:

  * Ez correlates POSITIVELY with helicity (r = +0.37 baseline, +0.71 under guidance),
    i.e. it rewards LOSS of secondary structure.
  * Structures that fail the physicality filter score BETTER than those that pass
    (-3.18 vs -2.84 at guidance_scale=100), i.e. it rewards breaking the chain.

Both follow from the same structural property: Ez is **additive and per-residue**,
E = sum_i f(type_i, depth_i), so the cheapest way to lower it is to gain the freedom to
place each residue at its own optimal depth. Unfolding and chain-breaking both buy that
freedom. No hyperparameter fixes this.

An objective is only usable for guidance if its optimum is REACHABLE BY A CONNECTED,
FOLDED CHAIN. This module collects candidates and tests exactly that.

Conventions
-----------
Every objective is a callable with the signature

    objective(coords_nm, *, sequence, normal=None, center=None) -> (energy, info)

where `coords_nm` is (B, n_res, 3) Ca coordinates in nm, `energy` is (B,) with LOWER =
MORE membrane-compatible (so it can be minimised by guidance, matching the sign
convention of `bioemu.steering.Potential`), and `info` is a dict of diagnostics.

If `normal`/`center` are None the objective solves for its own membrane frame.
All objectives are differentiable w.r.t. `coords_nm` unless stated otherwise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from .custom_potentials import (
    ONE_TO_THREE,
    kabsch,
    EzMembranePotential,
    ez_energy_per_residue,
    ez_parameters,
    fibonacci_hemisphere,
    normalize_resname,
)

# --- hydrophobicity scales -------------------------------------------------------------
# Eisenberg consensus scale (Eisenberg, Weiss & Terwilliger 1982), used for the
# hydrophobic-moment objectives. Positive = hydrophobic.
EISENBERG = {
    "A": 0.62, "R": -2.53, "N": -0.78, "D": -0.90, "C": 0.29,
    "Q": -0.85, "E": -0.74, "G": 0.48, "H": -0.40, "I": 1.38,
    "L": 1.06, "K": -1.50, "M": 0.64, "F": 1.19, "P": 0.12,
    "S": -0.18, "T": -0.05, "W": 0.81, "Y": 0.26, "V": 1.08,
}  # fmt: skip

_EPS = 1e-8
_SMOOTH_ABS_EPS_A2 = 1e-2


def _smooth_abs(z: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(z**2 + _SMOOTH_ABS_EPS_A2)


def hydrophobicity_vector(sequence: str, device, dtype) -> torch.Tensor:
    return torch.tensor([EISENBERG.get(a.upper(), 0.0) for a in sequence], device=device, dtype=dtype)


def local_helix_axes(ca_A: torch.Tensor, window: int = 4) -> torch.Tensor:
    """Per-residue local chain direction, (B, n_res, 3), unit length.

    Uses the vector between residues i-window/2 and i+window/2, which is a standard
    cheap proxy for the local helix axis and is well defined for a coil too.
    """
    B, L, _ = ca_A.shape
    h = window // 2
    idx_lo = torch.clamp(torch.arange(L, device=ca_A.device) - h, min=0)
    idx_hi = torch.clamp(torch.arange(L, device=ca_A.device) + h, max=L - 1)
    v = ca_A[:, idx_hi, :] - ca_A[:, idx_lo, :]
    return v / v.norm(dim=-1, keepdim=True).clamp(min=_EPS)


def radial_directions(ca_A: torch.Tensor, window: int = 4) -> torch.Tensor:
    """Unit vector from the local helix axis out to each residue, (B, n_res, 3).

    This is the direction a side chain points, approximated from backbone geometry only:
    the component of (residue - local centroid) perpendicular to the local chain direction.
    """
    B, L, _ = ca_A.shape
    h = max(1, window // 2)
    # local centroid over a window, via cumulative sums
    pad = torch.nn.functional.pad(ca_A.transpose(1, 2), (h, h), mode="replicate").transpose(1, 2)
    kernel = 2 * h + 1
    centroid = torch.stack([pad[:, i : i + kernel, :].mean(1) for i in range(L)], dim=1)
    radial = ca_A - centroid
    axis = local_helix_axes(ca_A, window=window)
    radial = radial - (radial * axis).sum(-1, keepdim=True) * axis  # remove axial component
    return radial / radial.norm(dim=-1, keepdim=True).clamp(min=_EPS)


# ---------------------------------------------------------------------------------------
# Membrane frame search, shared by objectives that need their own frame.
# ---------------------------------------------------------------------------------------


def optimize_frame(
    ca_A: torch.Tensor,
    energy_fn,
    n_directions: int = 200,
    max_offset: float = 50.0,
    offset_step: float = 2.0,
    refine_steps: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Grid-search (normal, center) minimising `energy_fn(z)` where z = n.x - c (Angstrom).

    `energy_fn` takes signed depths (..., n_res) and returns (...,) energies.
    Returns (normal (B,3), center (B,)), both DETACHED: by the envelope theorem the
    gradient of the minimised energy w.r.t. coordinates is the partial derivative at the
    fixed optimum, which also makes the guidance free of net force and torque.
    """
    with torch.no_grad():
        B, L, _ = ca_A.shape
        dirs = fibonacci_hemisphere(n_directions).to(device=ca_A.device, dtype=ca_A.dtype)
        offsets = torch.arange(
            -max_offset, max_offset + 1e-6, offset_step, device=ca_A.device, dtype=ca_A.dtype
        )
        K, M = dirs.shape[0], offsets.shape[0]
        proj = torch.einsum("bld,kd->bkl", ca_A, dirs)             # (B,K,L)
        centers = proj.mean(-1, keepdim=True) + offsets            # (B,K,M)
        z = proj[:, :, None, :] - centers[..., None]               # (B,K,M,L)
        e = energy_fn(z)                                           # (B,K,M)
        flat = e.flatten(1).argmin(-1)
        k_idx, m_idx = flat // M, flat % M
        ar = torch.arange(B, device=ca_A.device)
        n = dirs[k_idx]
        c = centers[ar, k_idx, m_idx]
    return n.detach(), c.detach()


# ---------------------------------------------------------------------------------------
# The candidate objectives.
# ---------------------------------------------------------------------------------------


@dataclass
class Objective:
    """A candidate guidance objective plus metadata for the report."""

    key: str
    name: str
    rationale: str
    fn: object = field(repr=False)
    couples_residues: bool = False  # does it contain a term linking residues?
    bounded: bool = True  # is the optimum reachable by a connected chain?

    def __call__(self, coords_nm, *, sequence, normal=None, center=None):
        return self.fn(coords_nm, sequence=sequence, normal=normal, center=center)


# --- 1. Ez on Ca (the control: known-bad) ----------------------------------------------
def ez_ca(coords_nm, *, sequence, normal=None, center=None):
    ca_A = coords_nm * 10.0
    resnames = [normalize_resname(s) for s in sequence]
    params = {k: v.to(ca_A.device) for k, v in ez_parameters(resnames).items()}
    params = {k: (v.to(ca_A.dtype) if v.dtype != torch.bool else v) for k, v in params.items()}

    def energy_of_z(z):
        return ez_energy_per_residue(_smooth_abs(z), params).sum(-1)

    if normal is None:
        normal, center = optimize_frame(ca_A.detach(), energy_of_z)
    z = torch.einsum("bld,bd->bl", ca_A, normal) - center[:, None]
    e = ez_energy_per_residue(_smooth_abs(z), params).sum(-1)
    return e, {"depth_A": z, "normal": normal, "center_A": center}


# --- 2. Ez with per-residue reward clamped (removes the "expel to infinity" gain) -------
def ez_clamped(coords_nm, *, sequence, normal=None, center=None, floor: float = -0.9):
    """Ez, but each residue's energy is clamped BELOW at `floor` and ABOVE at 0.

    Clamping above at 0 is the important half: a residue pushed out of the bilayer then
    gains nothing further, which removes the unbounded 'expel the charges' direction.
    """
    ca_A = coords_nm * 10.0
    resnames = [normalize_resname(s) for s in sequence]
    params = {k: v.to(ca_A.device) for k, v in ez_parameters(resnames).items()}
    params = {k: (v.to(ca_A.dtype) if v.dtype != torch.bool else v) for k, v in params.items()}

    def energy_of_z(z):
        per = ez_energy_per_residue(_smooth_abs(z), params)
        return per.clamp(min=floor, max=0.0).sum(-1)

    if normal is None:
        normal, center = optimize_frame(ca_A.detach(), energy_of_z)
    z = torch.einsum("bld,bd->bl", ca_A, normal) - center[:, None]
    per = ez_energy_per_residue(_smooth_abs(z), params).clamp(min=floor, max=0.0)
    return per.sum(-1), {"depth_A": z, "normal": normal, "center_A": center}


# --- 3. Ez + centroid restraint (peptide must stay AT the bilayer) ----------------------
def ez_anchored(coords_nm, *, sequence, normal=None, center=None, k: float = 0.02):
    """Ez plus a harmonic penalty on the peptide centroid's distance from the midplane.

    Blocks the degenerate solution 'move the whole peptide out of the membrane', which
    is what produced the 92 A depths in pilot v1.
    """
    ca_A = coords_nm * 10.0
    resnames = [normalize_resname(s) for s in sequence]
    params = {k_: v.to(device=ca_A.device) for k_, v in ez_parameters(resnames).items()}
    params = {k_: (v.to(ca_A.dtype) if v.dtype != torch.bool else v) for k_, v in params.items()}

    def energy_of_z(z):
        return ez_energy_per_residue(_smooth_abs(z), params).sum(-1) + k * (z.mean(-1) ** 2)

    if normal is None:
        normal, center = optimize_frame(ca_A.detach(), energy_of_z)
    z = torch.einsum("bld,bd->bl", ca_A, normal) - center[:, None]
    e = ez_energy_per_residue(_smooth_abs(z), params).sum(-1) + k * (z.mean(-1) ** 2)
    return e, {"depth_A": z, "normal": normal, "center_A": center}


# --- 4. Hydrophobic moment magnitude ----------------------------------------------------
def hydrophobic_moment(coords_nm, *, sequence, normal=None, center=None):
    """-|mu_H|, where mu_H = sum_i h_i * r_hat_i  (r_hat = radial direction from helix axis).

    Classical amphipathicity measure (Eisenberg). A disordered chain has incoherent radial
    directions so the vector sum cancels; a well-formed amphipathic helix gives a large
    moment. Needs NO membrane frame, and critically it CANNOT be improved by stretching
    the chain - pulling residues apart destroys the coherence that creates the moment.
    """
    ca_A = coords_nm * 10.0
    h = hydrophobicity_vector(sequence, ca_A.device, ca_A.dtype)
    r_hat = radial_directions(ca_A)
    mu = (h[None, :, None] * r_hat).sum(1)                   # (B,3)
    mag = mu.norm(dim=-1)
    return -mag, {"moment_vec": mu, "moment_mag": mag}


# --- 5. Hydrophobic moment ALIGNED with the membrane normal -----------------------------
def moment_alignment(coords_nm, *, sequence, normal=None, center=None):
    """+(mu_H . n): rewards an amphipathic helix whose hydrophobic face points INTO the core.

    Combines amphipathicity with orientation. If no frame is supplied the optimal normal is
    simply mu_H itself (the alignment is maximised by n = mu_H/|mu_H|), so this reduces to
    the moment magnitude but additionally REPORTS the implied membrane normal, which is the
    quantity rotational guidance would act on.
    """
    ca_A = coords_nm * 10.0
    h = hydrophobicity_vector(sequence, ca_A.device, ca_A.dtype)
    r_hat = radial_directions(ca_A)
    mu = (h[None, :, None] * r_hat).sum(1)
    if normal is None:
        # free choice of frame: point the normal opposite the moment, which is the
        # minimiser of mu . n and makes this reduce to -|mu_H|.
        n = (-mu / mu.norm(dim=-1, keepdim=True).clamp(min=_EPS)).detach()
    else:
        n = normal
    align = (mu * n).sum(-1)
    return align, {"moment_vec": mu, "normal": n, "alignment": align}


# --- 6. Amphipathic depth contrast ------------------------------------------------------
def depth_contrast(coords_nm, *, sequence, normal=None, center=None):
    """-(mean depth of polar residues - mean depth of hydrophobic residues), in Angstrom.

    Directly scores 'hydrophobics deeper than polars'. Unlike Ez this is a DIFFERENCE of
    means, so uniformly translating the peptide does nothing and expelling everything
    scores zero, not infinity. Still per-residue additive, so included to test whether
    additivity alone is the problem or specifically the unbounded reward.
    """
    ca_A = coords_nm * 10.0
    h = hydrophobicity_vector(sequence, ca_A.device, ca_A.dtype)
    phobic = (h > 0.4).to(ca_A.dtype)
    polar = (h < -0.4).to(ca_A.dtype)
    if phobic.sum() == 0 or polar.sum() == 0:
        raise ValueError("sequence has no clear hydrophobic/polar split")

    def energy_of_z(z):
        za = _smooth_abs(z)
        return (za * phobic).sum(-1) / phobic.sum() - (za * polar).sum(-1) / polar.sum()

    if normal is None:
        normal, center = optimize_frame(ca_A.detach(), energy_of_z)
    z = torch.einsum("bld,bd->bl", ca_A, normal) - center[:, None]
    za = _smooth_abs(z)
    e = (za * phobic).sum(-1) / phobic.sum() - (za * polar).sum(-1) / polar.sum()
    return e, {"depth_A": z, "normal": normal, "center_A": center}


# --- 7. Hydrophobic burial with a saturating reward -------------------------------------
def burial_saturating(coords_nm, *, sequence, normal=None, center=None, half_width: float = 15.0):
    """-sum_i h_i * sigma(depth_i), sigma = smooth indicator of 'inside the bilayer core'.

    Each residue contributes at most |h_i|: hydrophobics gain by being inside, polars lose.
    Bounded on BOTH sides, so there is no gain from pushing a residue to infinity (sigma
    saturates at 0) nor from burying it infinitely deep (sigma saturates at 1).
    """
    ca_A = coords_nm * 10.0
    h = hydrophobicity_vector(sequence, ca_A.device, ca_A.dtype)

    def inside(z):
        return torch.sigmoid((half_width - _smooth_abs(z)) / 3.0)

    def energy_of_z(z):
        return -(h * inside(z)).sum(-1)

    if normal is None:
        normal, center = optimize_frame(ca_A.detach(), energy_of_z)
    z = torch.einsum("bld,bd->bl", ca_A, normal) - center[:, None]
    e = -(h * inside(z)).sum(-1)
    return e, {"depth_A": z, "normal": normal, "center_A": center}


# --- 8. Moment alignment + saturating burial (the combined candidate) -------------------
def combined(coords_nm, *, sequence, normal=None, center=None, w_moment: float = 0.5):
    """Saturating burial (places the peptide) + hydrophobic moment (keeps it a folded helix).

    The burial term says WHERE the peptide should sit; the moment term says it must remain
    an amphipathic helix to get credit. The moment term is what couples residues, and it is
    destroyed by both unfolding and chain-breaking - the two failure modes Ez rewarded.
    """
    e_burial, info_b = burial_saturating(coords_nm, sequence=sequence, normal=normal, center=center)
    e_moment, info_m = hydrophobic_moment(coords_nm, sequence=sequence)
    return e_burial + w_moment * e_moment, {**info_b, **info_m}


OBJECTIVES: list[Objective] = [
    Objective("ez_ca", "Ez depth potential (Ca)",
              "Senes et al. 2007 depth potential on Ca atoms. The control: this is what we "
              "guided with, and it rewarded unfolding and chain-breaking.",
              ez_ca, couples_residues=False, bounded=False),
    Objective("ez_clamped", "Ez, per-residue reward clamped",
              "Ez with each residue's contribution clamped to [floor, 0]. Removes the gain "
              "from expelling a residue to infinity while keeping the Ez shape.",
              ez_clamped, couples_residues=False, bounded=True),
    Objective("ez_anchored", "Ez + centroid anchor",
              "Ez plus a harmonic penalty keeping the peptide centroid at the midplane, "
              "blocking the 'push the whole peptide out of the membrane' solution.",
              ez_anchored, couples_residues=False, bounded=True),
    Objective("depth_contrast", "Amphipathic depth contrast",
              "Mean depth of polar minus mean depth of hydrophobic residues. A difference "
              "of means, so global translation is free and total expulsion scores zero.",
              depth_contrast, couples_residues=False, bounded=True),
    Objective("burial_saturating", "Saturating hydrophobic burial",
              "Hydrophobicity-weighted smooth indicator of being inside the bilayer core. "
              "Bounded on both sides: no reward for infinite depth or infinite distance.",
              burial_saturating, couples_residues=False, bounded=True),
    Objective("hydrophobic_moment", "Hydrophobic moment |mu_H|",
              "Eisenberg amphipathic moment from backbone geometry. Requires COHERENT "
              "radial directions, so unfolding destroys it. Needs no membrane frame.",
              hydrophobic_moment, couples_residues=True, bounded=True),
    Objective("moment_alignment", "Moment aligned to membrane normal",
              "Hydrophobic moment projected on the membrane normal: rewards an amphipathic "
              "helix presenting its hydrophobic face to the bilayer core.",
              moment_alignment, couples_residues=True, bounded=True),
    Objective("combined_moment_heavy", "Burial + moment (moment-weighted 3x)",
              "Same two terms, but with the coupling term dominant. Included to find the "
              "weight at which the structure-preserving term overrides the gameable one.",
              lambda c, *, sequence, normal=None, center=None: combined(
                  c, sequence=sequence, normal=normal, center=center, w_moment=3.0),
              couples_residues=True, bounded=True),
    Objective("combined", "Saturating burial + moment",
              "Burial term places the peptide in the bilayer; moment term requires it to "
              "stay a folded amphipathic helix to collect the reward.",
              combined, couples_residues=True, bounded=True),
]

OBJECTIVES_BY_KEY = {o.key: o for o in OBJECTIVES}


# ---------------------------------------------------------------------------------------
# Reference membrane frame (OPM).
#
# The per-structure frame search in `optimize_frame` was measured to recover membrane
# normals 66-83 degrees away from the experimentally determined one: it finds whichever
# placement minimises the objective, not the physical placement. Transferring the frame
# from a structure whose membrane placement is already known is the better option.
# ---------------------------------------------------------------------------------------


def load_reference_ca(pdb_path: str) -> torch.Tensor:
    """Ca trace of an OPM-oriented structure, (n_res, 3) in Angstrom.

    OPM files are pre-rotated so the bilayer midplane is z = 0 and the hydrocarbon
    boundary is at |z| = (the "1/2 of bilayer thickness" REMARK). So in this frame the
    membrane normal is +z and the midplane offset is 0, by construction.
    """
    import mdtraj

    tr = mdtraj.load_pdb(pdb_path)
    ca = [a.index for a in tr.topology.atoms if a.name == "CA"]
    return torch.tensor(tr.xyz[0, ca, :] * 10.0, dtype=torch.float32)


def reference_frame(ca_A: torch.Tensor, ref_ca_A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Membrane frame transferred from an OPM reference by Kabsch superposition.

    Superimposes each structure onto `ref_ca_A` (which is in the OPM frame, midplane
    z = 0) and expresses the resulting membrane plane back in the INPUT structure's own
    coordinates, as (normal, center) with depth z = normal . x - center.

    Derivation: kabsch gives rot, tran with  x @ rot + tran ~= ref, so the aligned
    z-coordinate is  x . rot[:, 2] + tran[2]. Matching z = n . x - c gives
    n = rot[:, 2] and c = -tran[2].

    Detached, like `optimize_frame`: the alignment is held fixed when differentiating, so
    the gradient does not try to re-pose the molecule.
    """
    with torch.no_grad():
        rot, tran = kabsch(ca_A.double(), ref_ca_A.double().expand_as(ca_A))
        normal = rot[:, :, 2].to(ca_A.dtype)
        center = (-tran[:, 2]).to(ca_A.dtype)
    return normal.detach(), center.detach()
