""" HI pratham, I am leaving these comments here, the PLaceholder class defined below contains a classifier
this is for the Frame_class GNN which you trained, you can use it here for any potential which you deem
fit. I have used the below class in the denoiser.py to generate guidance gradients."""
""" Also the Placeholder potentials inherit from the Potential Abstract Base CLass, thus you will have to implement the 
loss_fn and energy_form_cv functions explicitly. Their detqails are as follows(copied from the Potential ABC)
    @abstractmethod
    def __call__(self, ca_pos_nm: torch.Tensor, *, t=None, sequence=None) -> torch.Tensor:
        Compute potential energy from Cα positions (in nm).

    @abstractmethod
    def loss_fn(self, x: torch.Tensor) -> torch.Tensor:
       Per-element loss function using instance attributes.

    @abstractmethod
    def energy_from_cv(self, cv_values: torch.Tensor, t: float | None = None) -> torch.Tensor:
       Compute energy from precomputed CV values.
    """

import torch
from .potentials import Potential
import csv
import warnings

from membrane_csv import *

class MembranePotentialPlaceholder(Potential):
    """
    Membrane-insertion guidance potential.

    Loads a per-residue CSV derived offline from an equilibrium MD
    trajectory (md.xtc) - e.g. membrane_per_residue.csv - and uses only its
    `resname` and `membrane_interacting` columns: resname for per-residue-type
    parameter lookup, and membrane_interacting as a boolean mask selecting
    which residues are scored. All other CSV columns (mean_dist_A,
    frac_inserted, etc.) are the reference MD statistics that were used to
    derive that mask upstream and are not consumed here.

    At every guidance step this evaluates the depth-dependent "Ez-potential"
    of Senes A, Chadi DC, Law PB, Walters RF, Nanda V, DeGrado WF, "Ez, A
    Depth-Dependent Potential for Assessing the Energies of Insertion of
    Amino Acid Side-Chains into Membranes...", J Mol Biol 2007 (366:436-448),
    Table 1, at the LIVE depth of each Ca (computed from the current
    Tweedie-estimated ca_pos_nm), not at a fixed target from the CSV.

    The paper uses two different functional forms depending on residue type,
    both applied to z as the ABSOLUTE distance (Angstrom) from the bilayer
    midplane (the paper treats both membrane faces symmetrically):

      * 17 residues (Ala, Asp, Glu, Phe, Gly, His, Ile, Lys, Leu, Met, Asn,
        Pro, Gln, Arg, Ser, Thr, Val) - sigmoidal form (Eq. 2):

            dE_z = dE0 / (1 + (z / Zmid) ** n)

      * Trp, Tyr - Gaussian form (Eq. 3), since their propensity peaks in
        the headgroup region rather than decaying monotonically:

            dE_z = dEmin * exp( -(z - Zmin)^2 / (2 * sigma^2) )

      * Cys - the paper found a scattered distribution and used a flat
        zero-energy curve, so Cys always contributes 0.

    Only residues flagged membrane_interacting == True in the CSV
    contribute to the summed guidance energy (only_interacting=True,
    the default).
    """

    # --- Table 1, functional form 1 (Eq. 2, sigmoidal): resname -> (dE0, Zmid, n) ---
    _SIGMOID_PARAMS = {
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

    # --- Table 1, functional form 2 (Eq. 3, Gaussian): resname -> (dEmin, Zmin, sigma) ---
    _GAUSSIAN_PARAMS = {
        "TRP": (-0.85, 11.65, 7.20),
        "TYR": (-0.42, 13.04, 6.20),
    }

    # Cys was not fit (scattered distribution) - flat zero energy curve.
    _ZERO_RESIDUES = {"CYS"}

    # CHARMM-style protonation-state / alternate resnames -> canonical 3-letter code
    _RESNAME_ALIASES = {
        "HSD": "HIS", "HSE": "HIS", "HSP": "HIS",
        "HID": "HIS", "HIE": "HIS", "HIP": "HIS",
        "CYX": "CYS", "CYM": "CYS",
        "ASH": "ASP", "GLH": "GLU", "LYN": "LYS",
    }
    _ONE_TO_THREE = {
        "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
        "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
        "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
        "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
    }

    @classmethod
    def _normalize_resname(cls, name):
        name = name.strip().upper()
        if len(name) == 1:
            name = cls._ONE_TO_THREE.get(name, name)
        return cls._RESNAME_ALIASES.get(name, name)

    def __init__(
        self,
        csv_path: str,
        membrane_axis: int = 2,
        membrane_center: float = 0.0,
        only_interacting: bool = True,
    ):
        """
        csv_path:         Path to the per-residue membrane CSV. Row order is
                           assumed to match the Ca residue ordering in
                           ca_pos_nm unless `sequence` is passed explicitly
                           to __call__.
        membrane_axis:     Index (0/1/2) of the coordinate axis treated as
                           the membrane normal. Default 2 (z).
        membrane_center:   Bilayer midplane position along membrane_axis, in
                           Angstrom, in the same frame as ca_pos_nm after the
                           nm->Angstrom conversion below. Default 0.0.
        only_interacting:  If True (default), only residues flagged
                           membrane_interacting == True in the CSV
                           contribute to the summed energy.
        """

        
        self.membrane_axis = membrane_axis
        self.membrane_center = membrane_center
        self.only_interacting = only_interacting

        resnames, interacting = [], []
        with open(csv_path, newline="") as f:
            rows = sorted(csv.DictReader(f), key=lambda r: int(r["resid"]))
        for row in rows:
            resnames.append(self._normalize_resname(row["resname"]))
            interacting.append(row["membrane_interacting"].strip().lower() == "true")
        self.residue_types = resnames
        self._csv_interacting_mask = torch.tensor(interacting, dtype=torch.bool)

        (
            self._default_dE0,
            self._default_zmid,
            self._default_n,
            self._default_is_gaussian,
            self._default_is_known,
        ) = self._build_param_tensors(self.residue_types)
        
        # New addition for dtype mismatches.

        has_ref, ref_xyz = [], []
        for row in rows:
            has_ref.append(row["has_ref"].strip().lower() == "true")
            ref_xyz.append([float(row["ref_x"]), float(row["ref_y"]), float(row["ref_z"])])
        has_ref = torch.tensor(has_ref, dtype=torch.bool)
        self._fit_idx = torch.nonzero(has_ref).squeeze(-1)
        self._ref_fit_A = torch.tensor(ref_xyz, dtype=torch.float64)[has_ref]
        if len(self._fit_idx) < 3:
            raise ValueError("Need at least 3 residues with a reference for Kabsch.")

    def _build_param_tensors(self, resnames):

        n_res = len(resnames)
        # p1 = E0 (sigmoid) or Emin (gaussian); p2 = Zmid or Zmin; p3 = n or sigma.
        p1 = torch.zeros(n_res)
        p2 = torch.zeros(n_res)
        p3 = torch.ones(n_res)  # avoid div-by-zero for zero/unknown residues
        is_gaussian = torch.zeros(n_res, dtype=torch.bool)
        is_known = torch.zeros(n_res, dtype=torch.bool)
        unknown = set()
        for i, name in enumerate(resnames):
            if name in self._ZERO_RESIDUES:
                is_known[i] = True  # known, deliberately zero (Cys)
                continue
            if name in self._GAUSSIAN_PARAMS:
                p1[i], p2[i], p3[i] = self._GAUSSIAN_PARAMS[name]
                is_gaussian[i] = True
                is_known[i] = True
            elif name in self._SIGMOID_PARAMS:
                p1[i], p2[i], p3[i] = self._SIGMOID_PARAMS[name]
                is_known[i] = True
            else:
                unknown.add(name)
        if unknown:
            warnings.warn(
                f"No Ez-potential parameters for residue type(s) {sorted(unknown)}; "
                "these residues will contribute zero energy."
            )
        return p1, p2, p3, is_gaussian, is_known

    def _ez_energy(self, z_abs, p1, p2, p3, is_gaussian, is_known):
        sigmoid_energy = p1 / (1.0 + (z_abs / p2) ** p3)
        gaussian_energy = p1 * torch.exp(-((z_abs - p2) ** 2) / (2.0 * p3 ** 2))
        energy = torch.where(is_gaussian, gaussian_energy, sigmoid_energy)
        return energy * is_known  # unknown/zero-form residues (Cys, unrecognized) -> 0

    def __call__(self, ca_pos_nm: torch.Tensor, *, t=None, sequence=None) -> torch.Tensor:
        """
        This takes the Tweedie estimate, computes each Ca's live depth along
        the membrane normal, and evaluates the Senes et al. 2007 Ez-potential
        per residue type, summed over CSV-flagged membrane_interacting
        residues, to produce the Custom_Potential energy score.

        ca_pos_nm: (..., n_res, 3) tensor of Ca coordinates in nm.
        sequence:  Optional override of residue identities (list/tuple of
                   1- or 3-letter codes, length n_res). If provided, the
                   membrane_interacting mask still comes from the CSV
                   (assumed aligned to the same residue ordering); only the
                   per-residue-type Ez parameters are taken from `sequence`.
        """
        n_res = ca_pos_nm.shape[-2]
        if len(self.residue_types) != n_res:
            raise ValueError(
                f"CSV has {len(self.residue_types)} residues but ca_pos_nm "
                f"has {n_res}; residue ordering must match."
            )

        if sequence is not None:
            if len(sequence) != n_res:
                raise ValueError(
                    f"sequence has {len(sequence)} residues but ca_pos_nm "
                    f"has {n_res}."
                )
            resnames = [self._normalize_resname(s) for s in sequence]
            p1, p2, p3, is_gaussian, is_known = self._build_param_tensors(resnames)
        else:
            p1, p2, p3 = self._default_dE0, self._default_zmid, self._default_n
            is_gaussian, is_known = self._default_is_gaussian, self._default_is_known

        device, dtype = ca_pos_nm.device, ca_pos_nm.dtype
        p1 = p1.to(device=device, dtype=dtype)
        p2 = p2.to(device=device, dtype=dtype)
        p3 = p3.to(device=device, dtype=dtype)
        is_gaussian = is_gaussian.to(device)
        is_known = is_known.to(device=device, dtype=dtype)

        # nm -> Angstrom (Ez parameters are tabulated in Angstrom); z is the
        # ABSOLUTE distance from the bilayer midplane (both
        # membrane faces are treated symmetrically).
        # z_ang = ca_pos_nm[..., self.membrane_axis] * 10.0
        # z_abs = torch.abs(z_ang - self.membrane_center)  # (..., n_res)

        # The above z calculation has to be exchanged with the distance from the OPM-derived membrane plane.
        # It has been noted that the OPM algorithm assumes the membrane-plane to be parallel to the x-y plane at z = -15.40. Assuming that to be the case, for all OPM entries,
        # I have hardcoded the membrane distance to be the difference between the z-axis coordinate and -15.40.
        # np_ca_pos_nm = numpy_from_torch(ca_pos_nm)      # finding a function which does the conversion to numpy from torch. 
        # np_ref_pos_nm = numpy_from_torch(ref_pos_nm)
        # rot, tran, _ = kabsch_rot_tran(np_ca_pos_nm, np_ref_pos_nm)   
        # ca_pos_nm = torch_from_numpy(np_ca_pos_nm)              # finding a function which does the conversion from numpy to torch.


        # rot, tran, _ = kabsch_rot_tran(numpy_from_torch(x[fit_idx]), numpy_from_torch(ref_fit_A))
        # rot  = torch_from_numpy(rot,  x)
        # tran = torch_from_numpy(tran, x)
        # x_aligned = x @ rot + tran 

        # x0_aligned = ca_pos_nm @ rot + tran

        # z_coords  = x0_aligned[..., self.membrane_axis] * 10.0
        # z_abs = torch.abs(z_coords - (-15.40))
        ca_A = ca_pos_nm * 10.0                                  # nm -> Angstrom
        batch_shape = ca_A.shape[:-2]
        ca_A = ca_A.reshape(-1, n_res, 3)                        # (num_graphs, n_res, 3)
        fit_idx = self._fit_idx.to(device)
        ref_fit_A = self._ref_fit_A.to(device=device, dtype=dtype)

        z_list = []
        for g in range(ca_A.shape[0]):
            x = ca_A[g]                                          # (n_res, 3)
            rot, tran, _ = kabsch_rot_tran(numpy_from_torch(x[fit_idx]), numpy_from_torch(ref_fit_A))
            rot = torch_from_numpy(rot, x)
            tran = torch_from_numpy(tran, x)
            x_aligned = x @ rot + tran                           # move ALL residues
            z_list.append(x_aligned[:, self.membrane_axis])
        z = torch.stack(z_list).reshape(*batch_shape, n_res)
        z_abs = torch.abs(z - self.membrane_center)              # midplane = 0
        
        per_res_energy = self._ez_energy(z_abs, p1, p2, p3, is_gaussian, is_known)

        if self.only_interacting:
            mask = self._csv_interacting_mask.to(device)
        else:
            mask = torch.ones(n_res, dtype=torch.bool, device=device)
        per_res_energy = per_res_energy * mask

        return per_res_energy.sum(dim=-1)

    # (Require dummy implementations of base class abstract methods)
    def loss_fn(self, x):
        # __call__ already returns the final scalar Ez guidance energy, so
        # no additional per-element transform is needed here.
        return x

    def energy_from_cv(self, cv, t=None):
        """
        cv: precomputed per-residue ABSOLUTE depths from the bilayer
        midplane in Angstrom (i.e. already |z - membrane_center|), shape
        (..., n_res), aligned to self.residue_types order.
        """
        device, dtype = cv.device, cv.dtype
        p1 = self._default_dE0.to(device=device, dtype=dtype)
        p2 = self._default_zmid.to(device=device, dtype=dtype)
        p3 = self._default_n.to(device=device, dtype=dtype)
        is_gaussian = self._default_is_gaussian.to(device)
        is_known = self._default_is_known.to(device=device, dtype=dtype)

        per_res_energy = self._ez_energy(cv, p1, p2, p3, is_gaussian, is_known)
        if self.only_interacting:
            mask = self._csv_interacting_mask.to(device)
        else:
            mask = torch.ones(cv.shape[-1], dtype=torch.bool, device=device)
        return (per_res_energy * mask).sum(dim=-1)

def numpy_from_torch(t):
    return t.detach().cpu().double().numpy()

def torch_from_numpy(a, like):
    return torch.as_tensor(a, dtype=like.dtype, device=like.device)
