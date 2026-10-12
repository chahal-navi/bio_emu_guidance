"""
SO(3) rotational guidance for the BioEmu denoiser.

Extends the existing translational (position) guidance in
`membrane_guided_dpm_solver` to also steer `node_orientations`. Uses the
quaternion representation for the potential's forward pass and a Riemannian
projection to route the gradient back into the axis-angle score convention
used by `get_score` / `EulerMaruyamaPredictor`.

HOW TO INTEGRATE INTO denoiser.py
----------------------------------
1. Paste `matrix_to_quaternion`, `quaternion_to_matrix`, `vee`, and
   `so3_riemannian_gradient` in as free functions (e.g. just above
   `heun_denoiser`).
2. In `membrane_guided_dpm_solver`, replace everything from
   `with torch.enable_grad():` down to `# --- END OF GUIDANCE INJECTION ---`
   with a call to `guided_step_translation_and_rotation` (see the bottom of
   this file for the exact call site).
3. Extend `MembranePotentialPlaceholder.__call__` (and the `Potential` ABC in
   custom_potential.py) to accept an `orientations` kwarg. Right now it only
   takes `ca_pos_nm`, so until you (or Pratham) add that, `rot_grad` below
   will just come back as zero via `allow_unused=True`, which is safe, it
   just means rotational guidance is a no-op until the classifier actually
   uses orientation.
4. RUN `_check_so3_convention` once (see bottom) before trusting the sign of
   the rotational term. Whether the correct projection is "body" or "space"
   frame depends on how `apply_rotvec_to_rotmat` applies its rotvec, and I
   don't have that source, so this is not something to assume.
"""

import torch


# ---------------------------------------------------------------------------
# Quaternion <-> rotation matrix.
# Convention: unit quaternion (w, x, y, z), same as pytorch3d.transforms.
# If bioemu or RegFlow already ships quaternion utilities, prefer those for
# consistency with whatever the classifier was trained on; the only thing
# that matters is that guidance-time and training-time conventions match.
# Validate this against scipy.spatial.transform.Rotation before trusting it
# in a real run. Matrix/quaternion sign or branch bugs are silent, they don't
# error, they just quietly steer in the wrong direction.
# ---------------------------------------------------------------------------

def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """Rotation matrices (..., 3, 3) -> unit quaternions (..., 4), (w, x, y, z).

    Numerically stable "pick the largest candidate" method (Shepperd/Markley),
    the same family of algorithm pytorch3d uses. Differentiable.
    """
    m = matrix.reshape(*matrix.shape[:-2], 9)
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = m.unbind(-1)

    q_abs_sq = torch.stack(
        [
            1.0 + m00 + m11 + m22,
            1.0 + m00 - m11 - m22,
            1.0 - m00 + m11 - m22,
            1.0 - m00 - m11 + m22,
        ],
        dim=-1,
    ).clamp(min=0.0)
    q_abs = q_abs_sq.sqrt()  # (..., 4): proportional to |w|, |x|, |y|, |z|

    quat_candidates = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )  # (..., 4, 4)

    quat = quat_candidates / (2.0 * q_abs.clamp(min=1e-8)[..., None])
    best = q_abs.argmax(dim=-1)
    quat = torch.gather(quat, -2, best[..., None, None].expand(*best.shape, 1, 4))[..., 0, :]
    return quat / quat.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Unit quaternions (..., 4), (w, x, y, z) -> rotation matrices (..., 3, 3)."""
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    w, x, y, z = q.unbind(-1)
    R = torch.stack(
        [
            1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2),
        ],
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)
    return R


# ---------------------------------------------------------------------------
# SO(3) tangent-space projection.
# ---------------------------------------------------------------------------

def vee(skew: torch.Tensor) -> torch.Tensor:
    """Inverse of the hat map: (..., 3, 3) skew-symmetric -> (..., 3) vector."""
    return torch.stack([skew[..., 2, 1], skew[..., 0, 2], skew[..., 1, 0]], dim=-1)


def so3_riemannian_gradient(
    R: torch.Tensor, ambient_grad: torch.Tensor, frame: str = "body"
) -> torch.Tensor:
    """
    Project a Euclidean/ambient gradient dE/dR, shape (N, 3, 3), onto the
    SO(3) tangent space at R, and express it as a 3-vector in so(3), in the
    same axis-angle convention as score["node_orientations"].

    frame="body":  matches an update rule R_new = R @ Exp(v)
    frame="space": matches an update rule R_new = Exp(v) @ R

    Which one is correct depends on `apply_rotvec_to_rotmat`'s convention.
    Use `_check_so3_convention` below to determine this empirically rather
    than guessing.
    """
    Rt = R.transpose(-1, -2)
    if frame == "body":
        lie_alg = 0.5 * (Rt @ ambient_grad - ambient_grad.transpose(-1, -2) @ R)
    elif frame == "space":
        lie_alg = 0.5 * (ambient_grad @ Rt - R @ ambient_grad.transpose(-1, -2))
    else:
        raise ValueError(f"Unknown frame convention: {frame!r}")
    return vee(lie_alg)


@torch.no_grad()
def _check_so3_convention(R: torch.Tensor, apply_rotvec_to_rotmat, tol: float = 1e-7) -> str:
    """
    Empirically determine whether `apply_rotvec_to_rotmat` applies a tangent
    vector as a body-frame update (R @ Exp(v)) or a space-frame update
    (Exp(v) @ R). Run this once (e.g. on a random batch of orientations) and
    use the returned string as `so3_frame` in `so3_riemannian_gradient` and
    `guided_step_translation_and_rotation`.

    Usage:
        frame = _check_so3_convention(batch.node_orientations, apply_rotvec_to_rotmat)
    """
    v = torch.randn(R.shape[0], 3, dtype=R.dtype, device=R.device) * 1e-4
    R_pert = apply_rotvec_to_rotmat(R, v, tol=tol)

    def so3_exp(v):
        theta = v.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        k = v / theta
        K = torch.zeros(*v.shape[:-1], 3, 3, dtype=v.dtype, device=v.device)
        K[..., 0, 1], K[..., 0, 2] = -k[..., 2], k[..., 1]
        K[..., 1, 0], K[..., 1, 2] = k[..., 2], -k[..., 0]
        K[..., 2, 0], K[..., 2, 1] = -k[..., 1], k[..., 0]
        th = theta[..., None]
        eye = torch.eye(3, dtype=v.dtype, device=v.device).expand_as(K)
        return eye + torch.sin(th) * K + (1 - torch.cos(th)) * (K @ K)

    err_body = (R_pert - R @ so3_exp(v)).abs().max().item()
    err_space = (R_pert - so3_exp(v) @ R).abs().max().item()
    return "body" if err_body < err_space else "space"


# ---------------------------------------------------------------------------
# Joint guidance step: replaces the `with torch.enable_grad(): ...` block
# inside `membrane_guided_dpm_solver`.
# ---------------------------------------------------------------------------

def guided_step_translation_and_rotation(
    *,
    batch_hat,
    score_model,
    sdes: dict,
    pos_sde,
    so3_sde,
    t_hat: torch.Tensor,
    t_next: torch.Tensor,
    eps_t: float,
    batch_idx: torch.LongTensor,
    membrane_evaluator,
    get_score,
    _get_dpm_coefficients,
    _so3_step,
    eta_scale_pos: float = 1.5,
    eta_scale_rot: float = 1.5,
    so3_frame: str = "body",
):
    """
    Returns (score, batch_hat, coeffs). `coeffs` is returned because the
    caller (`membrane_guided_dpm_solver`) reuses it for the subsequent
    `_predict_midpoint` / `second_order_step_dpmsolver` calls, exactly as the
    original code did with the `coeffs` it computed inline.
    """
    with torch.enable_grad():
        batch_hat.pos.requires_grad_(True)
        batch_hat.node_orientations.requires_grad_(True)

        # One joint forward pass, both fields' scores, so cross-dependence
        # between rotation and translation through the shared GNN is
        # captured correctly by the single backward pass below.
        score = get_score(batch=batch_hat, t=t_hat, score_model=score_model, sdes=sdes)

        # Translation: closed-form Tweedie estimate (VP-SDE is linear-Gaussian).
        coeffs = _get_dpm_coefficients(pos_sde, batch_hat.pos, t_hat, t_next, batch_idx)
        x0_pos_hat = (batch_hat.pos + (coeffs.sigma_t**2) * score["pos"]) / coeffs.alpha_t

        # Rotation: no closed-form Tweedie for IGSO(3). Approximate x0 with a
        # single deterministic reverse step straight to eps_t. First-order,
        # exact only in the small-noise limit. Swap in something exact if
        # so3_sde exposes one.
        dt_to_x0 = eps_t - t_hat[0]
        x0_rot_hat, _ = _so3_step(
            so3_sde=so3_sde,
            node_orientations=batch_hat.node_orientations,
            score=score["node_orientations"],
            t=t_hat,
            dt=dt_to_x0,
            batch_idx=batch_idx,
            noise_weight=0.0,
        )
        x0_quat_hat = matrix_to_quaternion(x0_rot_hat)

        # Joint potential on the (translation, rotation) x0 estimate.
        # Requires MembranePotentialPlaceholder.__call__ to accept `orientations`.
        custom_pot_energy = membrane_evaluator(x0_pos_hat, orientations=x0_quat_hat)

        pos_grad, rot_grad = torch.autograd.grad(
            outputs=custom_pot_energy.sum(),
            inputs=[batch_hat.pos, batch_hat.node_orientations],
            allow_unused=True,
        )
        pos_grad = pos_grad if pos_grad is not None else torch.zeros_like(batch_hat.pos)
        rot_grad = (
            rot_grad
            if rot_grad is not None
            else torch.zeros_like(batch_hat.node_orientations)
        )

        # Project the ambient (3,3) gradient onto T_R SO(3), express as a
        # 3-vector in the same axis-angle convention as score["node_orientations"].
        rot_guidance = so3_riemannian_gradient(
            R=batch_hat.node_orientations, ambient_grad=rot_grad, frame=so3_frame
        )

        score["pos"] = score["pos"] + eta_scale_pos * pos_grad
        score["node_orientations"] = score["node_orientations"] + eta_scale_rot * rot_guidance

        batch_hat.pos = batch_hat.pos.detach()
        batch_hat.node_orientations = batch_hat.node_orientations.detach()
        score["pos"] = score["pos"].detach()
        score["node_orientations"] = score["node_orientations"].detach()

    return score, batch_hat, coeffs


# ---------------------------------------------------------------------------
# Exact call site to drop into membrane_guided_dpm_solver, replacing the old
# `with torch.enable_grad(): ... # --- END OF GUIDANCE INJECTION ---` block:
# ---------------------------------------------------------------------------
#
#   score, batch_hat, coeffs = guided_step_translation_and_rotation(
#       batch_hat=batch_hat,
#       score_model=score_model,
#       sdes=sdes,
#       pos_sde=pos_sde,
#       so3_sde=so3_sde,
#       t_hat=t_hat,
#       t_next=t_next,
#       eps_t=eps_t,
#       batch_idx=batch_idx,
#       membrane_evaluator=membrane_evaluator,
#       get_score=get_score,
#       _get_dpm_coefficients=_get_dpm_coefficients,
#       _so3_step=_so3_step,
#       so3_frame="body",  # set via _check_so3_convention first
#   )
#
#   # then continue exactly as before:
#   batch_lambda = _predict_midpoint(
#       batch=batch_hat, coeffs=coeffs, score_pos=score["pos"],
#       score_so3=score["node_orientations"], so3_sde=so3_sde,
#       t=t_hat, batch_idx=batch_idx,
#   )
#   ... (unchanged)
