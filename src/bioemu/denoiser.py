# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import logging
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
from torch_geometric.data.batch import Batch
from tqdm.auto import tqdm

from bioemu.chemgraph import ChemGraph
from bioemu.sde_lib import SDE, CosineVPSDE
from bioemu.so3_sde import SO3SDE, apply_rotvec_to_rotmat

logger = logging.getLogger(__name__)


class EulerMaruyamaPredictor:
    """Euler-Maruyama predictor."""

    def __init__(
        self,
        *,
        corruption: SDE,
        noise_weight: float = 1.0,
        marginal_concentration_factor: float = 1.0,
    ):
        """
        Args:
            noise_weight: A scalar factor applied to the noise during each update. The parameter controls the stochasticity of the integrator. A value of 1.0 is the
            standard Euler Maruyama integration scheme whilst a value of 0.0 is the probability flow ODE.
            marginal_concentration_factor: A scalar factor that controls the concentration of the sampled data distribution. The sampler targets p(x)^{MCF} where p(x)
            is the data distribution. A value of 1.0 is the standard Euler Maruyama / probability flow ODE integration.

            See feynman/projects/diffusion/sampling/samplers_readme.md for more details.

        """
        self.corruption = corruption
        self.noise_weight = noise_weight
        self.marginal_concentration_factor = marginal_concentration_factor

    def reverse_drift_and_diffusion(
        self, *, x: torch.Tensor, t: torch.Tensor, batch_idx: torch.LongTensor, score: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:

        score_weight = 0.5 * self.marginal_concentration_factor * (1 + self.noise_weight**2)
        drift, diffusion = self.corruption.sde(x=x, t=t, batch_idx=batch_idx)
        drift = drift - diffusion**2 * score * score_weight
        return drift, diffusion

    def update_given_drift_and_diffusion(
        self,
        *,
        x: torch.Tensor,
        dt: torch.Tensor,
        drift: torch.Tensor,
        diffusion: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.randn_like(drift)

        # Update to next step using either special update for SDEs on SO(3) or standard update.
        if isinstance(self.corruption, SO3SDE):
            mean = apply_rotvec_to_rotmat(x, drift * dt, tol=self.corruption.tol)
            sample = apply_rotvec_to_rotmat(
                mean,
                self.noise_weight * diffusion * torch.sqrt(dt.abs()) * z,
                tol=self.corruption.tol,
            )
        else:
            mean = x + drift * dt
            sample = mean + self.noise_weight * diffusion * torch.sqrt(dt.abs()) * z
        return sample, mean

    def update_given_score(
        self,
        *,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        batch_idx: torch.LongTensor,
        score: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # Set up different coefficients and terms.
        drift, diffusion = self.reverse_drift_and_diffusion(
            x=x, t=t, batch_idx=batch_idx, score=score
        )

        # Update to next step using either special update for SDEs on SO(3) or standard update.
        return self.update_given_drift_and_diffusion(
            x=x,
            dt=dt,
            drift=drift,
            diffusion=diffusion,
        )

    def forward_sde_step(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        batch_idx: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update to next step using either special update for SDEs on SO(3) or standard update.
        Handles both SO(3) and Euclidean updates."""

        drift, diffusion = self.corruption.sde(x=x, t=t, batch_idx=batch_idx)
        # Update to next step using either special update for SDEs on SO(3) or standard update.
        return self.update_given_drift_and_diffusion(x=x, dt=dt, drift=drift, diffusion=diffusion)


def get_score(
    batch: ChemGraph, sdes: dict[str, SDE], score_model: torch.nn.Module, t: torch.Tensor
) -> dict[str, torch.Tensor]:
    """
    Calculate predicted score for the batch.

    Args:
        batch: Batch of corrupted data.
        sdes: SDEs.
        score_model: Score model.  The score model is parametrized to predict a multiple of the score.
          This function converts the score model output to a score.
        t: Diffusion timestep. Shape [batch_size,]
    """
    tmp = score_model(batch, t)
    # Score is in axis angle representation [N,3] (vector is along axis of rotation, vector length
    # is rotation angle in radians).
    assert isinstance(sdes["node_orientations"], SO3SDE)
    node_orientations_score = (
        tmp["node_orientations"]
        * sdes["node_orientations"].get_score_scaling(t, batch_idx=batch.batch)[:, None]
    )

    # Score model is trained to predict score * std, so divide by std to get the score.
    _, pos_std = sdes["pos"].marginal_prob(
        x=torch.ones_like(tmp["pos"]),
        t=t,
        batch_idx=batch.batch,
    )
    pos_score = tmp["pos"] / pos_std

    return {"node_orientations": node_orientations_score, "pos": pos_score}


def heun_denoiser(
    *,
    sdes: dict[str, SDE],
    N: int,
    eps_t: float,
    max_t: float,
    device: torch.device,
    batch: Batch,
    score_model: torch.nn.Module,
    noise: float,
) -> ChemGraph:
    """Sample from prior and then denoise."""

    """
    Get x0(x_t) from score
    Create batch of samples with the same information
    """

    batch = batch.to(device)
    if isinstance(score_model, torch.nn.Module):
        # permits unit-testing with dummy model
        score_model = score_model.to(device)
    assert isinstance(sdes["node_orientations"], torch.nn.Module)  # shut up mypy
    sdes["node_orientations"] = sdes["node_orientations"].to(device)
    batch = batch.replace(
        pos=sdes["pos"].prior_sampling(batch.pos.shape, device=device),
        node_orientations=sdes["node_orientations"].prior_sampling(
            batch.node_orientations.shape, device=device
        ),
    )

    ts_min = 0.0
    ts_max = 1.0
    timesteps = torch.linspace(max_t, eps_t, N, device=device)
    dt = -torch.tensor((max_t - eps_t) / (N - 1)).to(device)
    fields = list(sdes.keys())
    predictors = {
        name: EulerMaruyamaPredictor(
            corruption=sde, noise_weight=0.0, marginal_concentration_factor=1.0
        )
        for name, sde in sdes.items()
    }
    noisers = {
        name: EulerMaruyamaPredictor(
            corruption=sde, noise_weight=1.0, marginal_concentration_factor=1.0
        )
        for name, sde in sdes.items()
    }
    batch_size = batch.num_graphs

    for i in range(N):
        # Set the timestep
        t = torch.full((batch_size,), timesteps[i], device=device)
        t_next = t + dt  # dt is negative; t_next is slightly less noisy than t.

        # Select temporarily increased noise level t_hat.
        # To be more general than Algorithm 2 in Karras et al. we select a time step between the
        # current and the previous t.
        t_hat = t - noise * dt if (i > 0 and t[0] > ts_min and t[0] < ts_max) else t

        # Apply noise.
        vals_hat = {}
        for field in fields:
            vals_hat[field] = noisers[field].forward_sde_step(
                x=batch[field], t=t, dt=(t_hat - t)[0], batch_idx=batch.batch
            )[0]
        batch_hat = batch.replace(**vals_hat)

        score = get_score(batch=batch_hat, t=t_hat, score_model=score_model, sdes=sdes)

        # First-order denoising step from t_hat to t_next.
        drift_hat = {}
        for field in fields:
            drift_hat[field], _ = predictors[field].reverse_drift_and_diffusion(
                x=batch_hat[field], t=t_hat, batch_idx=batch.batch, score=score[field]
            )

        for field in fields:
            batch[field], _ = predictors[field].update_given_drift_and_diffusion(
                x=batch_hat[field],
                dt=(t_next - t_hat)[0],
                drift=drift_hat[field],
                diffusion=0.0,
            )

        # Apply 2nd order correction.
        if t_next[0] > 0.0:
            score = get_score(batch=batch, t=t_next, score_model=score_model, sdes=sdes)

            drifts = {}
            avg_drift = {}
            for field in fields:
                drifts[field], _ = predictors[field].reverse_drift_and_diffusion(
                    x=batch[field], t=t_next, batch_idx=batch.batch, score=score[field]
                )

                avg_drift[field] = (drifts[field] + drift_hat[field]) / 2
            for field in fields:
                sample, _ = predictors[field].update_given_drift_and_diffusion(
                    x=batch_hat[field],
                    dt=(t_next - t_hat)[0],
                    drift=avg_drift[field],
                    diffusion=1.0,
                )
                batch[field] = sample

    return batch


def _t_from_lambda(sde: CosineVPSDE, lambda_t: torch.Tensor) -> torch.Tensor:
    """
    Used for DPMsolver. https://arxiv.org/abs/2206.00927 Appendix Section D.4
    """
    f_lambda = -1 / 2 * torch.log(torch.exp(-2 * lambda_t) + 1)
    exponent = f_lambda + torch.log(torch.cos(torch.tensor(np.pi * sde.s / 2 / (1 + sde.s))))
    t_lambda = 2 * (1 + sde.s) / np.pi * torch.acos(torch.exp(exponent)) - sde.s
    return t_lambda


@dataclass
class DPMCoefficients:
    """Coefficients for DPM-Solver++ step."""

    alpha_t: torch.Tensor
    sigma_t: torch.Tensor
    alpha_t_next: torch.Tensor
    sigma_t_next: torch.Tensor
    alpha_t_lambda: torch.Tensor
    sigma_t_lambda: torch.Tensor
    h_t: torch.Tensor  # lambda_t_next - lambda_t
    t_lambda: torch.Tensor  # midpoint time


@dataclass
class GuidedScore:
    """Guided score with optional reward information."""

    pos: torch.Tensor
    so3: torch.Tensor
    reward: torch.Tensor
    reward_grad_t: torch.Tensor  # gradient of reward w.r.t. time (for FKC weights)
    x0: torch.Tensor
    raw_score: dict[str, torch.Tensor]  # original unguided scores


def _get_dpm_coefficients(
    pos_sde: CosineVPSDE,
    batch_pos: torch.Tensor,
    t: torch.Tensor,
    t_next: torch.Tensor,
    batch_idx: torch.LongTensor,
) -> DPMCoefficients:
    """Compute all DPM-Solver coefficients for a step from t to t_next."""
    alpha_t, sigma_t = pos_sde.mean_coeff_and_std(x=batch_pos, t=t, batch_idx=batch_idx)
    lambda_t = torch.log(alpha_t / sigma_t)

    alpha_t_next, sigma_t_next = pos_sde.mean_coeff_and_std(
        x=batch_pos, t=t_next, batch_idx=batch_idx
    )
    lambda_t_next = torch.log(alpha_t_next / sigma_t_next)

    h_t = lambda_t_next - lambda_t

    # Compute midpoint time t_lambda
    lambda_t_middle = (lambda_t + lambda_t_next) / 2
    t_lambda = _t_from_lambda(sde=pos_sde, lambda_t=lambda_t_middle)
    t_lambda = torch.full_like(t, t_lambda[0][0])

    alpha_t_lambda, sigma_t_lambda = pos_sde.mean_coeff_and_std(
        x=batch_pos, t=t_lambda, batch_idx=batch_idx
    )

    return DPMCoefficients(
        alpha_t=alpha_t,
        sigma_t=sigma_t,
        alpha_t_next=alpha_t_next,
        sigma_t_next=sigma_t_next,
        alpha_t_lambda=alpha_t_lambda,
        sigma_t_lambda=sigma_t_lambda,
        h_t=h_t,
        t_lambda=t_lambda,
    )


def _so3_step(
    so3_sde: SO3SDE,
    node_orientations: torch.Tensor,
    score: torch.Tensor,
    t: torch.Tensor,
    dt: torch.Tensor,
    batch_idx: torch.LongTensor,
    noise_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unified SO3 step combining drift computation and update.

    Args:
        so3_sde: The SO3 SDE for orientations.
        node_orientations: Current orientations (N, 3, 3).
        score: Score at current time (N, 3).
        t: Current time.
        dt: Time step (negative for reverse).
        batch_idx: Batch indices.
        noise_weight: Scale for stochastic noise.

    Returns:
        (sample, mean): Updated orientations and deterministic mean.
    """
    predictor = EulerMaruyamaPredictor(
        corruption=so3_sde, noise_weight=noise_weight, marginal_concentration_factor=1.0
    )

    drift, diffusion = predictor.reverse_drift_and_diffusion(
        x=node_orientations, score=score, t=t, batch_idx=batch_idx
    )

    sample, mean = predictor.update_given_drift_and_diffusion(
        x=node_orientations, drift=drift, diffusion=diffusion, dt=dt
    )

    return sample, mean


def _predict_midpoint(
    batch: Batch,
    coeffs: DPMCoefficients,
    score_pos: torch.Tensor,
    score_so3: torch.Tensor,
    so3_sde: SO3SDE,
    t: torch.Tensor,
    batch_idx: torch.LongTensor,
) -> Batch:
    """First-order prediction step from t to t_lambda (midpoint).

    Updates both pos and node_orientations to the intermediate time.
    No noise is applied (deterministic prediction to midpoint).

    Args:
        score_pos: Position score to use (can be guided or scaled unsteered).
        score_so3: SO3 score to use.
    """
    # Position update (DPM-Solver first-order to midpoint)
    u = (
        coeffs.alpha_t_lambda / coeffs.alpha_t * batch.pos
        + coeffs.sigma_t_lambda * coeffs.sigma_t * (torch.exp(coeffs.h_t / 2) - 1) * score_pos
    )

    # SO3 update (drift only, no noise for midpoint prediction)
    dt_lambda = coeffs.t_lambda[0] - t[0]
    so3_sample, _ = _so3_step(
        so3_sde=so3_sde,
        node_orientations=batch.node_orientations,
        score=score_so3,
        t=t,
        dt=dt_lambda,
        batch_idx=batch_idx,
        noise_weight=0.0,
    )

    return batch.replace(pos=u, node_orientations=so3_sample)


def second_order_step_dpmsolver_plusplus(
    batch: Batch,
    coeffs: DPMCoefficients,
    scaled_score_pos_t: torch.Tensor,
    scaled_score_pos_lambda: torch.Tensor,
    score_so3_t: torch.Tensor,
    score_so3_lambda: torch.Tensor,
    so3_sde: SO3SDE,
    t: torch.Tensor,
    t_next: torch.Tensor,
    batch_idx: torch.LongTensor,
    noise_weight: float,
    pos_noise: torch.Tensor | None = None,
) -> tuple[Batch, torch.Tensor]:
    """Second-order DPM-Solver++ update from t to t_next.

    Uses scores at both t and t_lambda for higher accuracy.
    Position scores should be PRE-SCALED by the caller:
        - For FKC: use (1 + a^2) / 2 * score + (a^2) / 2 * reward_grad

    Args:
        scaled_score_pos_t: Pre-scaled position score at time t.
        scaled_score_pos_lambda: Pre-scaled position score at midpoint t_lambda.
        score_so3_t: SO3 score at time t (no scaling).
        score_so3_lambda: SO3 score at midpoint t_lambda (no scaling).
        noise_weight: Scale for stochastic noise.
        pos_noise: Optional pre-generated position noise.

    Returns:
        (batch_next, pos_noise): Updated batch and position noise used.
    """
    # Generate noise if not provided
    if pos_noise is None:
        pos_noise = torch.randn_like(batch.pos)

    # Position update (DPM-Solver++ 2nd order)
    pos_next = (
        coeffs.alpha_t_next / coeffs.alpha_t * batch.pos
        + 2
        * coeffs.sigma_t_next
        * coeffs.sigma_t
        * (torch.exp(coeffs.h_t) - 1)
        * scaled_score_pos_t
        + 2
        * coeffs.sigma_t_next
        * (torch.exp(coeffs.h_t) - 1)
        * torch.exp(-coeffs.h_t / 2)
        * (coeffs.sigma_t_lambda * scaled_score_pos_lambda - coeffs.sigma_t * scaled_score_pos_t)
        + noise_weight * coeffs.sigma_t_next * torch.sqrt(torch.exp(2 * coeffs.h_t) - 1) * pos_noise
    )

    # SO3 update with 2nd-order score extrapolation (deterministic — noise_weight=0)
    dt_hat = t_next[0] - t[0]
    dt_lambda = coeffs.t_lambda[0] - t[0]
    score_correction = 0.5 * (score_so3_lambda - score_so3_t) / dt_lambda * dt_hat

    so3_sample, _ = _so3_step(
        so3_sde=so3_sde,
        node_orientations=batch.node_orientations,
        score=score_so3_lambda + score_correction,
        t=coeffs.t_lambda,
        dt=dt_hat,
        batch_idx=batch_idx,
        noise_weight=0.0,
    )

    return batch.replace(pos=pos_next, node_orientations=so3_sample), pos_noise


def second_order_step_dpmsolver(
    batch: Batch,
    coeffs: DPMCoefficients,
    score_pos_lambda: torch.Tensor,
    score_so3_t: torch.Tensor,
    score_so3_lambda: torch.Tensor,
    so3_sde: SO3SDE,
    t: torch.Tensor,
    t_next: torch.Tensor,
    batch_idx: torch.LongTensor,
) -> Batch:
    """DPM-Solver 2nd-order ODE step using midpoint score only (no noise).

    This implements the DPM-Solver-2 update (Algorithm 1 in https://arxiv.org/abs/2206.00927),
    which uses only the score evaluated at the midpoint t_lambda for the position update.
    Used by the unsteered `dpm_solver` loop.

    For the DPM-Solver++ SDE variant (two scores + noise), see
    `second_order_step_dpmsolver_plusplus`.

    Args:
        score_pos_lambda: Position score at midpoint t_lambda (unscaled).
        score_so3_t: SO3 score at time t.
        score_so3_lambda: SO3 score at midpoint t_lambda.
    """
    # Position: midpoint-only formula
    pos_next = (
        coeffs.alpha_t_next / coeffs.alpha_t * batch.pos
        + coeffs.sigma_t_next
        * coeffs.sigma_t_lambda
        * (torch.exp(coeffs.h_t) - 1)
        * score_pos_lambda
    )

    # SO3: 2nd-order score extrapolation (same as dpmsolver_plusplus, deterministic)
    dt_hat = t_next[0] - t[0]
    dt_lambda = coeffs.t_lambda[0] - t[0]
    score_correction = 0.5 * (score_so3_lambda - score_so3_t) / dt_lambda * dt_hat

    so3_sample, _ = _so3_step(
        so3_sde=so3_sde,
        node_orientations=batch.node_orientations,
        score=score_so3_lambda + score_correction,
        t=coeffs.t_lambda,
        dt=dt_hat,
        batch_idx=batch_idx,
        noise_weight=0.0,
    )

    return batch.replace(pos=pos_next, node_orientations=so3_sample)
def dpm_solver(
    sdes: dict[str, SDE],
    batch: Batch,
    N: int,
    score_model: torch.nn.Module,
    max_t: float,
    eps_t: float,
    device: torch.device,
    record_grad_steps: set[int] = set(),
    noise: float = 0.0,
) -> Batch:
    """
    Implements the DPM solver for the VPSDE, with the Cosine noise schedule.
    Following this paper: https://arxiv.org/abs/2206.00927 Algorithm 1 DPM-Solver-2.

    This is the unsteered denoiser. For steered sampling (FKC/SMC), use
    dpm_solver_fkc or dpm_solver_smc from the steering package. For membrane
    guidance, use membrane_guided_dpm_solver below.
    """
    grad_is_enabled = torch.is_grad_enabled()
    assert isinstance(batch, Batch)
    assert max_t < 1.0

    batch = batch.to(device)

    if isinstance(score_model, torch.nn.Module):
        score_model = score_model.to(device)
    pos_sde = sdes["pos"]
    assert isinstance(pos_sde, CosineVPSDE)

    batch = batch.replace(
        pos=sdes["pos"].prior_sampling(batch.pos.shape, device=device),
        node_orientations=sdes["node_orientations"].prior_sampling(
            batch.node_orientations.shape, device=device
        ),
    )
    batch = cast(ChemGraph, batch)

    so3_sde = sdes["node_orientations"]
    assert isinstance(so3_sde, SO3SDE)
    so3_sde.to(device)

    timesteps = torch.linspace(max_t, eps_t, N, device=device)
    dt = -torch.tensor((max_t - eps_t) / (N - 1)).to(device)
    ts_min = 0.0
    ts_max = 1.0
    fields = list(sdes.keys())
    noisers = {
        name: EulerMaruyamaPredictor(
            corruption=sde, noise_weight=1.0, marginal_concentration_factor=1.0
        )
        for name, sde in sdes.items()
    }

    for i in tqdm(range(N - 1), position=1, desc="Denoising: ", ncols=0, leave=False):
        t = torch.full((batch.num_graphs,), timesteps[i], device=device)
        t_hat = t - noise * dt if (i > 0 and t[0] > ts_min and t[0] < ts_max) else t
        t_next = t + dt

        # Pre-step noise injection (Karras/Heun style)
        vals_hat = {}
        for field in fields:
            vals_hat[field] = noisers[field].forward_sde_step(
                x=batch[field], t=t, dt=(t_hat - t)[0], batch_idx=batch.batch
            )[0]
        batch_hat = batch.replace(**vals_hat)

        # Evaluate score at (possibly noised) state
        with torch.set_grad_enabled(grad_is_enabled and (i in record_grad_steps)):
            score = get_score(batch=batch_hat, t=t_hat, score_model=score_model, sdes=sdes)

        batch_idx = batch_hat.batch

        # Coefficients + midpoint prediction
        coeffs = _get_dpm_coefficients(pos_sde, batch_hat.pos, t_hat, t_next, batch_idx)

        batch_lambda = _predict_midpoint(
            batch=batch_hat,
            coeffs=coeffs,
            score_pos=score["pos"],
            score_so3=score["node_orientations"],
            so3_sde=so3_sde,
            t=t_hat,
            batch_idx=batch_idx,
        )

        # Correction step: evaluate score at midpoint
        with torch.set_grad_enabled(grad_is_enabled and (i in record_grad_steps)):
            score_lambda = get_score(
                batch=batch_lambda, t=coeffs.t_lambda, sdes=sdes, score_model=score_model
            )

        # DPM-Solver 2nd-order ODE step (midpoint-only formula)
        batch = second_order_step_dpmsolver(
            batch=batch_hat,
            coeffs=coeffs,
            score_pos_lambda=score_lambda["pos"],
            score_so3_t=score["node_orientations"],
            score_so3_lambda=score_lambda["node_orientations"],
            so3_sde=so3_sde,
            t=t_hat,
            t_next=t_next,
            batch_idx=batch_idx,
        )

    return batch


def _guided_pos_score(
    *,
    batch: Batch,
    t: torch.Tensor,
    score_model: torch.nn.Module,
    sdes: dict[str, SDE],
    potential,
    guidance_scale: float,
    backprop_through_model: bool,
    max_guidance_norm: float | None,
    max_guidance_ratio: float | None = None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Score of the tilted marginal p_t(x) * exp(-guidance_scale * E(x0_hat(x))) (DPS-style).

    x0_hat is the Tweedie estimate (x + sigma_t^2 * score) / alpha_t. The potential is
    evaluated on x0_hat (nm, shape [num_graphs, n_res, 3]) and its gradient w.r.t. x_t is
    *subtracted* from the score (potential = energy, lower is better; cf. upstream
    steering where reward = -potential).

    backprop_through_model=True differentiates through the score network (full DPS);
    False treats the score as constant, i.e. grad ≈ (1/alpha_t) * dE/dx0 (cheaper).

    Returns the guided scores (detached) and the per-graph energy of x0_hat.
    """
    pos_sde = sdes["pos"]
    with torch.enable_grad():
        pos = batch.pos.detach().requires_grad_(True)
        batch_g = batch.replace(pos=pos)
        if backprop_through_model:
            score = get_score(batch=batch_g, t=t, score_model=score_model, sdes=sdes)
            score_pos_for_x0 = score["pos"]
        else:
            with torch.no_grad():
                score = get_score(batch=batch_g, t=t, score_model=score_model, sdes=sdes)
            score_pos_for_x0 = score["pos"].detach()
        alpha_t, sigma_t = pos_sde.mean_coeff_and_std(x=pos, t=t, batch_idx=batch_g.batch)
        x0_hat = (pos + sigma_t**2 * score_pos_for_x0) / alpha_t
        num_graphs = batch_g.num_graphs
        x0_per_graph = x0_hat.view(num_graphs, -1, x0_hat.shape[-1])
        sequence = batch.sequence[0] if hasattr(batch, "sequence") else None
        energy = potential(x0_per_graph, t=t[0], sequence=sequence)
        assert energy.shape == (num_graphs,), f"potential returned shape {energy.shape}"
        (grad,) = torch.autograd.grad(energy.sum(), pos)

    guidance = -guidance_scale * grad
    if max_guidance_norm is not None:
        g = guidance.view(num_graphs, -1)
        norm = g.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        guidance = (g * torch.clamp(max_guidance_norm / norm, max=1.0)).view_as(guidance)
    if max_guidance_ratio is not None:
        # Cap the guidance at a fraction of the model's own score, per structure. This is
        # scale-free in t (score magnitudes grow as t -> 0), so one setting holds across the
        # whole trajectory, unlike an absolute norm cap. Without it a stiff potential (e.g. a
        # chain-break barrier on an already-broken chain) can dominate a single step and
        # destroy the structure.
        g = guidance.view(num_graphs, -1)
        s_norm = score["pos"].detach().view(num_graphs, -1).norm(dim=-1, keepdim=True)
        g_norm = g.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        guidance = (g * torch.clamp(max_guidance_ratio * s_norm / g_norm, max=1.0)).view_as(guidance)

    guided = {
        "pos": (score["pos"] + guidance).detach(),
        "node_orientations": score["node_orientations"].detach(),
    }
    return guided, energy.detach()


def membrane_guided_dpm_solver(
    sdes: dict[str, SDE],
    batch: Batch,
    N: int,
    score_model: torch.nn.Module,
    max_t: float,
    eps_t: float,
    device: torch.device,
    record_grad_steps: set[int] = set(),
    noise: float = 0.0,
    potential=None,
    guidance_scale: float = 1.0,
    guidance_t_max: float = 1.0,
    guidance_t_min: float = 0.0,
    guide_midpoint: bool = True,
    backprop_through_model: bool = True,
    max_guidance_norm: float | None = None,
    max_guidance_ratio: float | None = None,
    membrane_csv_path: str | None = None,
) -> Batch:
    """DPM-Solver-2 (same integrator as `dpm_solver`) with DPS-style gradient guidance on
    the translational (Cα position) score.

    With potential=None (and no membrane_csv_path) this is numerically identical to
    `dpm_solver`, which makes it the matched unguided baseline.

    Args:
        potential: A `bioemu.steering.Potential` (energy, lower = better), called as
            potential(x0_hat_nm[num_graphs, n_res, 3], t=..., sequence=...).
        guidance_scale: lambda in p_t(x) exp(-lambda E). For E in kcal/mol, lambda = 1/kT
            ≈ 1.69 mol/kcal at 298 K corresponds to a Boltzmann tilt.
        guidance_t_max / guidance_t_min: apply guidance only for t in [t_min, t_max].
        guide_midpoint: also guide the midpoint score. DPM-Solver-2's position update uses
            *only* the midpoint score, so without this the guidance barely affects samples.
        backprop_through_model: full DPS (True) or score-as-constant approximation (False).
        max_guidance_norm: optional per-structure clip on the guidance term's norm.
        max_guidance_ratio: optional per-structure clip expressed as a fraction of the model's
            own score norm (e.g. 0.5 = guidance may never exceed half the score). Preferred
            over max_guidance_norm because it is scale-free in t.
        membrane_csv_path: backward-compatible shortcut for
            potential=MembranePotentialPlaceholder(csv_path) (Kabsch frame).
    """
    if potential is None and membrane_csv_path is not None:
        from bioemu.steering.custom_potentials import MembranePotentialPlaceholder

        potential = MembranePotentialPlaceholder(csv_path=membrane_csv_path)

    grad_is_enabled = torch.is_grad_enabled()
    assert isinstance(batch, Batch)
    assert max_t < 1.0

    batch = batch.to(device)
    if isinstance(score_model, torch.nn.Module):
        score_model = score_model.to(device)
    pos_sde = sdes["pos"]
    assert isinstance(pos_sde, CosineVPSDE)

    batch = batch.replace(
        pos=sdes["pos"].prior_sampling(batch.pos.shape, device=device),
        node_orientations=sdes["node_orientations"].prior_sampling(
            batch.node_orientations.shape, device=device
        ),
    )
    batch = cast(ChemGraph, batch)

    so3_sde = sdes["node_orientations"]
    assert isinstance(so3_sde, SO3SDE)
    so3_sde.to(device)

    timesteps = torch.linspace(max_t, eps_t, N, device=device)
    dt = -torch.tensor((max_t - eps_t) / (N - 1)).to(device)
    ts_min, ts_max = 0.0, 1.0
    fields = list(sdes.keys())
    noisers = {
        name: EulerMaruyamaPredictor(
            corruption=sde, noise_weight=1.0, marginal_concentration_factor=1.0
        )
        for name, sde in sdes.items()
    }
    guide_kwargs = dict(
        score_model=score_model,
        sdes=sdes,
        potential=potential,
        guidance_scale=guidance_scale,
        backprop_through_model=backprop_through_model,
        max_guidance_norm=max_guidance_norm,
        max_guidance_ratio=max_guidance_ratio,
    )

    desc = "Guided denoising: " if potential is not None else "Denoising: "
    for i in tqdm(range(N - 1), position=1, desc=desc, ncols=0, leave=False):
        t = torch.full((batch.num_graphs,), timesteps[i], device=device)
        t_hat = t - noise * dt if (i > 0 and t[0] > ts_min and t[0] < ts_max) else t
        t_next = t + dt

        vals_hat = {}
        for field in fields:
            vals_hat[field] = noisers[field].forward_sde_step(
                x=batch[field], t=t, dt=(t_hat - t)[0], batch_idx=batch.batch
            )[0]
        batch_hat = batch.replace(**vals_hat)
        batch_idx = batch_hat.batch

        guide_now = potential is not None and guidance_t_min <= float(t_hat[0]) <= guidance_t_max

        if guide_now:
            score, energy = _guided_pos_score(batch=batch_hat, t=t_hat, **guide_kwargs)
            if i % 10 == 0:
                logger.debug(f"step {i} t={float(t_hat[0]):.3f} mean E(x0_hat)={energy.mean():.3f}")
        else:
            with torch.set_grad_enabled(grad_is_enabled and (i in record_grad_steps)):
                score = get_score(batch=batch_hat, t=t_hat, score_model=score_model, sdes=sdes)

        coeffs = _get_dpm_coefficients(pos_sde, batch_hat.pos, t_hat, t_next, batch_idx)

        batch_lambda = _predict_midpoint(
            batch=batch_hat,
            coeffs=coeffs,
            score_pos=score["pos"],
            score_so3=score["node_orientations"],
            so3_sde=so3_sde,
            t=t_hat,
            batch_idx=batch_idx,
        )

        if guide_now and guide_midpoint:
            score_lambda, _ = _guided_pos_score(batch=batch_lambda, t=coeffs.t_lambda, **guide_kwargs)
        else:
            with torch.set_grad_enabled(grad_is_enabled and (i in record_grad_steps)):
                score_lambda = get_score(
                    batch=batch_lambda, t=coeffs.t_lambda, sdes=sdes, score_model=score_model
                )

        batch = second_order_step_dpmsolver(
            batch=batch_hat,
            coeffs=coeffs,
            score_pos_lambda=score_lambda["pos"],
            score_so3_t=score["node_orientations"],
            score_so3_lambda=score_lambda["node_orientations"],
            so3_sde=so3_sde,
            t=t_hat,
            t_next=t_next,
            batch_idx=batch_idx,
        )

    return batch
