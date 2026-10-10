
"""PPO-coefficient-aware conditional score centering.

This module consumes mean_logp from independent auxiliary actions drawn
from the actual behavior distribution at the SAME token prefix.
It does not construct that behavior distribution.
"""
import torch


def apply_sc_correction(pg_loss, log_prob, mean_logp, response_mask):
    if mean_logp is None:
        raise RuntimeError("SC enabled but auxiliary mean log-prob is missing.")
    if pg_loss.ndim != 0:
        raise ValueError("Expected scalar aggregated PPO loss.")
    if log_prob.shape != mean_logp.shape:
        raise ValueError("SC mean log-prob and PPO log-prob shapes differ.")
    if log_prob.shape != response_mask.shape:
        raise ValueError("SC response mask shape differs.")
    if not log_prob.requires_grad or not mean_logp.requires_grad:
        raise RuntimeError("SC requires differentiable current-policy log-probs.")

    mask = response_mask.bool()
    if not mask.any():
        raise ValueError("SC received a batch without assistant tokens.")
    if not torch.isfinite(mean_logp[mask]).all():
        raise FloatingPointError("Nonfinite SC auxiliary log-prob.")

    # Includes the actual PPO clipping and aggregation coefficients.
    coefficient = torch.autograd.grad(
        pg_loss,
        log_prob,
        retain_graph=True,
        create_graph=False,
        allow_unused=False,
    )[0].detach()

    if not torch.isfinite(coefficient).all():
        raise FloatingPointError("Nonfinite PPO coefficient.")
    if torch.count_nonzero(coefficient[~mask]).item() != 0:
        raise RuntimeError("Masked tokens have nonzero PPO coefficients.")

    # Do not multiply 0 by NaN/Inf on tool or padding positions.
    safe_mean = torch.where(mask, mean_logp, torch.zeros_like(mean_logp))
    zero_forward_term = safe_mean - safe_mean.detach()
    correction = -(coefficient * zero_forward_term).sum()
    corrected_loss = pg_loss + correction

    metrics = {
        "actor/sc_active": 1.0,
        "actor/sc_coefficient_abs_mean":
            coefficient[mask].abs().mean().item(),
        "actor/sc_coefficient_nonzero_fraction":
            (coefficient[mask] != 0).float().mean().item(),
        "actor/sc_aux_mean_logp":
            safe_mean[mask].detach().mean().item(),
        "actor/sc_forward_delta": correction.detach().item(),
    }
    return corrected_loss, metrics
