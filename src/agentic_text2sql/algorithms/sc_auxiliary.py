
"""Current-policy auxiliary log-probs; q sampling is handled upstream."""
import torch
from torch.utils.checkpoint import checkpoint


def shifted_auxiliary_ids(aux_ids, offsets):
    """IDs at token j belong to logits at j-1; never cross sequences."""
    if aux_ids.ndim != 2 or aux_ids.shape[1] != 4:
        raise ValueError("Expected packed auxiliary IDs [N, 4].")
    if aux_ids.dtype != torch.long:
        raise TypeError("Auxiliary IDs must be int64.")
    if offsets.ndim != 1 or offsets.numel() < 2:
        raise ValueError("Invalid sequence offsets.")
    if offsets[0].item() != 0 or offsets[-1].item() != aux_ids.shape[0]:
        raise ValueError("Offsets do not cover packed auxiliary IDs.")
    if not (offsets[1:] > offsets[:-1]).all():
        raise ValueError("Empty or non-increasing sequence offsets.")
    if (aux_ids < -1).any():
        raise ValueError("Only -1 is allowed as an absent-ID sentinel.")

    present = aux_ids >= 0
    if (present.any(-1) != present.all(-1)).any():
        raise ValueError("Each prefix must contain all four IDs or four -1s.")

    shifted = torch.roll(aux_ids, shifts=-1, dims=0).clone()
    ends = offsets[1:].to(device=aux_ids.device, dtype=torch.long) - 1
    shifted[ends] = -1
    return shifted


def auxiliary_mean_logp_packed(logits, aux_ids, offsets, chunk_size=32):
    """Input logits already follow the Actor temperature convention."""
    if logits.ndim != 2 or logits.shape[0] != aux_ids.shape[0]:
        raise ValueError("Logits and auxiliary sequence lengths differ.")
    if logits.device != aux_ids.device:
        raise ValueError("Logits and auxiliary IDs must share a device.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    shifted = shifted_auxiliary_ids(aux_ids, offsets)
    valid = shifted[:, 0] >= 0
    positions = valid.nonzero(as_tuple=False).flatten()
    if positions.numel() == 0:
        return logits[:, 0] * 0.0
    if (shifted[positions] >= logits.shape[-1]).any():
        raise ValueError("Auxiliary ID exceeds vocabulary size.")

    def calculate(x, ids):
        # Preserve float64 for tests; use float32 numerics for bf16 training.
        x = x if x.dtype == torch.float64 else x.float()
        selected = x.gather(-1, ids).mean(-1)
        return selected - torch.logsumexp(x, dim=-1)

    parts = []
    for indices in positions.split(chunk_size):
        x = logits.index_select(0, indices)
        ids = shifted.index_select(0, indices)
        if torch.is_grad_enabled() and x.requires_grad:
            value = checkpoint(
                calculate, x, ids,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            value = calculate(x, ids)
        parts.append(value)

    values = torch.cat(parts)
    result = torch.zeros(
        logits.shape[0], device=logits.device, dtype=values.dtype
    )
    return result.index_copy(0, positions, values)
