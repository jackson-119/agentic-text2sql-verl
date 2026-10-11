"""Turn-aware credit assignment for multi-turn GRPO.

The terminal correctness advantage is broadcast to every LLM-generated
token. Process and format signals remain local to their corresponding
assistant turn.

Tool observations and padding are excluded through ``response_mask``.

This module is intentionally independent from the reward manager and
trainer wiring so its invariants can be tested before online training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Optional, Sequence, Union

import torch


GroupIds = Union[
    torch.Tensor,
    Sequence[Hashable],
]


@dataclass
class TurnAwareCreditOutput:
    """Components of token-level turn-aware advantages."""

    advantages: torch.Tensor
    terminal_advantages: torch.Tensor
    process_advantages: torch.Tensor
    format_advantages: torch.Tensor
    turn_ids: torch.Tensor


def _validate_matrix(
    value: torch.Tensor,
    name: str,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(
            "{} must be a torch.Tensor".format(name)
        )

    if value.ndim != 2:
        raise ValueError(
            "{} must have shape [batch, tokens]".format(
                name
            )
        )


def infer_assistant_turn_ids(
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Recover assistant turns from contiguous action-token regions.

    ``response_mask == 1`` means an LLM-generated token.
    ``response_mask == 0`` means a tool observation or padding.

    Returned IDs use:

    - 0 for tool observations and padding;
    - 1 for the first assistant turn;
    - 2 for the second assistant turn;
    - and so on.
    """

    _validate_matrix(
        response_mask,
        "response_mask",
    )

    action_mask = response_mask.bool()

    previous_is_action = torch.zeros_like(
        action_mask
    )
    previous_is_action[:, 1:] = action_mask[:, :-1]

    turn_starts = (
        action_mask
        & ~previous_is_action
    )

    turn_ids = torch.cumsum(
        turn_starts.to(torch.long),
        dim=-1,
    )

    return torch.where(
        action_mask,
        turn_ids,
        torch.zeros_like(turn_ids),
    )


def _normalize_group_key(value):
    if isinstance(value, list):
        return tuple(value)

    if isinstance(value, dict):
        return tuple(
            sorted(value.items())
        )

    try:
        hash(value)
        return value
    except TypeError:
        return repr(value)


def _group_id_list(
    group_ids: GroupIds,
    batch_size: int,
):
    if isinstance(group_ids, torch.Tensor):
        if group_ids.numel() != batch_size:
            raise ValueError(
                "group_ids length does not match batch"
            )

        raw_ids = (
            group_ids.detach()
            .cpu()
            .reshape(-1)
            .tolist()
        )
    else:
        raw_ids = list(group_ids)

        if len(raw_ids) != batch_size:
            raise ValueError(
                "group_ids length does not match batch"
            )

    return [
        _normalize_group_key(value)
        for value in raw_ids
    ]


def center_values_by_group(
    values: torch.Tensor,
    group_ids: GroupIds,
    eligible: Optional[torch.Tensor] = None,
    normalize_by_std: bool = False,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Center one scalar per trajectory within each prompt group.

    When ``eligible`` is supplied, only eligible trajectories
    participate in the baseline and all ineligible outputs remain zero.
    """

    if values.ndim != 1:
        raise ValueError(
            "values must have shape [batch]"
        )

    batch_size = values.shape[0]
    ids = _group_id_list(
        group_ids,
        batch_size,
    )

    if eligible is None:
        eligible_mask = torch.ones(
            batch_size,
            dtype=torch.bool,
            device=values.device,
        )
    else:
        if eligible.shape != values.shape:
            raise ValueError(
                "eligible must match values shape"
            )

        eligible_mask = eligible.bool()

    groups = {}

    for index, group_id in enumerate(ids):
        groups.setdefault(
            group_id,
            [],
        ).append(index)

    centered = torch.zeros_like(values)

    for positions in groups.values():
        selected = [
            index
            for index in positions
            if bool(eligible_mask[index].item())
        ]

        if not selected:
            continue

        selected_tensor = torch.tensor(
            selected,
            dtype=torch.long,
            device=values.device,
        )

        selected_values = values.index_select(
            0,
            selected_tensor,
        )

        local_centered = (
            selected_values
            - selected_values.mean()
        )

        if normalize_by_std:
            standard_deviation = (
                selected_values.std(
                    unbiased=False
                )
            )

            if bool(
                (standard_deviation > eps).item()
            ):
                local_centered = (
                    local_centered
                    / (
                        standard_deviation
                        + eps
                    )
                )
            else:
                local_centered = torch.zeros_like(
                    local_centered
                )

        centered.index_copy_(
            0,
            selected_tensor,
            local_centered,
        )

    return centered


def _validate_turn_scores(
    scores: torch.Tensor,
    batch_size: int,
    name: str,
) -> None:
    if not isinstance(scores, torch.Tensor):
        raise TypeError(
            "{} must be a torch.Tensor".format(name)
        )

    if scores.ndim != 2:
        raise ValueError(
            "{} must have shape [batch, turns]".format(
                name
            )
        )

    if scores.shape[0] != batch_size:
        raise ValueError(
            "{} batch dimension does not match".format(
                name
            )
        )


def _turn_presence(
    turn_ids: torch.Tensor,
    turn_count: int,
) -> torch.Tensor:
    if turn_count == 0:
        return torch.zeros(
            (
                turn_ids.shape[0],
                0,
            ),
            dtype=torch.bool,
            device=turn_ids.device,
        )

    return torch.stack(
        [
            (turn_ids == turn_index).any(dim=-1)
            for turn_index in range(
                1,
                turn_count + 1,
            )
        ],
        dim=-1,
    )


def _gather_turn_values(
    turn_values: torch.Tensor,
    turn_ids: torch.Tensor,
) -> torch.Tensor:
    batch_size = turn_ids.shape[0]
    turn_count = turn_values.shape[1]

    if turn_count == 0:
        return torch.zeros_like(
            turn_ids,
            dtype=turn_values.dtype,
        )

    maximum_turn = int(
        turn_ids.max().item()
    )

    if maximum_turn > turn_count:
        raise ValueError(
            "turn score count {} is smaller than "
            "observed assistant turn count {}".format(
                turn_count,
                maximum_turn,
            )
        )

    zero_column = torch.zeros(
        (
            batch_size,
            1,
        ),
        dtype=turn_values.dtype,
        device=turn_values.device,
    )

    lookup = torch.cat(
        [
            zero_column,
            turn_values,
        ],
        dim=-1,
    )

    return torch.gather(
        lookup,
        dim=1,
        index=turn_ids,
    )


def compose_turn_aware_advantages(
    terminal_scores: torch.Tensor,
    response_mask: torch.Tensor,
    group_ids: GroupIds,
    process_turn_scores: Optional[
        torch.Tensor
    ] = None,
    format_turn_scores: Optional[
        torch.Tensor
    ] = None,
    execution_correct: Optional[
        torch.Tensor
    ] = None,
    process_coefficient: float = 1.0,
    format_coefficient: float = 1.0,
    normalize_terminal_by_std: bool = False,
    normalize_process_by_std: bool = False,
    format_penalty_only: bool = True,
    turn_ids: Optional[torch.Tensor] = None,
) -> TurnAwareCreditOutput:
    """Construct token-level turn-aware GRPO advantages.

    Terminal correctness:

    - centered within each prompt group;
    - broadcast to all assistant-generated tokens.

    Process score:

    - enabled only for execution-correct trajectories;
    - used directly without cross-trajectory centering or normalization;
    - accumulated backward over turns (gamma=1), then mapped to tokens.

    Format score:

    - applied only to its corresponding assistant turn;
    - by default, positive format bonuses are removed and only
      local format penalties remain.
    """

    _validate_matrix(
        response_mask,
        "response_mask",
    )

    batch_size, token_count = response_mask.shape

    if terminal_scores.ndim != 1:
        raise ValueError(
            "terminal_scores must have shape [batch]"
        )

    if terminal_scores.shape[0] != batch_size:
        raise ValueError(
            "terminal_scores batch dimension "
            "does not match response_mask"
        )

    if turn_ids is None:
        turn_ids = infer_assistant_turn_ids(
            response_mask
        )
    else:
        if turn_ids.shape != response_mask.shape:
            raise ValueError(
                "turn_ids must match response_mask shape"
            )

        turn_ids = turn_ids.to(
            dtype=torch.long,
            device=response_mask.device,
        )

    action_mask = response_mask.to(
        dtype=terminal_scores.dtype
    )

    terminal_sequence_advantage = (
        center_values_by_group(
            values=terminal_scores,
            group_ids=group_ids,
            eligible=None,
            normalize_by_std=(
                normalize_terminal_by_std
            ),
        )
    )

    terminal_token_advantage = (
        terminal_sequence_advantage.unsqueeze(-1)
        * action_mask
    )

    process_token_advantage = torch.zeros(
        (
            batch_size,
            token_count,
        ),
        dtype=terminal_scores.dtype,
        device=terminal_scores.device,
    )

    if process_turn_scores is not None:
        _validate_turn_scores(
            process_turn_scores,
            batch_size,
            "process_turn_scores",
        )

        process_turn_scores = (
            process_turn_scores.to(
                dtype=terminal_scores.dtype,
                device=terminal_scores.device,
            )
        )

        if execution_correct is None:
            correct_mask = (
                terminal_scores > 0
            )
        else:
            if execution_correct.shape != (
                terminal_scores.shape
            ):
                raise ValueError(
                    "execution_correct must match "
                    "terminal_scores shape"
                )

            correct_mask = (
                execution_correct.bool()
            )

        turn_count = (
            process_turn_scores.shape[1]
        )

        presence = _turn_presence(
            turn_ids,
            turn_count,
        )

        # TURN_CREDIT_DIRECT_PROCESS_V2
        # Keep the existing variable name for downstream gathering.
        # Local signals do not use any peer trajectory statistics.
        if normalize_process_by_std:
            raise ValueError(
                "Direct local credit requires normalize_process_by_std=False"
            )
        centered_process = torch.where(
            correct_mask.unsqueeze(-1) & presence,
            process_turn_scores,
            torch.zeros_like(process_turn_scores),
        )


        # TURN_CREDIT_CUMULATIVE_PROCESS_V1
        # gamma=1: sum current and future gated process rewards.
        cumulative_process = torch.flip(
            torch.cumsum(
                torch.flip(centered_process, dims=[1]), dim=1
            ),
            dims=[1],
        )
        process_token_advantage = (
            _gather_turn_values(
                cumulative_process,
                turn_ids,
            )
            * action_mask
            * float(process_coefficient)
        )

        process_token_advantage = torch.where(
            correct_mask.unsqueeze(-1),
            process_token_advantage,
            torch.zeros_like(
                process_token_advantage
            ),
        )

    format_token_advantage = torch.zeros(
        (
            batch_size,
            token_count,
        ),
        dtype=terminal_scores.dtype,
        device=terminal_scores.device,
    )

    if format_turn_scores is not None:
        _validate_turn_scores(
            format_turn_scores,
            batch_size,
            "format_turn_scores",
        )

        format_turn_scores = (
            format_turn_scores.to(
                dtype=terminal_scores.dtype,
                device=terminal_scores.device,
            )
        )

        if format_penalty_only:
            format_turn_scores = torch.clamp(
                format_turn_scores,
                max=0.0,
            )

        format_token_advantage = (
            _gather_turn_values(
                format_turn_scores,
                turn_ids,
            )
            * action_mask
            * float(format_coefficient)
        )

    advantages = (
        terminal_token_advantage
        + process_token_advantage
        + format_token_advantage
    )

    advantages = (
        advantages
        * action_mask
    )

    return TurnAwareCreditOutput(
        advantages=advantages,
        terminal_advantages=(
            terminal_token_advantage
        ),
        process_advantages=(
            process_token_advantage
        ),
        format_advantages=(
            format_token_advantage
        ),
        turn_ids=turn_ids,
    )


def select_advantages(
    enabled: bool,
    legacy_advantages: torch.Tensor,
    turn_aware_output: Optional[
        TurnAwareCreditOutput
    ] = None,
) -> torch.Tensor:
    """Preserve the original GRPO path when disabled."""

    if not enabled:
        return legacy_advantages

    if turn_aware_output is None:
        raise ValueError(
            "turn_aware_output is required "
            "when enabled=True"
        )

    return turn_aware_output.advantages


def sequence_mean_token_mean(
    loss_matrix: torch.Tensor,
    loss_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Average tokens inside each sequence, then average sequences."""

    _validate_matrix(
        loss_matrix,
        "loss_matrix",
    )
    _validate_matrix(
        loss_mask,
        "loss_mask",
    )

    if loss_matrix.shape != loss_mask.shape:
        raise ValueError(
            "loss_matrix and loss_mask must match"
        )

    mask = loss_mask.to(
        dtype=loss_matrix.dtype
    )

    token_counts = mask.sum(dim=-1)
    valid_sequences = token_counts > 0

    if not bool(valid_sequences.any().item()):
        return loss_matrix.sum() * 0.0

    per_sequence = (
        (loss_matrix * mask).sum(dim=-1)
        / (
            token_counts
            + eps
        )
    )

    return per_sequence[
        valid_sequences
    ].mean()
