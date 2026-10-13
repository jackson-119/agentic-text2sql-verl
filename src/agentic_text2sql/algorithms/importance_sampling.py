# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Narrow token-level TIS reference adapted from verl.

Implements the historical token / upper-cap / no-normalization setting.
Consumes a supplied log-ratio; does not establish probability semantics.
"""
import math
import torch

SAFETY_BOUND = 20.0


def token_truncated_is_weights(log_ratio, response_mask, upper_bound=2.0):
    """Return detached token weights; no batch normalization or rejection."""
    if log_ratio.ndim != 2 or log_ratio.shape != response_mask.shape:
        raise ValueError("Expected matching [batch, tokens] tensors.")
    if not log_ratio.is_floating_point():
        raise TypeError("log_ratio must have a floating dtype.")
    if log_ratio.device != response_mask.device:
        raise ValueError("log_ratio and response_mask must share a device.")
    if not torch.isfinite(log_ratio).all():
        raise ValueError("All log-ratios must be finite, including masked positions.")
    if not ((response_mask == 0) | (response_mask == 1)).all():
        raise ValueError("response_mask must be binary.")
    if isinstance(upper_bound, bool) or not isinstance(upper_bound, (int, float)):
        raise TypeError("upper_bound must be a real scalar, not bool.")
    if not math.isfinite(upper_bound) or upper_bound <= 0:
        raise ValueError("upper_bound must be finite and positive.")

    raw_weights = log_ratio.clamp(
        min=-SAFETY_BOUND, max=SAFETY_BOUND
    ).exp()
    raw_weights = raw_weights * response_mask
    return raw_weights.clamp(max=upper_bound).detach()
