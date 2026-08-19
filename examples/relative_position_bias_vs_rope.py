# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A minimal comparison of T5-style relative bias and RoPE."""

import math

import torch
from torch import Tensor, nn

SEQUENCE_LENGTH = 4
NUM_HEADS = 2
HEAD_DIMENSION = 4
MAX_RELATIVE_DISTANCE = SEQUENCE_LENGTH - 1


def apply_rope(x: Tensor, positions: Tensor) -> Tensor:
    """Rotate each adjacent pair of query or key features."""
    dimension = x.shape[-1]
    frequencies = 1.0 / (10_000 ** (torch.arange(0, dimension, 2, dtype=x.dtype) / dimension))
    angles = positions[:, None] * frequencies[None, :]
    cosines, sines = angles.cos()[None], angles.sin()[None]

    pairs = x.unflatten(-1, (dimension // 2, 2))
    even, odd = pairs[..., 0], pairs[..., 1]
    return torch.stack(
        (even * cosines - odd * sines, even * sines + odd * cosines), dim=-1
    ).flatten(-2)


def attention(queries: Tensor, keys: Tensor, bias: Tensor | None = None) -> Tensor:
    """Return attention probabilities."""
    logits = queries @ keys.transpose(-1, -2) / math.sqrt(HEAD_DIMENSION)
    if bias is not None:
        logits = logits + bias
    return logits.softmax(dim=-1)


def main() -> None:
    """Run both position mechanisms on the same queries and keys."""
    torch.manual_seed(0)
    torch.set_printoptions(precision=3, sci_mode=False)

    positions = torch.arange(SEQUENCE_LENGTH)
    queries = torch.randn(NUM_HEADS, SEQUENCE_LENGTH, HEAD_DIMENSION)
    keys = torch.randn(NUM_HEADS, SEQUENCE_LENGTH, HEAD_DIMENSION)

    # T5: look up one learned scalar per head and relative-distance bucket,
    # then add it to the attention logits. Real T5 groups far distances into
    # logarithmic buckets; exact clipped distances keep this example short.
    relative_distance = positions[None, :] - positions[:, None]
    bucket_ids = (
        relative_distance.clamp(-MAX_RELATIVE_DISTANCE, MAX_RELATIVE_DISTANCE)
        + MAX_RELATIVE_DISTANCE
    )
    bias_table = nn.Embedding(2 * MAX_RELATIVE_DISTANCE + 1, NUM_HEADS)
    t5_bias = bias_table(bucket_ids).permute(2, 0, 1)

    no_position = attention(queries, keys)
    t5_attention = attention(queries, keys, t5_bias)
    rope_attention = attention(apply_rope(queries, positions), apply_rope(keys, positions))

    print("Relative distances [query, key]:\n", relative_distance)
    print("\nT5 bias added to head 0 logits:\n", t5_bias[0].detach())
    print("\nHead 0 attention for the last query:")
    print("None:", no_position[0, -1])
    print("T5: ", t5_attention[0, -1].detach())
    print("RoPE:", rope_attention[0, -1])


if __name__ == "__main__":
    main()
