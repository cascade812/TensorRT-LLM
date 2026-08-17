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
"""Tests for the experimental DSA per-layer host-cache transfer operators."""

import torch

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.attention_backend.sparse.dsa.kv_offload_prototype import (
    _get_offloaded_layer_indices,
    _select_full_layers,
)

_NUM_BLOCKS = 2
_NUM_LAYERS = 5
_TOKENS_PER_BLOCK = 4
_LAYER_BYTES = 16
_HOST_ROW_BYTES = 3 * _LAYER_BYTES
_STRIDE_FACTOR = _NUM_LAYERS * _TOKENS_PER_BLOCK
_FULL_LAYER = 1
_SHARED_LAYERS = (2, 3, 4)


def _global_index(block: int, layer: int, token: int) -> int:
    return block * _STRIDE_FACTOR + layer * _TOKENS_PER_BLOCK + token


def _make_source() -> torch.Tensor:
    num_rows = _NUM_BLOCKS * _STRIDE_FACTOR
    values = torch.arange(num_rows * _LAYER_BYTES, dtype=torch.int64)
    return (values % 251).to(torch.uint8).reshape(num_rows, _LAYER_BYTES).cuda()


def _mirror_group(
    source: torch.Tensor,
    host_pool: torch.Tensor,
    coordinates,
) -> None:
    for layer_in_group, layer in enumerate(_SHARED_LAYERS):
        indices = torch.tensor(
            [_global_index(block, layer, token) for block, token in coordinates],
            dtype=torch.int32,
            device="cuda",
        )
        torch.ops.trtllm.dsa_kv_cache_offload_mirror(
            source,
            indices,
            host_pool,
            _STRIDE_FACTOR,
            _TOKENS_PER_BLOCK,
            layer,
            layer_in_group,
        )


def _expected_layer_rows(
    source: torch.Tensor,
    coordinates,
    layer: int,
) -> torch.Tensor:
    rows = [source[_global_index(block, layer, token)] for block, token in coordinates]
    return torch.stack(rows)


def test_dsa_kv_cache_offload_mirror_and_gather():
    source = _make_source()
    host_pool = torch.zeros(
        (_NUM_BLOCKS * _TOKENS_PER_BLOCK, _HOST_ROW_BYTES),
        dtype=torch.uint8,
        pin_memory=True,
    )
    coordinates = [(0, 1), (1, 3)]
    _mirror_group(source, host_pool, coordinates)

    full_indices = torch.tensor(
        [_global_index(block, _FULL_LAYER, token) for block, token in coordinates] + [-1],
        dtype=torch.int32,
        device="cuda",
    )
    for layer_in_group, layer in enumerate(_SHARED_LAYERS):
        output = torch.empty(
            (len(coordinates) + 1, _LAYER_BYTES),
            dtype=torch.uint8,
            device="cuda",
        )
        torch.ops.trtllm.dsa_kv_cache_offload_gather(
            host_pool,
            full_indices,
            output,
            _STRIDE_FACTOR,
            _TOKENS_PER_BLOCK,
            _FULL_LAYER,
            layer_in_group,
        )

        expected = _expected_layer_rows(source, coordinates, layer)
        torch.testing.assert_close(output[:-1], expected)
        torch.testing.assert_close(output[-1], torch.zeros_like(output[-1]))


def test_dsa_kv_cache_offload_gather_cuda_graph_replay():
    source = _make_source()
    host_pool = torch.zeros(
        (_NUM_BLOCKS * _TOKENS_PER_BLOCK, _HOST_ROW_BYTES),
        dtype=torch.uint8,
        pin_memory=True,
    )
    coordinates = [(0, 1), (1, 3)]
    _mirror_group(source, host_pool, coordinates)
    torch.cuda.synchronize()

    static_indices = torch.full((1,), -1, dtype=torch.int32, device="cuda")
    output = torch.empty(
        (len(_SHARED_LAYERS), 1, _LAYER_BYTES),
        dtype=torch.uint8,
        device="cuda",
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for layer_in_group in range(len(_SHARED_LAYERS)):
            torch.ops.trtllm.dsa_kv_cache_offload_gather(
                host_pool,
                static_indices,
                output[layer_in_group],
                _STRIDE_FACTOR,
                _TOKENS_PER_BLOCK,
                _FULL_LAYER,
                layer_in_group,
            )

    for coordinates_index, (block, token) in enumerate(coordinates):
        static_indices.fill_(_global_index(block, _FULL_LAYER, token))
        graph.replay()
        torch.cuda.synchronize()
        for layer_in_group, layer in enumerate(_SHARED_LAYERS):
            expected = _expected_layer_rows(source, coordinates, layer)[coordinates_index]
            torch.testing.assert_close(output[layer_in_group, 0], expected)


class _FakeSparseParams:

    def __init__(self, is_full_indexer_layer: bool):
        self.is_full_indexer_layer = is_full_indexer_layer


class _FakeSparseAttentionConfig:

    def to_sparse_params(self, pretrained_config, layer_idx):
        del pretrained_config
        is_full = layer_idx < 2 or layer_idx % 4 == 2
        return _FakeSparseParams(is_full)


class _FakeCacheManager:
    layer_offsets = dict(enumerate(range(78)))


def test_dsa_kv_cache_offload_select_all_complete_groups():
    full_layers = _select_full_layers(
        "all",
        _FakeCacheManager(),
        _FakeSparseAttentionConfig(),
        pretrained_config=object(),
    )

    assert full_layers == tuple(range(2, 78, 4))
    assert len(full_layers) == 19



def test_dsa_kv_cache_offload_select_group_layers(monkeypatch):
    monkeypatch.setenv(
        "TRTLLM_DSA_KV_OFFLOAD_PROTOTYPE_GROUP_LAYERS",
        "3,4",
    )

    assert _get_offloaded_layer_indices() == (1, 2)
