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
"""Tests for the experimental DSA host-cache transfer operators."""

import torch

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.attention_backend.sparse.dsa.kv_offload_prototype import (
    _get_offloaded_layer_indices,
    _select_full_layers,
    configure_cache_manager,
)
from tensorrt_llm.bindings import DataType

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


def _make_host_state():
    host_rows = _NUM_BLOCKS * _TOKENS_PER_BLOCK
    host_pool = torch.zeros(
        (host_rows, _HOST_ROW_BYTES),
        dtype=torch.uint8,
        pin_memory=True,
    )
    host_versions = torch.zeros(
        (host_rows, len(_SHARED_LAYERS)),
        dtype=torch.int32,
        device="cuda",
    )
    return host_pool, host_versions


def _mirror_group(
    source: torch.Tensor,
    host_pool: torch.Tensor,
    host_versions: torch.Tensor,
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
            host_versions,
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


def _make_working_set(max_requests: int, rows_per_request: int):
    cache_shape = (max_requests, 2 * rows_per_request)
    return {
        "keys": torch.full(cache_shape, -1, dtype=torch.int32, device="cuda"),
        "versions": torch.full(cache_shape, -1, dtype=torch.int32, device="cuda"),
        "row_to_slot": torch.full(
            (_NUM_BLOCKS * _TOKENS_PER_BLOCK,),
            -1,
            dtype=torch.int32,
            device="cuda",
        ),
        "epoch": torch.full((1,), -1, dtype=torch.int32, device="cuda"),
        "slot_epochs": torch.full(
            cache_shape,
            -1,
            dtype=torch.int32,
            device="cuda",
        ),
        "free_slots": torch.empty(cache_shape, dtype=torch.int32, device="cuda"),
        "free_counts": torch.zeros(
            (max_requests,), dtype=torch.int32, device="cuda"
        ),
        "allocation_counts": torch.zeros(
            (max_requests,), dtype=torch.int32, device="cuda"
        ),
        "values": torch.empty(
            (*cache_shape, _LAYER_BYTES),
            dtype=torch.uint8,
            device="cuda",
        ),
        "slots": torch.empty(
            (max_requests * rows_per_request,),
            dtype=torch.int32,
            device="cuda",
        ),
        "miss_count": torch.zeros((1,), dtype=torch.int32, device="cuda"),
    }


def _run_incremental(
    host_pool: torch.Tensor,
    host_versions: torch.Tensor,
    working_set,
    coordinates_by_request,
    layer_in_group: int = 0,
):
    rows_per_request = len(coordinates_by_request[0])
    coordinates = [coordinate for request in coordinates_by_request for coordinate in request]
    indices = torch.tensor(
        [
            _global_index(block, _FULL_LAYER, token)
            for block, token in coordinates
        ],
        dtype=torch.int32,
        device="cuda",
    )
    torch.ops.trtllm.dsa_kv_cache_offload_incremental_gather(
        host_pool,
        indices,
        host_versions,
        working_set["keys"],
        working_set["versions"],
        working_set["row_to_slot"],
        working_set["epoch"],
        working_set["slot_epochs"],
        working_set["free_slots"],
        working_set["free_counts"],
        working_set["allocation_counts"],
        working_set["values"],
        working_set["slots"],
        working_set["miss_count"],
        rows_per_request,
        _STRIDE_FACTOR,
        _TOKENS_PER_BLOCK,
        _FULL_LAYER,
        layer_in_group,
    )
    return coordinates


def _assert_working_set_values(source, working_set, coordinates, layer):
    flat_values = working_set["values"].reshape(-1, _LAYER_BYTES)
    slots = working_set["slots"][: len(coordinates)].tolist()
    actual = torch.stack([flat_values[slot] for slot in slots])
    torch.testing.assert_close(actual, _expected_layer_rows(source, coordinates, layer))


def test_dsa_kv_cache_offload_mirror_and_gather():
    source = _make_source()
    host_pool, host_versions = _make_host_state()
    coordinates = [(0, 1), (1, 3)]
    _mirror_group(source, host_pool, host_versions, coordinates)

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
        assert host_versions[:, layer_in_group].sum().item() == len(coordinates)


def test_dsa_kv_cache_offload_gather_cuda_graph_replay():
    source = _make_source()
    host_pool, host_versions = _make_host_state()
    coordinates = [(0, 1), (1, 3)]
    _mirror_group(source, host_pool, host_versions, coordinates)
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


def test_dsa_kv_cache_offload_incremental_hits_misses_and_invalidation():
    source = _make_source()
    host_pool, host_versions = _make_host_state()
    all_coordinates = [
        (block, token)
        for block in range(_NUM_BLOCKS)
        for token in range(_TOKENS_PER_BLOCK)
    ]
    _mirror_group(source, host_pool, host_versions, all_coordinates)
    working_set = _make_working_set(max_requests=2, rows_per_request=2)

    first = [[(0, 0), (0, 1)], [(1, 0), (1, 1)]]
    coordinates = _run_incremental(host_pool, host_versions, working_set, first)
    torch.cuda.synchronize()
    assert working_set["miss_count"].item() == 4
    first_slots = working_set["slots"][:4].tolist()
    _assert_working_set_values(source, working_set, coordinates, _SHARED_LAYERS[0])

    second = [[(0, 1), (0, 2)], [(1, 0), (1, 3)]]
    coordinates = _run_incremental(host_pool, host_versions, working_set, second)
    torch.cuda.synchronize()
    assert working_set["miss_count"].item() == 2
    second_slots = working_set["slots"][:4].tolist()
    assert second_slots[0] == first_slots[1]
    assert second_slots[2] == first_slots[2]
    _assert_working_set_values(source, working_set, coordinates, _SHARED_LAYERS[0])

    coordinates = _run_incremental(host_pool, host_versions, working_set, second)
    torch.cuda.synchronize()
    assert working_set["miss_count"].item() == 0
    assert working_set["slots"][:4].tolist() == second_slots
    _assert_working_set_values(source, working_set, coordinates, _SHARED_LAYERS[0])

    updated_coordinate = (0, 1)
    block, token = updated_coordinate
    updated_source_row = _global_index(block, _SHARED_LAYERS[0], token)
    source[updated_source_row].fill_(173)
    _mirror_group(source, host_pool, host_versions, [updated_coordinate])
    coordinates = _run_incremental(host_pool, host_versions, working_set, second)
    torch.cuda.synchronize()
    assert working_set["miss_count"].item() == 1
    _assert_working_set_values(source, working_set, coordinates, _SHARED_LAYERS[0])


def test_dsa_kv_cache_offload_incremental_cuda_graph_replay():
    source = _make_source()
    host_pool, host_versions = _make_host_state()
    coordinates = [(0, 0), (0, 1), (0, 2)]
    _mirror_group(source, host_pool, host_versions, coordinates)
    torch.cuda.synchronize()

    rows_per_request = 2
    working_set = _make_working_set(max_requests=1, rows_per_request=rows_per_request)
    static_indices = torch.tensor(
        [
            _global_index(0, _FULL_LAYER, 0),
            _global_index(0, _FULL_LAYER, 1),
        ],
        dtype=torch.int32,
        device="cuda",
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        torch.ops.trtllm.dsa_kv_cache_offload_incremental_gather(
            host_pool,
            static_indices,
            host_versions,
            working_set["keys"],
            working_set["versions"],
            working_set["row_to_slot"],
            working_set["epoch"],
            working_set["slot_epochs"],
            working_set["free_slots"],
            working_set["free_counts"],
            working_set["allocation_counts"],
            working_set["values"],
            working_set["slots"],
            working_set["miss_count"],
            rows_per_request,
            _STRIDE_FACTOR,
            _TOKENS_PER_BLOCK,
            _FULL_LAYER,
            0,
        )

    working_set["keys"].fill_(-1)
    working_set["versions"].fill_(-1)
    working_set["row_to_slot"].fill_(-1)
    working_set["epoch"].fill_(-1)
    working_set["slot_epochs"].fill_(-1)
    graph.replay()
    torch.cuda.synchronize()
    assert working_set["miss_count"].item() == 2
    _assert_working_set_values(source, working_set, coordinates[:2], _SHARED_LAYERS[0])

    graph.replay()
    torch.cuda.synchronize()
    assert working_set["miss_count"].item() == 0

    static_indices[1] = _global_index(0, _FULL_LAYER, 2)
    graph.replay()
    torch.cuda.synchronize()
    assert working_set["miss_count"].item() == 1
    _assert_working_set_values(
        source,
        working_set,
        [coordinates[0], coordinates[2]],
        _SHARED_LAYERS[0],
    )


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


class _FakeIncrementalCacheManager:
    layer_offsets = dict(enumerate(range(4)))
    is_estimating_kv_cache = False
    spec_config = None
    dtype = DataType.FP8
    head_dim = 576
    blocks_in_primary_pool = 2
    tokens_per_block = 4
    max_batch_size = 2


class _FakeIncrementalSparseParams:

    def __init__(self, is_full_indexer_layer: bool):
        self.is_full_indexer_layer = is_full_indexer_layer
        self.index_topk = 2


class _FakeIncrementalSparseAttentionConfig:

    def to_sparse_params(self, pretrained_config, layer_idx=None):
        del pretrained_config
        return _FakeIncrementalSparseParams(layer_idx in (None, 0))


def test_dsa_kv_cache_offload_configure_incremental_working_set(monkeypatch):
    monkeypatch.setenv("TRTLLM_DSA_KV_OFFLOAD_PROTOTYPE_FULL_LAYER", "0")
    monkeypatch.setenv("TRTLLM_DSA_KV_OFFLOAD_PROTOTYPE_INCREMENTAL", "1")
    cache_manager = _FakeIncrementalCacheManager()

    configure_cache_manager(
        cache_manager,
        _FakeIncrementalSparseAttentionConfig(),
        pretrained_config=object(),
    )

    group = cache_manager.dsa_kv_offload_groups[0]
    assert group.host_pool.shape == (8, 3 * 576)
    assert group.host_versions.shape == (8, 3)
    assert group.working_set is not None
    assert group.working_set.keys.shape == (3, 2, 4)
    assert group.working_set.row_to_slot.shape == (3, 8)
    assert group.working_set.slot_epochs.shape == (3, 2, 4)
    assert group.working_set.free_slots.shape == (3, 2, 4)
    assert group.working_set.free_counts.shape == (3, 2)
    assert group.working_set.allocation_counts.shape == (3, 2)
    assert group.working_set.values.shape == (3, 2, 4, 576)
